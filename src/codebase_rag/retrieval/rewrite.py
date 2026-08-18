"""Pre-retrieval query rewriting: expand a terse query with likely identifiers.

Reranking buys ordering, not coverage, so the questions that stay unanswered
fail before any reranker sees them: they fail on vocabulary. A query asking
about "dependencies" cannot be reranked into a file that says
`find_package(Eigen3)`. This stage is the only one that can reach those, by
asking the local model for the symbol and identifier names a codebase would
actually use for the query, and appending them to the original.

The expansion only ever adds terms. The retriever runs on
`original + " " + identifiers`, so a bad expansion can add noise but can never
lose the user's own words, and a model failure or timeout falls back to the
original query unchanged rather than raising. Model reasoning is deliberately
not enabled: it was measured at roughly 20x the latency on a comparable ranking
task, which is too slow for the shipped retrieval path.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from langchain_core.documents import Document

from .retriever_protocol import RetrieverProtocol

logger = logging.getLogger(__name__)

# Seconds to wait for the local model's expansion before giving up and
# retrieving on the original query. Kept short because this is on the critical
# path before any retrieval happens; a slow rewrite is worse than none.
DEFAULT_REWRITE_TIMEOUT_S = 5.0

# How many expansions may run at once. One keeps load on the local model exactly
# as a single query would produce it; a query that cannot get a slot skips the
# expansion immediately rather than queueing behind another's and paying the
# timeout to receive nothing. Higher values trade that prompt fallback for more
# expansions, moving the contention into the model server rather than removing it.
DEFAULT_REWRITE_MAX_CONCURRENCY = 1

# Cap on how many terms the expansion may add. The model is asked for
# identifiers, not prose, but nothing stops a chatty model returning a
# paragraph; appending that to a keyword query dilutes it rather than sharpening
# it. Keeping only the first N whitespace-separated tokens bounds the damage.
MAX_EXPANSION_TERMS = 32

_REWRITE_PROMPT = (
    "You expand a codebase search query with likely identifiers so keyword "
    "retrieval finds the right files. Given the query below, list the function "
    "names, class names, symbols, and technical terms a codebase would use for "
    "it. Answer with only the terms, space-separated, no explanation.\n\n"
    "Query: {query}\n\nTerms: "
)


class RewritingRetriever:
    """Wrap a retriever with a local-model query-expansion step.

    Satisfies `RetrieverProtocol` (exposes `search(query, k)`), so it stands in
    anywhere a retriever is expected and can be composed with the rerank stage.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: Any,
        timeout_s: float = DEFAULT_REWRITE_TIMEOUT_S,
        max_concurrency: int = DEFAULT_REWRITE_MAX_CONCURRENCY,
    ) -> None:
        """Initialize the rewrite stage.

        Args:
            retriever: The retriever the expanded query feeds.
            llm: Local model client exposing `invoke(prompt) -> str`.
            timeout_s: Seconds to wait for an expansion before falling back.
            max_concurrency: How many expansions this stage may run at once. The
                admission semaphore and the worker pool are both sized from this one
                value, so a slot always maps to a free worker and a query's timeout can
                only be measuring the model, never the wait for a worker.

        Raises:
            ValueError: If `max_concurrency` is below 1. `Config.get_instance` rejects
                that at load time, but a `Config` built directly bypasses it, and a
                `ThreadPoolExecutor(max_workers=0)` would then fail with a generic error
                that does not name the setting.
        """
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self.retriever = retriever
        self.llm = llm
        self.timeout_s = timeout_s
        self._max_concurrency = max_concurrency
        # The admission gate and the worker pool are the same limit expressed twice, on this
        # one instance. Each host composes a single instance in normal operation, so this is
        # the host's bound; a rebuild swaps in a fresh one while an in-flight request still
        # holds the old, whose pool is shut down and so only falls back rather than expanding.
        # Bounded so an over-release (a third release path added by mistake) raises instead of
        # silently admitting more than max_concurrency expansions.
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)

    def _expand(self, query: str) -> str:
        """Return `query` plus model-suggested identifiers, or `query` on any failure.

        The model call is run under a timeout; a timeout, an empty response, or
        any exception falls back to the original query and is logged, not raised.
        """
        # Decide admission before submitting. A query that cannot get a slot skips the
        # expansion immediately rather than queueing behind another query's expansion and
        # paying the full timeout to receive nothing.
        if not self._slots.acquire(blocking=False):
            logger.info(
                "Query rewrite skipped, %d expansion(s) already in flight; using original query",
                self._max_concurrency,
            )
            return query
        try:
            future = self._executor.submit(self.llm.invoke, _REWRITE_PROMPT.format(query=query))
        except Exception as e:
            # Submitting fails once the pool is shut down, which happens when the stack is
            # rebuilt after an ingest while a request still holds the old one. The slot was
            # never handed to a worker, so release it here rather than in the done callback.
            self._slots.release()
            logger.warning("Query rewrite unavailable (%s); using original query", e)
            return query

        # The worker releases the slot when the call actually finishes, never the caller. A
        # caller that timed out and released on its way out would hand the next query a slot
        # while this worker is still blocked in llm.invoke, restoring the queue wait in a form
        # that only appears under load. The callback fires on success, on a model error, and
        # on a cancelled-but-still-queued call alike, so the slot is always released exactly once.
        future.add_done_callback(lambda _future: self._slots.release())

        try:
            terms = str(future.result(timeout=self.timeout_s)).strip()
        except FutureTimeoutError:
            # Cancel rather than abandon. A still-queued expansion left in place would start
            # its own clock late and time out in turn, and the queue would grow without bound
            # while every query paid the full timeout. The slot is released by the done
            # callback when the call finishes, not here.
            future.cancel()
            logger.warning("Query rewrite timed out after %.1fs; using original query", self.timeout_s)
            return query
        except Exception as e:
            logger.warning("Query rewrite failed (%s); using original query", e)
            return query

        if not terms:
            return query
        # Bound the expansion: keep only the first MAX_EXPANSION_TERMS tokens so a chatty
        # model cannot append a paragraph to a keyword query.
        capped = " ".join(terms.split()[:MAX_EXPANSION_TERMS])
        expanded = f"{query} {capped}"
        logger.info("Expanded query '%s' -> '%s'", query, expanded)
        return expanded

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Expand the query, then retrieve on the expansion.

        Args:
            query: The user's original query.
            k: Number of results to return, passed through to the retriever.

        Returns:
            List of (document, score) tuples from the wrapped retriever.
        """
        return self.retriever.search(self._expand(query), k=k)

    def close(self) -> None:
        """Shut down the expansion thread pool.

        Called when the composed stack is rebuilt (e.g. after an ingest) so a
        worker thread is not leaked per rebuild. Idempotent: a second call after
        shutdown is a no-op.
        """
        self._executor.shutdown(wait=False)
