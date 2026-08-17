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
    ) -> None:
        """Initialize the rewrite stage.

        Args:
            retriever: The retriever the expanded query feeds.
            llm: Local model client exposing `invoke(prompt) -> str`.
            timeout_s: Seconds to wait for an expansion before falling back.
        """
        self.retriever = retriever
        self.llm = llm
        self.timeout_s = timeout_s
        # A single worker is enough: rewriting runs one expansion per query, and
        # the executor exists only to bound that one call with a timeout.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _expand(self, query: str) -> str:
        """Return `query` plus model-suggested identifiers, or `query` on any failure.

        The model call is run under a timeout; a timeout, an empty response, or
        any exception falls back to the original query and is logged, not raised.
        """
        try:
            future = self._executor.submit(self.llm.invoke, _REWRITE_PROMPT.format(query=query))
        except Exception as e:
            # Submitting fails once the pool is shut down, which happens when the stack is
            # rebuilt after an ingest while a request still holds the old one.
            logger.warning("Query rewrite unavailable (%s); using original query", e)
            return query

        try:
            terms = str(future.result(timeout=self.timeout_s)).strip()
        except FutureTimeoutError:
            # Cancel rather than abandon. The pool has one worker, so a still-queued expansion
            # left in place would start its own clock late and time out in turn, and the queue
            # would grow without bound while every query paid the full timeout.
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
