"""Cross-encoder reranking stage that wraps any retriever satisfying the protocol.

The stage sits between the configured first-stage retriever and the caller: it
asks the first stage for a deep candidate list, rescores those candidates with a
local cross-encoder, and returns the top `k` by cross-encoder score. It retrieves
nothing the first stage did not, so it can only reorder the candidate list. Note
that reordering a deep list into a shallow output can still *lose* coverage: a
document the first stage had inside the output window can be pushed out of it by
one the first stage ranked deeper.

Loading the model reaches the network on a cold cache. `CrossEncoder` resolves
the name through the HuggingFace hub and downloads roughly 2GB the first time,
inside the first `search()` call. After that it is served from the local cache.

`BAAI/bge-reranker-v2-m3` at candidate depth 50 was chosen by measurement, and
what it buys depends on where you measure. Scored offline at output depth 10 it
moved the shipped BM25 arm's MRR from 0.5830 to 0.7357 with hit rate flat at
0.8571. Measured end to end at the application's top_k of 5 it moved MRR from
0.6087 to 0.6833 but cost hit rate, 0.8333 down to 0.7857, and about 1.3s of
extra time to first token. Both defaults are off for that reason. Depth 50 rather
than 100 because the input list's recall is identical at both (0.9048), so the
deeper list contains no document the shallower one missed, at twice the latency.
`cross-encoder/ms-marco-MiniLM-L6-v2` was measured as unreliable and worse than
no reranker on this corpus; do not substitute it.

The reranker model is loaded lazily on first `search()` and cached on the
instance, so constructing the stage (which happens whether or not reranking is
enabled) pays nothing until a query actually runs through it.
"""

import logging
import threading
from typing import Any

from langchain_core.documents import Document

from .retriever_protocol import RetrieverProtocol

logger = logging.getLogger(__name__)

# Decided by measurement (see module docstring). The candidate depth the first
# stage is asked for before rescoring; the top-k the caller asked for is applied
# after rescoring.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CANDIDATE_DEPTH = 50

# Passage truncation before the cross-encoder scores it, matching the offline
# benchmark's default so the live stage rescores passages the same length the
# measured stage did. A chunk longer than this is scored on its head.
DEFAULT_MAX_PASSAGE_CHARS = 2000

# Output size when the caller passes no `k`. The protocol's `k=None` means "the
# retriever's own default"; the rerank stage's own default is a small top-k, not
# the full candidate list, so a `None` caller does not get 50 chunks handed to
# the prompt. Matches `BM25Retriever.DEFAULT_TOP_K`.
DEFAULT_TOP_K = 4


class RerankingRetriever:
    """Wrap a first-stage retriever with a local cross-encoder rerank stage.

    Structural typing: this satisfies `RetrieverProtocol` itself (it exposes
    `search(query, k)`), so it can stand in anywhere a retriever is expected,
    including in front of `BM25Retriever`, `VectorRetriever`, or
    `HybridRetriever` without any of them knowing they are being wrapped.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        model_name: str = DEFAULT_RERANK_MODEL,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        device: str | None = None,
    ) -> None:
        """Initialize the rerank stage.

        Args:
            retriever: The first-stage retriever whose candidates are rescored.
            model_name: Cross-encoder model. Defaults to the measured winner.
            candidate_depth: How many candidates to pull from the first stage
                before rescoring. Defaults to the measured depth.
            max_passage_chars: Passage truncation before scoring.
            device: Torch device for the cross-encoder. `None` lets
                sentence-transformers pick (MPS/CUDA/CPU as available).
        """
        self.retriever = retriever
        self.model_name = model_name
        self.candidate_depth = candidate_depth
        self.max_passage_chars = max_passage_chars
        self.device = device
        self._model: Any = None
        # Both hosts are multi-threaded: Starlette runs the sync /search endpoint in a
        # threadpool, and Streamlit gives each session its own thread over one shared runtime.
        self._model_lock = threading.Lock()

    def _load_model(self) -> Any:
        """Load the cross-encoder once, on first use, and cache it on the instance.

        Locked because an unguarded check-then-set lets two concurrent first
        queries each construct a model, which loads (and on a cold cache
        downloads) roughly 2GB twice and leaks one of them.
        """
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker model '%s'", self.model_name)
                    self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Retrieve a deep candidate list, rescore it, and return the top `k`.

        The first stage is asked for at least `candidate_depth` results: the
        whole point of the stage is to look deeper than the output depth and pull
        a better-ordered `k` out of it. A caller asking for more than that gets
        its own `k` honoured instead, because callers over-fetch deliberately
        (the search service multiplies `k` when a repo filter has to be applied
        after ranking), and quietly shrinking their request to `candidate_depth`
        would hand back fewer results than the same call returns with reranking
        off. When the first stage returns nothing, this returns nothing without
        loading the model.

        Args:
            query: The search query.
            k: Number of results to return after reranking. `None` uses
                `DEFAULT_TOP_K`, so a `None` caller gets a small top-k rather
                than the whole candidate list.

        Returns:
            List of (document, cross-encoder score) tuples, ordered by score.
        """
        fetch_depth = max(self.candidate_depth, k) if k is not None else self.candidate_depth
        candidates = self.retriever.search(query, k=fetch_depth)
        if not candidates:
            return []

        docs = [doc for doc, _ in candidates]
        model = self._load_model()
        pairs = [(query, doc.page_content[: self.max_passage_chars]) for doc in docs]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True)
        reranked = [(doc, float(score)) for doc, score in ranked]

        output_size = k if k is not None else DEFAULT_TOP_K
        logger.info("Reranked %d candidates for '%s', returning top %d", len(candidates), query, output_size)
        return reranked[:output_size]
