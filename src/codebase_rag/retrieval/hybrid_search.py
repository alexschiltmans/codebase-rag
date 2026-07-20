"""Hybrid search retriever that combines vector and keyword search.

This module implements a hybrid search approach that combines the strengths of
vector similarity search and traditional keyword search (BM25) for optimal retrieval.

Decision: Keep BM25-based hybrid search with a JSON-persisted index rather than
migrating to Qdrant sparse vectors. Rationale:
- At the scale of a few repositories, the BM25 corpus is small and fast to rebuild
- Qdrant sparse vector migration adds complexity for marginal benefit at this scale
- Hybrid search (vector + keyword) demonstrably improves codebase Q&A, especially for
  exact symbol/function name lookups where BM25 excels
- Each repo's documents are persisted as their own JSON corpus file, and the
  combined index is rebuilt from all of them on every ingest and on repo
  deletion, keeping it in sync with the vector store across every repo

Fusion uses Reciprocal Rank Fusion (RRF) rather than a weighted average of raw
scores: vector cosine similarity and BM25 scores live on unrelated scales, and
per-query max-normalizing BM25 (the previous approach) gives the top keyword
hit a fixed 1.0 regardless of how weak the match actually is. RRF fuses by
each ranker's rank order instead, so the two lists never need comparable
units.
"""

import logging

from langchain_core.documents import Document

from .retriever_protocol import RetrieverProtocol

logger = logging.getLogger(__name__)


class HybridRetriever:
    """A retriever that combines vector and BM25 search results."""

    def __init__(
        self,
        vector_retriever: RetrieverProtocol,
        bm25_retriever: RetrieverProtocol | None = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_retriever: The vector retriever.
            bm25_retriever: The BM25 retriever. If None, only vector search is used.
            vector_weight: Weight for the vector ranker's contribution (default: 0.7).
            bm25_weight: Weight for the BM25 ranker's contribution (default: 0.3).
            top_k: Default number of results to return (default: 5).
            rrf_k: Reciprocal Rank Fusion constant (default: 60, the standard value used
                by Elasticsearch and the original RRF paper — it dampens the influence of
                low ranks so results outside the top few contribute little).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents using both vector and BM25 search, fused via RRF.

        Each ranker's result list is combined by rank rather than by raw score, so the
        two lists never need to be on comparable scales (avoids weak keyword matches
        getting an inflated score purely from per-query max-normalization). Fused scores
        are rescaled to [0, 1] against the rankers that actually returned results
        (so a document topping every contributing ranker scores 1.0) and returned
        for display, but are not used to filter results here: after RRF a document
        ranked #1 by even one ranker always scores well above zero, so a fixed
        cutoff on the fused score can't distinguish relevant from irrelevant —
        relevance filtering happens upstream, on the vector retriever's raw cosine
        score (see `VectorRetriever.search`). Returns an empty list when both
        component retrievers return nothing.

        Args:
            query: The search query.
            k: Number of top results to return (defaults to self.top_k).

        Returns:
            List of (document, score) tuples.
        """
        k_value = k if k is not None else self.top_k

        # Get vector search results (fetch extra for better reranking)
        vector_results = self.vector_retriever.search(query, k=k_value * 2)

        # Get BM25 search results (if BM25 retriever is available)
        bm25_results = self.bm25_retriever.search(query, k=k_value * 2) if self.bm25_retriever is not None else []

        if not vector_results and not bm25_results:
            logger.warning("No results from either vector or BM25 search")
            return []

        # Reciprocal Rank Fusion: each ranker contributes weight / (rrf_k + rank),
        # using each list's own rank order rather than its raw score magnitude.
        doc_to_score: dict[str, dict] = {}

        def doc_id(doc: Document) -> str:
            return str(doc.metadata.get("source", "")) + str(doc.metadata.get("chunk_index", ""))

        for rank, (doc, _score) in enumerate(vector_results, start=1):
            entry = doc_to_score.setdefault(doc_id(doc), {"doc": doc, "rrf_score": 0.0})
            entry["rrf_score"] += self.vector_weight / (self.rrf_k + rank)

        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            entry = doc_to_score.setdefault(doc_id(doc), {"doc": doc, "rrf_score": 0.0})
            entry["rrf_score"] += self.bm25_weight / (self.rrf_k + rank)

        # Rescale so a document ranked #1 by every ranker that actually
        # contributed scores 1.0. This keys off the results, not off which
        # retriever objects exist: a configured-but-empty BM25 index (what
        # the app holds before its first ingest) contributes nothing, and
        # counting its weight here would cap every score at vector_weight.
        max_possible_score = 0.0
        if vector_results:
            max_possible_score += self.vector_weight / (self.rrf_k + 1)
        if bm25_results:
            max_possible_score += self.bm25_weight / (self.rrf_k + 1)

        results = [
            (entry["doc"], entry["rrf_score"] / max_possible_score if max_possible_score > 0 else 0.0)
            for entry in doc_to_score.values()
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k_value]
