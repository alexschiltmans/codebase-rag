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
"""

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    """A retriever that combines vector and BM25 search results."""

    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: Any = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        min_score_threshold: float = 0.1,
        top_k: int = 5,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_retriever: The vector retriever.
            bm25_retriever: The BM25 retriever. If None, only vector search is used.
            vector_weight: Weight for vector search results (default: 0.7).
            bm25_weight: Weight for BM25 search results (default: 0.3).
            min_score_threshold: Minimum score for results to be returned (default: 0.1).
            top_k: Default number of results to return (default: 5).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.min_score_threshold = min_score_threshold
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents using both vector and BM25 search.

        Args:
            query: The search query.
            k: Number of top results to return (defaults to self.top_k).

        Returns:
            List of (document, score) tuples.
        """
        try:
            k_value = k if k is not None else self.top_k

            # Get vector search results (fetch extra for better reranking)
            vector_results = self.vector_retriever.search(query, k=k_value * 2)

            # Get BM25 search results (if BM25 retriever is available)
            bm25_results = self.bm25_retriever.search(query, k=k_value * 2) if self.bm25_retriever else []

            if not vector_results and not bm25_results:
                logger.warning("No results from either vector or BM25 search")
                return []

            # Normalize BM25 scores to [0, 1]
            if bm25_results:
                max_bm25_score = max(score for _, score in bm25_results)
                max_bm25_score = max(max_bm25_score, 1e-10)
                normalized_bm25 = [(doc, score / max_bm25_score) for doc, score in bm25_results]
            else:
                normalized_bm25 = []

            # Combine results by document identity
            doc_to_score: dict[str, dict] = {}

            for doc, score in vector_results:
                doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_index", ""))
                doc_to_score[doc_id] = {"doc": doc, "vector_score": score, "bm25_score": 0.0}

            for doc, score in normalized_bm25:
                doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_index", ""))
                if doc_id in doc_to_score:
                    doc_to_score[doc_id]["bm25_score"] = score
                else:
                    doc_to_score[doc_id] = {"doc": doc, "vector_score": 0.0, "bm25_score": score}

            # Calculate weighted combined scores
            results = []
            for scores in doc_to_score.values():
                combined_score = self.vector_weight * scores["vector_score"] + self.bm25_weight * scores["bm25_score"]
                if combined_score >= self.min_score_threshold:
                    results.append((scores["doc"], combined_score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k_value]

        except Exception as e:
            logger.error("Error in hybrid search: %s", str(e))
            return []

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve relevant documents using hybrid search.

        Implements the standard LangChain retriever interface.

        Args:
            query: The search query.

        Returns:
            List of relevant documents.
        """
        results = self.search(query)
        return [doc for doc, _ in results]

    def aget_relevant_documents(self, query: str) -> list[Document]:
        """Retrieve relevant documents using hybrid search (sync fallback).

        Args:
            query: The search query.

        Returns:
            List of relevant documents.
        """
        return self.get_relevant_documents(query)

    def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve documents using hybrid search.

        Args:
            query: The search query.
            **kwargs: Additional parameters (top_k supported).

        Returns:
            List of retrieved documents.
        """
        k = kwargs.get("top_k", self.top_k)
        results = self.search(query, k=k)
        return [doc for doc, _ in results]
