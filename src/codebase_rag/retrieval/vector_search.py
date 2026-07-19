"""Vector similarity search retriever.

This module implements vector-based document retrieval, searching for documents
by computing the similarity between query and document vectors.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from ..database.vector_store_protocol import VectorStoreProtocol

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Vector-based document retriever.

    This retriever searches for documents by computing vector similarity between
    the query and document embeddings.

    Implements the Strategy pattern to allow different vector store backends.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize the vector retriever.

        Args:
            vector_store: The vector store to search (any VectorStoreProtocol implementation).
            score_threshold: Optional minimum similarity score threshold.
        """
        self.vector_store = vector_store
        self.score_threshold = score_threshold

        logger.info("Initialized VectorRetriever with %s", vector_store.__class__.__name__)
        if score_threshold:
            logger.info("Using score threshold: %s", score_threshold)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search for documents similar to the query.

        Args:
            query: The search query.
            k: Number of documents to retrieve.

        Returns:
            List of (document, score) tuples.
        """
        results = self.vector_store.similarity_search_with_score(query, k)
        if not results:
            logger.debug("Empty results from similarity_search_with_score")
        return results

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve relevant documents using vector similarity.

        Args:
            query: The search query.
            **kwargs: Additional parameters for the retrieval.

        Returns:
            List of relevant documents.
        """
        top_k = kwargs.get("top_k", 5)
        results = self.search(query, k=top_k)
        return [doc for doc, _ in results]

    def aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve relevant documents using vector similarity (sync fallback).

        Args:
            query: The search query.
            **kwargs: Additional parameters for the retrieval.

        Returns:
            List of relevant documents.
        """
        return self.get_relevant_documents(query, **kwargs)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve documents using vector similarity.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieved documents.
        """
        return self.vector_store.similarity_search(query, k=top_k)
