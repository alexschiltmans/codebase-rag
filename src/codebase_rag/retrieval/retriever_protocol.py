"""Protocol defining the interface for retriever implementations."""

from typing import Protocol

from langchain_core.documents import Document


class RetrieverProtocol(Protocol):
    """Protocol defining the interface for document retrievers.

    Any retriever consumed by `RAGChain` conforms to this single method —
    structural typing means `VectorRetriever`, `BM25Retriever`, and
    `HybridRetriever` satisfy it without inheriting from it.
    """

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents relevant to the query.

        Args:
            query: The search query.
            k: Number of results to return. `None` means the retriever's
                own default.

        Returns:
            List of (document, score) tuples.
        """
        ...
