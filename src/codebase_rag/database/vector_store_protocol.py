"""Protocol defining the interface for vector store implementations."""

from typing import Any, Protocol

from langchain_core.documents import Document


class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for vector store providers."""

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
        """
        ...

    def similarity_search(self, query: str, k: int = 4, filter_query: dict[str, Any] | None = None) -> list[Document]:
        """Perform similarity search and return documents.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter_query: Optional filter criteria, ANDed across keys. A list value matches any of
                its entries; any other value must match exactly.

        Returns:
            List of retrieved documents.
        """
        ...

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_query: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Perform similarity search and return documents with scores.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter_query: Optional filter criteria, ANDed across keys. A list value matches any of
                its entries; any other value must match exactly.

        Returns:
            List of (document, score) tuples.
        """
        ...

    def collection_exists(self) -> bool:
        """Check if the collection exists in the vector store.

        Returns:
            Boolean indicating if the collection exists.
        """
        ...

    def delete_by_source(self, source: str) -> None:
        """Delete all points with the given source metadata value.

        Args:
            source: The source path to match.
        """
        ...
