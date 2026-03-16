"""Qdrant vector store implementation."""

import contextlib
import logging
import uuid
from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from codebase_rag.database.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector database store implementation.

    Conforms to the VectorStoreProtocol interface.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        recreate_collection: bool = False,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            collection_name: Name of the collection in Qdrant.
            embedding_model: Name of the HuggingFace model for embeddings.
            recreate_collection: Whether to recreate the collection if it exists.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.recreate_collection = recreate_collection

        self._embedding_manager = EmbeddingManager(model_name=embedding_model)

        self.client = QdrantClient(host=host, port=port)

        if recreate_collection and self.collection_exists():
            self.client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s' for recreation", collection_name)

        logger.info("Initialized QdrantStore with collection '%s'", collection_name)

    def _ensure_collection(self, vector_size: int) -> None:
        """Ensure the collection exists with the correct configuration.

        Args:
            vector_size: Dimensionality of the embedding vectors.
        """
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="repo",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created new Qdrant collection '%s' with vector size %d", self.collection_name, vector_size)

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store with idempotent upsert.

        Uses deterministic point IDs based on source path and chunk index,
        so re-ingesting the same content is a no-op and changed content
        is updated in place.

        Args:
            documents: List of documents to add.
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        try:
            # Generate embeddings for all documents
            texts = [doc.page_content for doc in documents]
            embeddings = self._embedding_manager.get_embeddings(texts)

            self._ensure_collection(vector_size=len(embeddings[0]))

            points = []
            for doc, embedding in zip(documents, embeddings, strict=True):
                point_id = self._deterministic_id(doc)
                payload = {
                    "page_content": doc.page_content,
                    **{k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool, list))},
                }
                points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

            # Upsert in batches, deterministic IDs mean re-runs
            # overwrite existing points rather than creating duplicates
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)

            logger.info("Upserted %d documents to collection '%s'", len(documents), self.collection_name)
        except Exception as e:
            logger.error("Error adding documents to Qdrant: %s", e)
            raise

    def delete_by_source(self, source: str) -> None:
        """Delete all points with the given source metadata value.

        Useful for removing stale chunks when a file is re-ingested or deleted.

        Args:
            source: The source path to match.
        """
        if not self.collection_exists():
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        )
        logger.info("Deleted points with source '%s' from collection '%s'", source, self.collection_name)

    @staticmethod
    def _deterministic_id(doc: Document) -> str:
        """Generate a deterministic UUID for a document based on its source and chunk index.

        Args:
            doc: The document to generate an ID for.

        Returns:
            A deterministic UUID string.
        """
        source = doc.metadata.get("source", "")
        chunk_index = doc.metadata.get("chunk_index", 0)
        key = f"{source}::chunk::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def similarity_search(self, query: str, k: int = 4, filter_query: dict[str, Any] | None = None) -> list[Document]:
        """Perform similarity search and return documents.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter_query: Optional filter criteria.

        Returns:
            List of retrieved documents.
        """
        results_with_scores = self.similarity_search_with_score(query, k, filter_query)
        return [doc for doc, _ in results_with_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_query: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Perform similarity search and return documents with scores.

        Args:
            query: Query text.
            k: Number of documents to return.
            filter_query: Optional filter criteria.

        Returns:
            List of (document, score) tuples.
        """
        if not self.collection_exists():
            logger.error("Collection '%s' does not exist, cannot perform search", self.collection_name)
            return []

        try:
            query_embedding = self._embedding_manager.get_query_embedding(query)

            query_filter = None
            if filter_query:
                conditions = [
                    FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filter_query.items()
                ]
                query_filter = Filter(must=conditions)  # type: ignore[arg-type]

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )

            doc_score_pairs = []
            for point in results.points:
                payload = point.payload or {}
                page_content = payload.pop("page_content", "")
                metadata = dict(payload.items())
                doc = Document(page_content=page_content, metadata=metadata)
                doc_score_pairs.append((doc, point.score))

            logger.info("Retrieved %d documents for query: %s...", len(doc_score_pairs), query[:50])
            return doc_score_pairs
        except Exception as e:
            logger.error("Error during similarity search: %s", e)
            return []

    def collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant.

        Returns:
            Boolean indicating if the collection exists.
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            logger.error("Error checking if collection exists: %s", e)
            return False

    def _ensure_repo_index(self) -> None:
        """Ensure a keyword payload index exists on the 'repo' field.

        This is idempotent — Qdrant silently ignores the call if the index
        already exists.
        """
        with contextlib.suppress(Exception):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="repo",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def list_repos(self) -> list[str]:
        """List distinct repository names stored in the collection.

        Uses the Qdrant facet API for an efficient single-request lookup
        instead of scrolling through all points.

        Returns:
            Sorted list of unique repo names.
        """
        if not self.collection_exists():
            return []

        try:
            response = self.client.facet(
                collection_name=self.collection_name,
                key="repo",
                limit=100,
            )
            return sorted(str(hit.value) for hit in response.hits if hit.value)
        except Exception:
            self._ensure_repo_index()
            try:
                response = self.client.facet(
                    collection_name=self.collection_name,
                    key="repo",
                    limit=100,
                )
                return sorted(str(hit.value) for hit in response.hits if hit.value)
            except Exception as e:
                logger.error("Error listing repos: %s", e)
                return []

    def delete_by_repo(self, repo_name: str) -> int:
        """Delete all points belonging to a specific repository.

        Args:
            repo_name: The repository name to delete.

        Returns:
            Number of points deleted (approximate).
        """
        if not self.collection_exists():
            return 0

        try:
            count_before = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo_name))]),
            ).count

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo_name))]),
            )
            logger.info("Deleted %d points for repo '%s'", count_before, repo_name)
            return int(count_before)
        except Exception as e:
            logger.error("Error deleting repo '%s': %s", repo_name, e)
            return 0
