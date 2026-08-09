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
    MatchAny,
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
        embedding_model: str | None = None,
        recreate_collection: bool = False,
        embedding_max_seq_length: int | None = None,
        embedding_dtype: str | None = None,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            collection_name: Name of the collection in Qdrant.
            embedding_model: Name of the HuggingFace model for embeddings. When None,
                EmbeddingManager falls back to Config.embedding_model.
            recreate_collection: Whether to recreate the collection if it exists.
            embedding_max_seq_length: Optional sequence-length override for the embedding model.
            embedding_dtype: Optional load precision for the embedding model.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.recreate_collection = recreate_collection

        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            max_seq_length=embedding_max_seq_length,
            dtype=embedding_dtype,
        )
        self._model_binding_verified = False

        self.client = QdrantClient(host=host, port=port)

        if recreate_collection and self.collection_exists():
            self.client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s' for recreation", collection_name)

        logger.info("Initialized QdrantStore with collection '%s'", collection_name)

    def _meta_collection_name(self) -> str:
        return f"{self.collection_name}__meta"

    def _encoding_identity(self) -> dict[str, object]:
        """Everything that changes the vectors a text produces.

        The model name alone is not enough: a prompt prefix or a sequence-length cap changes
        the output as surely as swapping the model does, and a collection built one way cannot
        be queried the other way.
        """
        manager = self.embedding_manager
        return {
            "embedding_model": manager.model_name,
            "query_prompt": getattr(manager, "query_prompt", ""),
            "document_prompt": getattr(manager, "document_prompt", ""),
            "max_seq_length": getattr(manager, "max_seq_length", None),
            "dtype": getattr(manager, "dtype", None),
        }

    def _record_model_binding(self, vector_size: int) -> None:
        """Record the encoding configuration a newly created collection was built with."""
        meta_name = self._meta_collection_name()
        # The sidecar outlives its collection: dropping a collection does not drop this, so a
        # recreated collection finds the old sidecar still there and must overwrite the binding
        # rather than fail trying to create it again.
        if not self.client.collection_exists(meta_name):
            self.client.create_collection(
                collection_name=meta_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
        self.client.upsert(
            collection_name=meta_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, meta_name)),
                    vector=[0.0],
                    payload={**self._encoding_identity(), "dimension": vector_size},
                )
            ],
        )

    def _verify_dimension(self) -> None:
        """Check the collection's vector width against the configured model's, for collections with no sidecar.

        The only thing recoverable about a collection built before bindings were recorded, or one
        whose sidecar was lost. It catches a model swap across dimensions and nothing else: two
        768-dimension models still slot into each other silently, so this narrows the hole rather
        than closing it.

        Raises:
            ValueError: If the collection's vectors are a different width than the configured
                model produces.
        """
        configured = self.embedding_manager.model.get_sentence_embedding_dimension()
        params = self.client.get_collection(self.collection_name).config.params.vectors
        stored = getattr(params, "size", None)

        # Both widths have to be plain integers to be worth comparing. A collection configured with
        # named vectors reports a mapping rather than a size, and this project does not create those;
        # inventing a mismatch from a shape that was never a width would block a working collection.
        if not isinstance(configured, int) or not isinstance(stored, int) or stored == configured:
            return

        raise ValueError(
            f"Collection '{self.collection_name}' holds {stored}-dimension vectors but the "
            f"configured model '{self.embedding_manager.model_name}' produces {configured}. "
            "The collection records no embedding configuration, so it predates that check or lost "
            "its sidecar; it has to be rebuilt with the model now in use."
        )

    def _verify_model_binding(self) -> None:
        """Verify the collection's recorded embedding model matches the configured one.

        Raises:
            ValueError: If the collection was built with a different embedding model
                than the one currently configured, naming both.
        """
        if self._model_binding_verified:
            return

        meta_name = self._meta_collection_name()
        # Existence is checked rather than inferred from a failed read. Catching every
        # exception here would read a connection blip as "nothing recorded" and then latch
        # the guard off for the rest of the process, which is the opposite of what it is for.
        # A transport failure propagates instead; the query it guards would have failed anyway.
        if not self.client.collection_exists(meta_name):
            self._verify_dimension()
            self._model_binding_verified = True
            return

        points = self.client.retrieve(
            collection_name=meta_name,
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, meta_name))],
        )
        payload = points[0].payload if points and points[0].payload else {}

        mismatches = []
        for field, configured in self._encoding_identity().items():
            # Absence and a recorded null are different answers. A field missing from the payload
            # predates this check; a field recorded as null is a value, and `dtype` is null for
            # every collection built without an explicit precision, which is the common case.
            # Reading the second as the first waves through exactly the mismatch this guards.
            if field not in payload:
                continue
            recorded = payload[field]
            if recorded != configured:
                mismatches.append(f"{field}: recorded '{recorded}', configured '{configured}'")

        if mismatches:
            raise ValueError(
                f"Collection '{self.collection_name}' was built with a different embedding "
                f"configuration than the one now in use ({'; '.join(mismatches)}). "
                "Querying it with vectors produced a different way gives meaningless scores."
            )
        self._model_binding_verified = True

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
            self._record_model_binding(vector_size)
            self._model_binding_verified = True
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
            embeddings = self.embedding_manager.get_embeddings(texts)

            self._ensure_collection(vector_size=len(embeddings[0]))
            # Verify before writing, not just before reading. Re-ingesting into an existing
            # collection with a different model of the same dimension is accepted by Qdrant and
            # leaves two models' vectors under one index; checking only on the read path would
            # report that after the damage rather than prevent it.
            self._verify_model_binding()

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
            filter_query: Optional filter criteria, ANDed across keys. A list value matches any of
                its entries; any other value must match exactly.

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
            filter_query: Optional filter criteria, ANDed across keys. A list value matches any of
                its entries; any other value must match exactly.

        Returns:
            List of (document, score) tuples.
        """
        if not self.collection_exists():
            logger.error("Collection '%s' does not exist, cannot perform search", self.collection_name)
            return []

        self._verify_model_binding()

        try:
            query_embedding = self.embedding_manager.get_query_embedding(query)

            query_filter = None
            if filter_query:
                # A list is any-of: the enclosing `must` would AND several MatchValue conditions into nothing.
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value) if isinstance(value, list) else MatchValue(value=value),
                    )
                    for key, value in filter_query.items()
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
            raise RuntimeError(f"Vector search failed: {e}") from e

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
