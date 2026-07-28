"""Vector similarity search retriever.

This module implements vector-based document retrieval, searching for documents
by computing the similarity between query and document vectors.
"""

import logging

from langchain_core.documents import Document

from ..database.vector_store_protocol import VectorStoreProtocol

logger = logging.getLogger(__name__)

# Below this raw cosine similarity, chunks are treated as irrelevant and
# excluded from retrieval (and therefore from RRF fusion). Validated against
# evals/testset.json: at 0.25, all 16 questions still retrieve documents.
# Shared by the app runtime and the eval's hybrid arm so the eval measures
# the same retrieval configuration production ships.
VECTOR_SCORE_THRESHOLD = 0.25

# Number of documents `search` returns when the caller passes no `k`. The
# protocol's `k=None` resolves to this, so the value lives here rather than
# in the signature (where `None` is the declared default) or in the docstring.
DEFAULT_TOP_K = 5


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
        if score_threshold is not None:
            logger.info("Using score threshold: %s", score_threshold)

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents similar to the query.

        Results below ``self.score_threshold`` (raw cosine similarity from the
        vector store) are dropped before returning, so relevance filtering
        happens here, on a real similarity signal, rather than after fusion.
        No filtering is applied when ``score_threshold`` is ``None``.

        Args:
            query: The search query.
            k: Number of documents to retrieve. ``None`` uses ``DEFAULT_TOP_K``.

        Returns:
            List of (document, score) tuples.
        """
        k_value = k if k is not None else DEFAULT_TOP_K
        results = self.vector_store.similarity_search_with_score(query, k_value)
        if not results:
            logger.debug("Empty results from similarity_search_with_score")
            return results

        if self.score_threshold is not None:
            filtered = [(doc, score) for doc, score in results if score >= self.score_threshold]
            if len(filtered) != len(results):
                logger.debug(
                    "Filtered %d/%d results below score_threshold=%s",
                    len(results) - len(filtered),
                    len(results),
                    self.score_threshold,
                )
            return filtered

        return results
