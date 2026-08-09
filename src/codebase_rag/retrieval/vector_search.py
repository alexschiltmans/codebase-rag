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

# Cosine thresholds are calibrated per embedding model's own score distribution,
# not a value shared across models. A model with no entry here has no calibrated
# cutoff and gets no threshold rather than silently inheriting another model's.
_CALIBRATED_SCORE_THRESHOLDS: dict[str, float] = {
    "sentence-transformers/all-mpnet-base-v2": VECTOR_SCORE_THRESHOLD,
}


def resolve_score_threshold(embedding_model: str) -> float | None:
    """Resolve the relevance cutoff calibrated for the given embedding model.

    Args:
        embedding_model: The embedding model name the collection was built with.

    Returns:
        The calibrated cosine threshold, or None if the model has no calibrated
        value (no threshold is applied rather than inheriting one calibrated for
        a different model's score distribution).
    """
    threshold = _CALIBRATED_SCORE_THRESHOLDS.get(embedding_model)
    if threshold is not None:
        logger.info("Resolved score threshold %s for embedding model '%s'", threshold, embedding_model)
    else:
        # Warned, not merely noted: the lookup is an exact string match, so a local path or
        # stray whitespace around a model that does have a calibrated cutoff drops filtering
        # entirely. That is a silent behavior change and an INFO line would not surface it.
        logger.warning(
            "No calibrated score threshold for embedding model '%s'; no relevance cutoff will be "
            "applied. Calibrated models are: %s. If this model was meant to match one of them, "
            "check for a path prefix or stray whitespace in the configured name.",
            embedding_model,
            sorted(_CALIBRATED_SCORE_THRESHOLDS),
        )
    return threshold


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
        repos: list[str] | None = None,
    ) -> None:
        """Initialize the vector retriever.

        Args:
            vector_store: The vector store to search (any VectorStoreProtocol implementation).
            score_threshold: Optional minimum similarity score threshold.
            repos: Optional repository names to restrict every search to. None searches
                everything in the store, which is what the app and the API do.
        """
        self.vector_store = vector_store
        self.score_threshold = score_threshold
        self.repos = repos

        logger.info("Initialized VectorRetriever with %s", vector_store.__class__.__name__)
        if score_threshold is not None:
            logger.info("Using score threshold: %s", score_threshold)
        if repos is not None:
            logger.info("Restricted to repositories: %s", ", ".join(repos))

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents similar to the query.

        Results below ``self.score_threshold`` (raw cosine similarity from the
        vector store) are dropped before returning, so relevance filtering
        happens here, on a real similarity signal, rather than after fusion.
        No filtering is applied when ``score_threshold`` is ``None``.

        A repository restriction, unlike the score threshold, goes to the store
        as a filter rather than being applied to what comes back, so ``k`` stays
        ``k`` in-scope results instead of however many survive discarding.

        Args:
            query: The search query.
            k: Number of documents to retrieve. ``None`` uses ``DEFAULT_TOP_K``.

        Returns:
            List of (document, score) tuples.
        """
        k_value = k if k is not None else DEFAULT_TOP_K
        filter_query = {"repo": self.repos} if self.repos is not None else None
        results = self.vector_store.similarity_search_with_score(query, k_value, filter_query)
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
