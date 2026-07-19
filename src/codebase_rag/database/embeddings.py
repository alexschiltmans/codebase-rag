"""Embedding models for converting text to vector representations."""

import logging
from typing import cast

from sentence_transformers import SentenceTransformer

from ..config import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manager class for text embedding models.

    Caches one instance per model name so repeated construction with the same
    model reuses the already-loaded `SentenceTransformer`, while a different
    model name gets its own instance instead of silently reusing the wrong one.
    """

    _instances: dict[str, "EmbeddingManager"] = {}

    def __new__(cls, model_name: str | None = None) -> "EmbeddingManager":
        config = Config.get_instance()
        resolved_model_name = model_name or config.embedding_model

        if resolved_model_name not in cls._instances:
            instance = super().__new__(cls)
            instance._initialize(resolved_model_name)
            cls._instances[resolved_model_name] = instance

        return cls._instances[resolved_model_name]

    def _initialize(self, model_name: str) -> None:
        self.model_name = model_name

        logger.info("Initializing embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model initialized")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], embeddings.tolist())

    def get_query_embedding(self, text: str) -> list[float]:
        """Get embedding for a query text."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return cast(list[float], embedding.tolist())
