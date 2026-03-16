"""Embedding models for converting text to vector representations."""

import logging
from typing import cast

from sentence_transformers import SentenceTransformer

from ..config import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manager class for text embedding models.

    Uses the singleton pattern to ensure only one instance of the embedding model
    is loaded in memory at any time.
    """

    _instance = None

    def __new__(cls, model_name: str | None = None) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_name)
        return cls._instance

    def _initialize(self, model_name: str | None = None) -> None:
        config = Config.get_instance()
        self.model_name = model_name or config.embedding_model

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
