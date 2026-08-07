"""Embedding models for converting text to vector representations."""

import logging
from typing import Any, cast

import torch
from sentence_transformers import SentenceTransformer

from ..config import Config

logger = logging.getLogger(__name__)

# Load precisions worth offering. Left unset, `SentenceTransformer` follows the
# checkpoint's own precision rather than defaulting to float32: Qwen3-Embedding-0.6B
# comes up bfloat16 on the pinned sentence-transformers 5.x stack. Asking for float32
# on a model stored at bf16 doubles its memory, which for a 4B model (7.5GB on disk)
# is ~16GB resident and, with a large encode batch, enough to push this machine into swap.
_SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


def _loaded_dtype(model: SentenceTransformer) -> str:
    """Return the precision a model actually loaded at, as a bare dtype name.

    Reported separately from the requested `dtype` because the two differ whenever nothing was
    requested: the checkpoint decides, and a published measurement labelled with the precision it
    asked for rather than the one it ran at is a wrong label on a number.
    """
    for parameter in model.parameters():
        return str(parameter.dtype).removeprefix("torch.")
    return "unknown"


class EmbeddingManager:
    """Manager class for text embedding models.

    Caches one instance per (model name, encoding settings) key so repeated
    construction with the same settings reuses the already-loaded
    `SentenceTransformer`, while different settings get their own instance
    instead of silently reusing the wrong one.
    """

    _instances: dict[tuple[str, str, str, int | None, str | None], "EmbeddingManager"] = {}

    def __new__(
        cls,
        model_name: str | None = None,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        max_seq_length: int | None = None,
        dtype: str | None = None,
    ) -> "EmbeddingManager":
        config = Config.get_instance()
        resolved_model_name = model_name or config.embedding_model
        resolved_query_prompt = query_prompt if query_prompt is not None else config.embedding_query_prompt
        resolved_document_prompt = document_prompt if document_prompt is not None else config.embedding_document_prompt
        resolved_max_seq_length = max_seq_length if max_seq_length is not None else config.embedding_max_seq_length
        resolved_dtype = dtype if dtype is not None else config.embedding_dtype

        if resolved_dtype and resolved_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"embedding dtype must be one of {_SUPPORTED_DTYPES}, got '{resolved_dtype}'")

        cache_key = (
            resolved_model_name,
            resolved_query_prompt,
            resolved_document_prompt,
            resolved_max_seq_length,
            resolved_dtype or None,
        )

        if cache_key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialize(
                resolved_model_name,
                resolved_query_prompt,
                resolved_document_prompt,
                resolved_max_seq_length,
                resolved_dtype or None,
            )
            cls._instances[cache_key] = instance

        return cls._instances[cache_key]

    def _initialize(
        self,
        model_name: str,
        query_prompt: str,
        document_prompt: str,
        max_seq_length: int | None,
        dtype: str | None,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype

        logger.info("Initializing embedding model: %s", self.model_name)
        model_kwargs: dict[str, Any] | None = {"torch_dtype": getattr(torch, dtype)} if dtype else None
        self.model = SentenceTransformer(self.model_name, model_kwargs=model_kwargs)
        self.loaded_dtype = _loaded_dtype(self.model)

        # Resolution order: explicit config wins, then the model's own declared prompts,
        # then no prefix at all.
        declared_prompts = self.model.prompts or {}
        self.query_prompt = query_prompt or declared_prompts.get("query", "")
        self.document_prompt = document_prompt or declared_prompts.get("document", "")

        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.max_seq_length = self.model.max_seq_length

        logger.info(
            "Embedding model initialized (query_prompt=%r, document_prompt=%r, max_seq_length=%s, dtype=%s)",
            self.query_prompt,
            self.document_prompt,
            self.max_seq_length,
            self.dtype or f"unset, loaded {self.loaded_dtype}",
        )

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Return the token length each text would have before truncation.

        Counts with the model's own tokenizer, including the special tokens
        and the document prompt, both of which come out of the same budget as
        the text itself. `get_embeddings` prepends that prompt, so counting the
        bare text would under-report by its length and call a chunk safe that
        the model will in fact cut.
        """
        if not texts:
            return []
        prompt = self.document_prompt or ""
        encoded = self.model.tokenizer([prompt + text for text in texts], add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        prompt = self.document_prompt or None
        embeddings = self.model.encode(texts, prompt=prompt, normalize_embeddings=True)
        return cast(list[list[float]], embeddings.tolist())

    def get_query_embedding(self, text: str) -> list[float]:
        """Get embedding for a query text."""
        prompt = self.query_prompt or None
        embedding = self.model.encode(text, prompt=prompt, normalize_embeddings=True)
        return cast(list[float], embedding.tolist())
