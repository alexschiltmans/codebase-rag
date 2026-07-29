"""Approximate token counting, shared by search budgeting and response reporting.

Exactness isn't worth a `tiktoken` dependency here: the budget is a cap on
context bloat, not billing-grade accounting. Uses the tokenizer already
loaded for sentence-transformers when available, else a chars/4 heuristic.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str, tokenizer: Any = None) -> int:
    """Estimate the token count of `text`.

    Args:
        text: The text to estimate.
        tokenizer: Optional HuggingFace-style tokenizer (must expose `encode`).
            When absent or when encoding fails, falls back to chars/4.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as exc:  # noqa: BLE001 - any tokenizer failure falls back to the heuristic
            logger.debug("Tokenizer failed, falling back to chars/4 estimate: %s", exc)
    return max(1, len(text) // CHARS_PER_TOKEN)


def get_tokenizer(embedding_manager: Any) -> Any:
    """Best-effort extraction of the underlying tokenizer from an EmbeddingManager."""
    model = getattr(embedding_manager, "model", None)
    return getattr(model, "tokenizer", None)
