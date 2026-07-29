"""Unit tests for services/token_estimator.py."""

from unittest.mock import MagicMock

from codebase_rag.services.token_estimator import estimate_tokens, get_tokenizer


class TestEstimateTokens:
    def test_chars_over_four_fallback(self) -> None:
        assert estimate_tokens("x" * 40) == 10

    def test_minimum_one_token(self) -> None:
        assert estimate_tokens("x") == 1

    def test_uses_tokenizer_when_provided(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        assert estimate_tokens("anything", tokenizer=tokenizer) == 3

    def test_falls_back_when_tokenizer_raises(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = RuntimeError("boom")

        assert estimate_tokens("x" * 8, tokenizer=tokenizer) == 2


class TestGetTokenizer:
    def test_extracts_tokenizer_from_embedding_manager(self) -> None:
        manager = MagicMock()
        manager.model.tokenizer = "the-tokenizer"

        assert get_tokenizer(manager) == "the-tokenizer"

    def test_returns_none_when_absent(self) -> None:
        manager = object()

        assert get_tokenizer(manager) is None
