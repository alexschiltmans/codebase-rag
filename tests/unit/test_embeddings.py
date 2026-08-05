"""Tests for EmbeddingManager's encoding settings resolution and instance cache."""

from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from codebase_rag.config import Config
from codebase_rag.database.embeddings import EmbeddingManager


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Clear the EmbeddingManager cache and Config singleton around each test."""
    EmbeddingManager._instances = {}
    Config._instance = None
    yield
    EmbeddingManager._instances = {}
    Config._instance = None


def _mock_transformer(prompts: dict[str, str], max_seq_length: int = 384) -> MagicMock:
    model = MagicMock()
    model.prompts = prompts
    model.max_seq_length = max_seq_length
    model.encode.return_value = np.array([[0.1, 0.2]])
    return model


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_model_declared_prompts_are_applied(mock_st: MagicMock) -> None:
    """A model that declares its own prompts gets them resolved with no config override."""
    mock_st.return_value = _mock_transformer({"query": "query: ", "document": "passage: "})

    manager = EmbeddingManager(model_name="some/model")

    assert manager.query_prompt == "query: "
    assert manager.document_prompt == "passage: "


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_configured_prompts_override_model_declared(mock_st: MagicMock) -> None:
    """Explicit config prompts win over whatever the model declares."""
    mock_st.return_value = _mock_transformer({"query": "query: ", "document": "passage: "})

    manager = EmbeddingManager(
        model_name="some/model",
        query_prompt="custom-query: ",
        document_prompt="custom-doc: ",
    )

    assert manager.query_prompt == "custom-query: "
    assert manager.document_prompt == "custom-doc: "


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_absent_prompts_encode_with_no_prefix(mock_st: MagicMock) -> None:
    """A model with no declared prompts and no config override encodes with no prefix."""
    mock_st.return_value = _mock_transformer({"query": "", "document": ""})

    manager = EmbeddingManager(model_name="some/model")

    assert manager.query_prompt == ""
    assert manager.document_prompt == ""

    manager.get_embeddings(["text"])
    _, kwargs = cast(MagicMock, manager.model).encode.call_args
    assert kwargs["prompt"] is None


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_cache_distinguishes_configs_that_differ_only_in_prompts(mock_st: MagicMock) -> None:
    """Two configurations differing only in prompts must not share a cached instance."""
    mock_st.side_effect = [
        _mock_transformer({"query": "", "document": ""}),
        _mock_transformer({"query": "", "document": ""}),
    ]

    plain = EmbeddingManager(model_name="some/model", query_prompt="", document_prompt="")
    prefixed = EmbeddingManager(model_name="some/model", query_prompt="query: ", document_prompt="passage: ")

    assert plain is not prefixed
    assert mock_st.call_count == 2


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_dtype_is_passed_as_model_kwargs(mock_st: MagicMock) -> None:
    """Load precision has to reach the model loader, or a large model silently loads
    at float32 and takes twice the memory it was budgeted."""
    mock_st.return_value = _mock_transformer({"query": "", "document": ""})

    EmbeddingManager(model_name="some/model", dtype="float16")

    _, kwargs = mock_st.call_args
    assert kwargs["model_kwargs"] == {"torch_dtype": torch.float16}


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_no_dtype_leaves_model_kwargs_unset(mock_st: MagicMock) -> None:
    """The shipped default must keep loading exactly as it did before."""
    mock_st.return_value = _mock_transformer({"query": "", "document": ""})

    EmbeddingManager(model_name="some/model")

    _, kwargs = mock_st.call_args
    assert kwargs["model_kwargs"] is None


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_unsupported_dtype_raises(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_transformer({"query": "", "document": ""})

    with pytest.raises(ValueError, match="embedding dtype must be one of"):
        EmbeddingManager(model_name="some/model", dtype="float8")


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_cache_distinguishes_dtypes(mock_st: MagicMock) -> None:
    """Two precisions of one model are two different models numerically."""
    mock_st.side_effect = [
        _mock_transformer({"query": "", "document": ""}),
        _mock_transformer({"query": "", "document": ""}),
    ]

    fp32 = EmbeddingManager(model_name="some/model")
    fp16 = EmbeddingManager(model_name="some/model", dtype="float16")

    assert fp32 is not fp16
    assert mock_st.call_count == 2


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_same_config_reuses_cached_instance(mock_st: MagicMock) -> None:
    """Identical settings reuse the same cached instance rather than reloading the model."""
    mock_st.return_value = _mock_transformer({"query": "", "document": ""})

    first = EmbeddingManager(model_name="some/model")
    second = EmbeddingManager(model_name="some/model")

    assert first is second
    assert mock_st.call_count == 1
