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


def _mock_transformer(
    prompts: dict[str, str], max_seq_length: int = 384, dtype: torch.dtype = torch.float32
) -> MagicMock:
    model = MagicMock()
    model.prompts = prompts
    model.max_seq_length = max_seq_length
    model.encode.return_value = np.array([[0.1, 0.2]])
    model.parameters.return_value = iter([torch.zeros(1, dtype=dtype)])
    return model


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_the_precision_that_actually_loaded_is_reported_separately(mock_st: MagicMock) -> None:
    """Requesting nothing does not mean float32: the checkpoint decides, and Qwen3-Embedding-0.6B
    comes up bfloat16. A benchmark arm labelled with what it asked for is a wrong label."""
    mock_st.return_value = _mock_transformer({}, dtype=torch.bfloat16)

    manager = EmbeddingManager(model_name="some/model")

    assert manager.dtype is None
    assert manager.loaded_dtype == "bfloat16"


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_a_requested_precision_is_reported_as_loaded_too(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_transformer({}, dtype=torch.float16)

    manager = EmbeddingManager(model_name="some/model", dtype="float16")

    assert (manager.dtype, manager.loaded_dtype) == ("float16", "float16")


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


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_count_tokens_measures_with_the_models_own_tokenizer(mock_st: MagicMock) -> None:
    """Token counts come from the model's tokenizer, specials included and untruncated."""
    model = _mock_transformer({"query": "", "document": ""})
    model.tokenizer.return_value = {"input_ids": [[1, 2, 3], [1, 2]]}
    mock_st.return_value = model

    manager = EmbeddingManager(model_name="some/model")
    lengths = manager.count_tokens(["longer text", "short"])

    assert lengths == [3, 2]
    model.tokenizer.assert_called_once_with(["longer text", "short"], add_special_tokens=True, truncation=False)


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_count_tokens_on_nothing_does_not_call_the_tokenizer(mock_st: MagicMock) -> None:
    """An empty batch is not a tokenizer call; some tokenizers reject one."""
    model = _mock_transformer({"query": "", "document": ""})
    mock_st.return_value = model

    assert EmbeddingManager(model_name="some/model").count_tokens([]) == []
    model.tokenizer.assert_not_called()


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_count_tokens_includes_the_document_prompt(mock_st: MagicMock) -> None:
    """The prompt is prepended before embedding, so it spends the same token budget."""
    model = _mock_transformer({"query": "query: ", "document": "passage: "})
    model.tokenizer.return_value = {"input_ids": [[1, 2, 3, 4]]}
    mock_st.return_value = model

    EmbeddingManager(model_name="some/model").count_tokens(["chunk text"])

    counted = model.tokenizer.call_args.args[0]
    assert counted == ["passage: chunk text"]


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_revision_is_threaded_to_the_model_loader(mock_st: MagicMock) -> None:
    """An unpinned model name resolves to whatever the hub's default branch points at that day."""
    mock_st.return_value = _mock_transformer({})

    manager = EmbeddingManager(model_name="some/model", revision="abc123")

    assert manager.revision == "abc123"
    assert mock_st.call_args.kwargs["revision"] == "abc123"


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_unset_revision_loads_unpinned_and_says_so(mock_st: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    mock_st.return_value = _mock_transformer({})

    with caplog.at_level("WARNING", logger="codebase_rag.database.embeddings"):
        manager = EmbeddingManager(model_name="some/model")

    assert manager.revision is None
    assert mock_st.call_args.kwargs["revision"] is None
    assert "EMBEDDING_MODEL_REVISION" in caplog.text


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_two_revisions_of_one_name_are_two_instances(mock_st: MagicMock) -> None:
    """Two revisions are two sets of weights, so the cache must not hand one back for the other."""
    mock_st.side_effect = [_mock_transformer({}), _mock_transformer({})]

    first = EmbeddingManager(model_name="some/model", revision="sha-one")
    second = EmbeddingManager(model_name="some/model", revision="sha-two")

    assert first is not second
    assert mock_st.call_count == 2


@patch("codebase_rag.database.embeddings.SentenceTransformer")
def test_the_same_revision_reuses_the_loaded_model(mock_st: MagicMock) -> None:
    mock_st.return_value = _mock_transformer({})

    first = EmbeddingManager(model_name="some/model", revision="sha-one")
    second = EmbeddingManager(model_name="some/model", revision="sha-one")

    assert first is second
    assert mock_st.call_count == 1
