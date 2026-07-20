"""Tests for the OllamaClient wrapper."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from codebase_rag.llm.ollama_client import OllamaClient


@patch("codebase_rag.llm.ollama_client.ChatOllama")
@patch("codebase_rag.llm.ollama_client.Config")
def test_num_ctx_defaults_from_config(mock_config_cls: MagicMock, mock_chat_ollama: MagicMock) -> None:
    """`num_ctx` falls back to `config.ollama_num_ctx` when not passed explicitly."""
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config

    OllamaClient(model_name="test-model")

    _, kwargs = mock_chat_ollama.call_args
    assert kwargs["num_ctx"] == 8192


@patch("codebase_rag.llm.ollama_client.ChatOllama")
@patch("codebase_rag.llm.ollama_client.Config")
def test_num_ctx_override(mock_config_cls: MagicMock, mock_chat_ollama: MagicMock) -> None:
    """An explicit `num_ctx` argument is passed through to `ChatOllama`."""
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config

    OllamaClient(model_name="test-model", num_ctx=4096)

    _, kwargs = mock_chat_ollama.call_args
    assert kwargs["num_ctx"] == 4096


@patch("codebase_rag.llm.ollama_client.ChatOllama")
@patch("codebase_rag.llm.ollama_client.Config")
def test_num_ctx_below_floor_is_rejected(mock_config_cls: MagicMock, mock_chat_ollama: MagicMock) -> None:
    """A `num_ctx` that leaves no usable prompt budget fails at construction."""
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config

    with pytest.raises(ValueError, match="OLLAMA_NUM_CTX"):
        OllamaClient(model_name="test-model", num_ctx=1024, max_tokens=1024)

    mock_chat_ollama.assert_not_called()


@patch("codebase_rag.llm.ollama_client.ChatOllama")
@patch("codebase_rag.llm.ollama_client.Config")
def test_num_ctx_at_workable_floor_constructs(mock_config_cls: MagicMock, mock_chat_ollama: MagicMock) -> None:
    """`OLLAMA_NUM_CTX=2048` against the default `max_tokens=1024` reservation still constructs."""
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config

    client = OllamaClient(model_name="test-model", num_ctx=2048, max_tokens=1024)

    assert client.prompt_budget_chars > 0


class TestOllamaClient:
    """Test cases for OllamaClient."""

    @patch("codebase_rag.llm.ollama_client.Config")
    def test_initialization(self, mock_config_cls: MagicMock) -> None:
        """Test OllamaClient initialization."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "default-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        client = OllamaClient(
            model_name="test-model",
            base_url="http://test:11434",
            timeout=60,
        )

        assert client.model_name == "test-model"
        assert client.base_url == "http://test:11434"
        assert client.timeout == 60

    @patch("codebase_rag.llm.ollama_client.Config")
    def test_invoke(self, mock_config_cls: MagicMock) -> None:
        """Test text generation via invoke."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        client = OllamaClient(model_name="test-model")

        # Mock the ChatOllama inner LLM
        mock_message = MagicMock()
        mock_message.content = "Generated text"
        client._llm = MagicMock()
        client._llm.invoke.return_value = mock_message

        result = client.invoke("Test prompt")
        assert result == "Generated text"
        client._llm.invoke.assert_called_once_with("Test prompt")

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_success(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test successful connection check."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response

        client = OllamaClient(model_name="test-model")
        result = client.check_connection()

        assert result["status"] == "connected"
        assert result["version"] == "0.1.0"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_failure(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test connection check when Ollama is not reachable."""

        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = OllamaClient(model_name="test-model")
        result = client.check_connection()

        assert result["status"] == "error"
        assert "Cannot connect" in result["message"]

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_available(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test model availability check when model exists."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        # First call: version check, second call: tags
        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "test-model"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="test-model")
        result = client.check_model_availability()

        assert result["status"] == "available"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_not_found(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Test model availability check when model is missing."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "missing-model"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "other-model"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="missing-model")
        result = client.check_model_availability()

        assert result["status"] == "not_found"
