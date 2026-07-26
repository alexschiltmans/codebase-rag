"""Tests for LLM provider factory and selection."""

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.llm.openai_compat_client import OpenAICompatClient
from codebase_rag.llm.provider_factory import create_llm_client


class TestProviderFactory:
    """Test cases for LLM provider factory."""

    @patch("codebase_rag.llm.provider_factory.OpenAICompatClient")
    @patch("codebase_rag.llm.provider_factory.OllamaClient")
    @patch("codebase_rag.llm.provider_factory.Config")
    def test_ollama_provider_selected_by_default(
        self, mock_config_cls: MagicMock, mock_ollama_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        """LLM_PROVIDER=ollama creates OllamaClient."""
        mock_config = MagicMock()
        mock_config.provider = "ollama"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.llm_model_name = "test-model"
        mock_config_cls.get_instance.return_value = mock_config

        create_llm_client(model_name="test-model")

        mock_ollama_client.assert_called_once()
        mock_openai_client.assert_not_called()

    @patch("codebase_rag.llm.provider_factory.OpenAICompatClient")
    @patch("codebase_rag.llm.provider_factory.OllamaClient")
    @patch("codebase_rag.llm.provider_factory.Config")
    def test_openai_compat_provider_creates_openai_client(
        self, mock_config_cls: MagicMock, mock_ollama_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        """LLM_PROVIDER=openai-compat creates OpenAICompatClient."""
        mock_config = MagicMock()
        mock_config.provider = "openai-compat"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config.llm_model_name = "test-model"
        mock_config_cls.get_instance.return_value = mock_config

        create_llm_client(model_name="test-model")

        mock_openai_client.assert_called_once()
        mock_ollama_client.assert_not_called()

    @patch("codebase_rag.llm.provider_factory.Config")
    def test_unknown_provider_raises_error(self, mock_config_cls: MagicMock) -> None:
        """Unknown LLM_PROVIDER value raises ValueError with valid options listed."""
        mock_config = MagicMock()
        mock_config.provider = "invalid-provider"
        mock_config_cls.get_instance.return_value = mock_config

        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER: invalid-provider"):
            create_llm_client()


class TestOpenAICompatClientBehavior:
    """Behavioral conformance tests for OpenAICompatClient with HTTP mocking."""

    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_initialization(self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock) -> None:
        """Test OpenAICompatClient initialization."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        client = OpenAICompatClient(model_name="test-model", base_url="http://localhost:1234/v1")

        assert client.model_name == "test-model"
        assert client.base_url == "http://localhost:1234/v1"

    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_base_url_required(self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock) -> None:
        """OpenAICompatClient requires LLM_BASE_URL."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = ""
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        with pytest.raises(ValueError, match="LLM_BASE_URL must be set"):
            OpenAICompatClient()

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_check_connection_reachable(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """check_connection succeeds when server is reachable."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        result = client.check_connection()

        assert result["status"] == "connected"
        assert result["model"] == "test-model"
        mock_get.assert_called_once_with("http://localhost:1234/v1/models", headers={}, timeout=5)

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_check_connection_connection_error(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """check_connection fails when server is unreachable."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_get.side_effect = __import__("requests").exceptions.ConnectionError("Connection refused")

        client = OpenAICompatClient()
        result = client.check_connection()

        assert result["status"] == "error"
        assert "Cannot connect" in result["message"]

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_check_model_availability_present(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """check_model_availability succeeds when model is available."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}, {"id": "other-model"}]}
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        result = client.check_model_availability()

        assert result["status"] == "available"
        assert result["model"] == "test-model"

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_check_model_availability_not_found(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """check_model_availability fails when model is not found."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "missing-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        result = client.check_model_availability()

        assert result["status"] == "not_found"
        assert "suggested_action" in result

    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_invoke_returns_content(self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock) -> None:
        """invoke returns generated content."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_llm_instance = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response"
        mock_llm_instance.invoke.return_value = mock_message
        mock_base_chat_openai.return_value = mock_llm_instance

        client = OpenAICompatClient()
        result = client.invoke("test prompt")

        assert result == "Generated response"

    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_stream_yields_chunks(self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock) -> None:
        """stream yields successive text chunks and skips empty ones."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Hello "
        mock_chunk2 = MagicMock()
        mock_chunk2.content = ""
        mock_chunk3 = MagicMock()
        mock_chunk3.content = "world"

        mock_llm_instance = MagicMock()
        mock_llm_instance.stream.return_value = iter([mock_chunk1, mock_chunk2, mock_chunk3])
        mock_base_chat_openai.return_value = mock_llm_instance

        client = OpenAICompatClient()
        chunks = list(client.stream("test prompt"))

        assert chunks == ["Hello ", "world"]

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_models_probe_sends_api_key(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """check_connection and check_model_availability send the configured API key as a Bearer token."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:8000/v1"
        mock_config.llm_api_key = "secret-token"
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        client.check_connection()
        client.check_model_availability()

        for call in mock_get.call_args_list:
            assert call.kwargs["headers"] == {"Authorization": "Bearer secret-token"}

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_models_probe_no_auth_header_without_api_key(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """No Authorization header is sent when no API key is configured."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:8000/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        client.check_connection()

        assert mock_get.call_args.kwargs["headers"] == {}

    @patch("codebase_rag.llm.openai_compat_client.requests.get")
    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_check_model_availability_handles_non_json_response(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock, mock_get: MagicMock
    ) -> None:
        """A 200 with a non-JSON body must return an error dict, not raise json.JSONDecodeError
        out of check_model_availability into the caller's own broad exception handler.
        """
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not JSON")
        mock_get.return_value = mock_response

        client = OpenAICompatClient()
        result = client.check_model_availability()

        assert result["status"] == "error"

    @patch("codebase_rag.llm.openai_compat_client.BaseChatOpenAI")
    @patch("codebase_rag.llm.openai_compat_client.Config")
    def test_base_url_trailing_slash_stripped(
        self, mock_config_cls: MagicMock, mock_base_chat_openai: MagicMock
    ) -> None:
        """A trailing slash on LLM_BASE_URL doesn't produce a doubled slash on /models."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config.llm_base_url = "http://localhost:1234/v1/"
        mock_config.llm_api_key = ""
        mock_config_cls.get_instance.return_value = mock_config

        client = OpenAICompatClient()

        assert client.base_url == "http://localhost:1234/v1"
