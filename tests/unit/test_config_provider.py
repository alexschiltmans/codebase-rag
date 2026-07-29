"""Tests for LLM provider configuration validation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.config import Config


class TestProviderValidation:
    """Test cases for LLM_PROVIDER configuration validation."""

    @patch("codebase_rag.config.load_dotenv")
    def test_valid_ollama_provider(self, mock_load_dotenv: MagicMock) -> None:
        """LLM_PROVIDER=ollama is accepted."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            config = Config.get_instance()
            assert config.provider == "ollama"

    @patch("codebase_rag.config.load_dotenv")
    def test_valid_openai_compat_provider(self, mock_load_dotenv: MagicMock) -> None:
        """LLM_PROVIDER=openai-compat is accepted, given a base URL."""
        env = {"LLM_PROVIDER": "openai-compat", "LLM_BASE_URL": "http://localhost:1234/v1"}
        with patch.dict(os.environ, env):
            config = Config.get_instance()
            assert config.provider == "openai-compat"

    @patch("codebase_rag.config.load_dotenv")
    def test_invalid_provider_raises_error(self, mock_load_dotenv: MagicMock) -> None:
        """Invalid LLM_PROVIDER value raises ValueError."""
        Config._instance = None
        with (
            patch.dict(os.environ, {"LLM_PROVIDER": "invalid-provider"}),
            pytest.raises(ValueError, match="LLM_PROVIDER must be 'ollama' or 'openai-compat'"),
        ):
            Config.get_instance()

    @patch("codebase_rag.config.load_dotenv")
    def test_default_provider_is_ollama(self, mock_load_dotenv: MagicMock) -> None:
        """Default LLM_PROVIDER is ollama when not set."""
        env_copy = dict(os.environ)
        env_copy.pop("LLM_PROVIDER", None)
        with patch.dict(os.environ, env_copy, clear=True):
            config = Config.get_instance()
            assert config.provider == "ollama"

    @patch("codebase_rag.config.load_dotenv")
    def test_empty_llm_model_name_falls_back_to_default(self, mock_load_dotenv: MagicMock) -> None:
        """An emptied LLM_MODEL_NAME= must not ship a request with "model": "" and fail at
        generation time instead of here, the same failure class as the LLM_PROVIDER fix.
        """
        Config._instance = None
        with patch.dict(os.environ, {"LLM_MODEL_NAME": ""}):
            config = Config.get_instance()
            assert config.llm_model_name == Config.llm_model_name

    @patch("codebase_rag.config.load_dotenv")
    def test_llm_base_url_config(self, mock_load_dotenv: MagicMock) -> None:
        """LLM_BASE_URL is loaded into config."""
        Config._instance = None
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:1234/v1"}):
            config = Config.get_instance()
            assert config.llm_base_url == "http://localhost:1234/v1"

    @patch("codebase_rag.config.load_dotenv")
    def test_empty_llm_provider_falls_back_to_default(self, mock_load_dotenv: MagicMock) -> None:
        """An emptied .env line (LLM_PROVIDER=) sets the var to "", not absent; getenv's own
        default only covers "absent", so this must fall back explicitly or every get_instance()
        call raises ValueError forever, since _instance is never assigned on a failed construction.
        """
        Config._instance = None
        with patch.dict(os.environ, {"LLM_PROVIDER": ""}):
            config = Config.get_instance()
            assert config.provider == "ollama"

    @patch("codebase_rag.config.load_dotenv")
    def test_openai_compat_without_base_url_raises_at_config_time(self, mock_load_dotenv: MagicMock) -> None:
        """openai-compat with no LLM_BASE_URL must fail here, with a clear message, rather than
        passing validation and dying inside OpenAICompatClient.__init__ as an uncaught exception
        (a raw Streamlit traceback in the app).
        """
        Config._instance = None
        env = {"LLM_PROVIDER": "openai-compat"}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="LLM_BASE_URL must be set"),
        ):
            Config.get_instance()

    @patch("codebase_rag.config.load_dotenv")
    def test_openai_compat_with_base_url_succeeds(self, mock_load_dotenv: MagicMock) -> None:
        Config._instance = None
        env = {"LLM_PROVIDER": "openai-compat", "LLM_BASE_URL": "http://localhost:1234/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = Config.get_instance()
            assert config.llm_base_url == "http://localhost:1234/v1"

    @patch("codebase_rag.config.load_dotenv")
    def test_llm_api_key_config(self, mock_load_dotenv: MagicMock) -> None:
        """LLM_API_KEY is loaded into config."""
        Config._instance = None
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key-123"}):
            config = Config.get_instance()
            assert config.llm_api_key == "test-key-123"


class TestRetrieverValidation:
    """Test cases for RETRIEVER configuration validation."""

    @patch("codebase_rag.config.load_dotenv")
    def test_default_retriever_is_bm25(self, mock_load_dotenv: MagicMock) -> None:
        Config._instance = None
        env_copy = dict(os.environ)
        env_copy.pop("RETRIEVER", None)
        with patch.dict(os.environ, env_copy, clear=True):
            config = Config.get_instance()
            assert config.retriever == "bm25"

    @patch("codebase_rag.config.load_dotenv")
    def test_valid_hybrid_retriever(self, mock_load_dotenv: MagicMock) -> None:
        Config._instance = None
        with patch.dict(os.environ, {"RETRIEVER": "hybrid"}):
            config = Config.get_instance()
            assert config.retriever == "hybrid"

    @patch("codebase_rag.config.load_dotenv")
    def test_invalid_retriever_raises_error(self, mock_load_dotenv: MagicMock) -> None:
        Config._instance = None
        with (
            patch.dict(os.environ, {"RETRIEVER": "reranked"}),
            pytest.raises(ValueError, match="RETRIEVER must be 'bm25' or 'hybrid'"),
        ):
            Config.get_instance()
