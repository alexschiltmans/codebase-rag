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


@patch("codebase_rag.llm.ollama_client.requests.get")
@patch("codebase_rag.llm.ollama_client.ChatOllama")
@patch("codebase_rag.llm.ollama_client.Config")
def test_check_connection_handles_non_json_response(
    mock_config_cls: MagicMock, mock_chat_ollama: MagicMock, mock_get: MagicMock
) -> None:
    """A 200 with a non-JSON body (e.g. a proxy's HTML error page) must return an error dict,
    not raise json.JSONDecodeError out of check_connection into the caller's own handler.
    """
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("not JSON")
    mock_get.return_value = mock_response

    client = OllamaClient(model_name="test-model")
    result = client.check_connection()

    assert result["status"] == "error"


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

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_untagged_matches_latest(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """An untagged model name matches only the `:latest` tag, not just any tag."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "llama3"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "llama3:latest"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="llama3")
        result = client.check_model_availability()

        assert result["status"] == "available"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_model_untagged_does_not_match_other_tag(
        self, mock_config_cls: MagicMock, mock_get: MagicMock
    ) -> None:
        """An untagged model name must not match an arbitrary stored tag like `:8b`."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "llama3"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}

        mock_get.side_effect = [version_resp, tags_resp]

        client = OllamaClient(model_name="llama3")
        result = client.check_model_availability()

        assert result["status"] == "not_found"


def _placement_client(mock_config_cls: MagicMock, base_url: str = "http://localhost:11434") -> OllamaClient:
    """Build a client whose Config is stubbed, for the placement and remedy tests."""
    mock_config = MagicMock()
    mock_config.llm_model_name = "test-model"
    mock_config.ollama_base_url = base_url
    mock_config.ollama_num_ctx = 8192
    mock_config_cls.get_instance.return_value = mock_config
    return OllamaClient(model_name="test-model", base_url=base_url)


def _ps_response(models: list[dict[str, object]], status_code: int = 200) -> MagicMock:
    """Stub an `/api/ps` reply carrying the given running-model entries."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"models": models}
    return response


class TestRuntimePlacement:
    """Placement reporting from Ollama's running-models list."""

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_model_resident_in_vram_reports_gpu(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A loaded model holding VRAM is reported as running on the GPU."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([{"name": "test-model", "size": 900, "size_vram": 900}])

        result = client.check_runtime_placement()

        assert result["placement"] == "gpu"
        assert result["url"] == "http://localhost:11434"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_model_loaded_with_no_vram_reports_cpu(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A loaded model holding zero VRAM is reported as running on the CPU."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([{"name": "test-model", "size": 900, "size_vram": 0}])

        result = client.check_runtime_placement()

        assert result["placement"] == "cpu"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_nothing_loaded_reports_unknown_not_cpu(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """An idle backend reports unknown, because claiming CPU here is wrong on GPU machines."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([])

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_another_model_loaded_reports_unknown(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A different model being loaded says nothing about the configured one."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([{"name": "other-model", "size_vram": 900}])

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_missing_size_vram_reports_unknown(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """An Ollama build that omits the field can't be read as CPU."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([{"name": "test-model", "size": 900}])

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_request_failure_reports_unknown(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A failed placement query degrades to unknown rather than raising."""
        client = _placement_client(mock_config_cls)
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"
        assert result["url"] == "http://localhost:11434"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_non_json_body_reports_unknown(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A 200 carrying a non-JSON body must not escape as a decode error."""
        client = _placement_client(mock_config_cls)
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("not JSON")
        mock_get.return_value = response

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_non_200_reports_unknown(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A backend that answers but refuses the query reports unknown."""
        client = _placement_client(mock_config_cls)
        mock_get.return_value = _ps_response([], status_code=404)

        result = client.check_runtime_placement()

        assert result["placement"] == "unknown"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_untagged_name_matches_latest(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """Placement reuses the availability check's bare-name-to-`:latest` matching."""
        mock_config = MagicMock()
        mock_config.llm_model_name = "llama3"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config
        mock_get.return_value = _ps_response([{"name": "llama3:latest", "size_vram": 900}])

        result = OllamaClient(model_name="llama3").check_runtime_placement()

        assert result["placement"] == "gpu"


class TestContainerizedRemedy:
    """Which pull command the missing-model remedy suggests, per endpoint."""

    @pytest.mark.parametrize(
        ("base_url", "containerized"),
        [
            ("http://ollama:11434", True),
            ("http://127.0.0.1:11435", True),
            ("http://localhost:11435", True),
            ("http://127.0.0.1:11434", False),
            ("http://192.168.1.20:11434", False),
        ],
    )
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_container_detection(self, mock_config_cls: MagicMock, base_url: str, containerized: bool) -> None:
        """The compose DNS name and the container's published host port both identify it.

        A native Ollama on 11434 and a LAN-hosted one must not, since `docker exec` would
        name a container that does not exist there.
        """
        client = _placement_client(mock_config_cls, base_url=base_url)

        assert client._is_ollama_containerized() is containerized

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_remedy_names_container_on_published_port(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A host process pointed at the container's published port gets the docker exec remedy."""
        client = _placement_client(mock_config_cls, base_url="http://127.0.0.1:11435")

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}
        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "other-model"}]}
        mock_get.side_effect = [version_resp, tags_resp]

        result = client.check_model_availability()

        assert result["status"] == "not_found"
        assert result["suggested_action"] == "Run 'docker exec codebase-rag-ollama ollama pull test-model'"

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_remedy_stays_plain_for_native_endpoint(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:
        """A native Ollama on 11434 keeps the plain pull command."""
        client = _placement_client(mock_config_cls, base_url="http://127.0.0.1:11434")

        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.1.0"}
        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {"models": [{"name": "other-model"}]}
        mock_get.side_effect = [version_resp, tags_resp]

        result = client.check_model_availability()

        assert result["suggested_action"] == "Run 'ollama pull test-model'"
