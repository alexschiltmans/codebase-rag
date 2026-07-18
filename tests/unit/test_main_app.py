"""Unit tests for app/main.py functions."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from codebase_rag.app.main import (
    _display_chat_interface,
    _display_initialization_status,
    _display_setup_banner,
    _get_rag_chain,
    _run_rag_query,
    _try_initialize_components,
    initialize_app_components,
    initialize_llm,
    initialize_session_state,
    initialize_vector_store,
    load_or_create_bm25_retriever,
    main,
    process_user_query,
    warm_up_vector_store,
)
from codebase_rag.retrieval.bm25_search import BM25Retriever


class TestInitializeSessionState:
    """Tests for initialize_session_state."""

    @patch("codebase_rag.app.main.st")
    def test_sets_defaults_when_empty(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}

        initialize_session_state()

        assert mock_st.session_state["initialized"] is False
        assert mock_st.session_state["initializing"] is False
        assert mock_st.session_state["initialization_error"] is None
        assert mock_st.session_state["should_retry"] is False
        assert mock_st.session_state["retry_count"] == 0
        assert mock_st.session_state["processing_query"] is False
        assert mock_st.session_state["thinking"] is False
        assert mock_st.session_state["query_to_process"] is None

    @patch("codebase_rag.app.main.st")
    def test_preserves_existing_values(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {"initialized": True, "retry_count": 3}

        initialize_session_state()

        assert mock_st.session_state["initialized"] is True
        assert mock_st.session_state["retry_count"] == 3


class TestLoadOrCreateBm25Retriever:
    """Tests for load_or_create_bm25_retriever."""

    @patch("codebase_rag.app.main.st")
    def test_creates_new_when_no_cache(self, _mock_st: MagicMock) -> None:

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            with patch("codebase_rag.app.main.Path", return_value=cache_dir):
                retriever = load_or_create_bm25_retriever()

        assert hasattr(retriever, "search")
        assert hasattr(retriever, "bm25")

    @patch("codebase_rag.app.main.st")
    def test_loads_from_cache(self, _mock_st: MagicMock) -> None:

        sample_docs = [
            Document(page_content="cached doc", metadata={"source": "cache"}),
        ]
        cached_retriever = BM25Retriever(sample_docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            bm25_file = cache_dir / "bm25_retriever.json"
            cached_retriever.save_json(bm25_file)

            with patch("codebase_rag.app.main.Path", return_value=cache_dir):
                result = load_or_create_bm25_retriever()

        assert isinstance(result, BM25Retriever)


class TestInitializeVectorStore:
    """Tests for initialize_vector_store."""

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.QdrantStore")
    def test_returns_qdrant_store(self, mock_qdrant_cls: MagicMock, _mock_st: MagicMock) -> None:

        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_qdrant_cls.return_value = mock_store

        config = MagicMock()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.collection_name = "test"

        result = initialize_vector_store(config)

        assert result is mock_store
        mock_qdrant_cls.assert_called_once_with(host="localhost", port=6333, collection_name="test")

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.QdrantStore")
    def test_handles_missing_collection(self, mock_qdrant_cls: MagicMock, _mock_st: MagicMock) -> None:

        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False
        mock_qdrant_cls.return_value = mock_store

        config = MagicMock()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.collection_name = "test"

        result = initialize_vector_store(config)
        assert result is mock_store


class TestInitializeLlm:
    """Tests for initialize_llm."""

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.OllamaClient")
    def test_returns_llm_client(self, mock_ollama_cls: MagicMock, _mock_st: MagicMock) -> None:

        mock_llm = MagicMock()
        mock_llm.check_connection.return_value = {"status": "connected"}
        mock_llm.check_model_availability.return_value = {"status": "available"}
        mock_ollama_cls.return_value = mock_llm

        config = MagicMock()
        config.llm_model_name = "test-model"
        config.ollama_base_url = "http://localhost:11434"

        result = initialize_llm(config)
        assert result is mock_llm

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.OllamaClient")
    def test_handles_connection_issue(self, mock_ollama_cls: MagicMock, _mock_st: MagicMock) -> None:

        mock_llm = MagicMock()
        mock_llm.check_connection.return_value = {"status": "error", "message": "Connection refused"}
        mock_llm.check_model_availability.return_value = {"status": "error", "message": "Not connected"}
        mock_ollama_cls.return_value = mock_llm

        config = MagicMock()
        config.llm_model_name = "test-model"
        config.ollama_base_url = "http://localhost:11434"

        result = initialize_llm(config)
        assert result is mock_llm


class TestWarmUpVectorStore:
    """Tests for warm_up_vector_store."""

    @patch("codebase_rag.app.main.st")
    def test_successful_warmup(self, _mock_st: MagicMock) -> None:

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [(Document(page_content="test"), 0.9)]

        warm_up_vector_store(mock_retriever)
        mock_retriever.search.assert_called_once()

    @patch("codebase_rag.app.main.st")
    def test_warmup_handles_error(self, _mock_st: MagicMock) -> None:

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = ConnectionError("timeout")

        # Should not raise
        warm_up_vector_store(mock_retriever)


class TestInitializeAppComponents:
    """Tests for initialize_app_components."""

    @patch("codebase_rag.app.main.warm_up_vector_store")
    @patch("codebase_rag.app.main.initialize_llm")
    @patch("codebase_rag.app.main.load_or_create_bm25_retriever")
    @patch("codebase_rag.app.main.initialize_vector_store")
    @patch("codebase_rag.app.main.Config")
    @patch("codebase_rag.app.main.st")
    def test_returns_all_components(
        self,
        mock_st: MagicMock,
        mock_config_cls: MagicMock,
        mock_init_vs: MagicMock,
        mock_load_bm25: MagicMock,
        mock_init_llm: MagicMock,
        mock_warmup: MagicMock,
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_config_cls.get_instance.return_value = MagicMock()
        mock_init_vs.return_value = MagicMock()
        mock_load_bm25.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_init_llm.return_value = mock_llm

        # Clear the cache_resource decorator's cache
        result = initialize_app_components.__wrapped__()

        assert result["rag_chain"] is not None
        assert result["vector_retriever"] is not None
        assert result["bm25_retriever"] is not None
        assert result["hybrid_retriever"] is not None
        assert result["llm"] is mock_llm


class TestProcessUserQuery:
    """Tests for process_user_query."""

    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.st")
    def test_not_initialized(self, mock_st: MagicMock, mock_add: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = False

        process_user_query("test")

        mock_add.assert_called_once()
        assert "initializing" in mock_add.call_args[0][1].lower()

    @patch("codebase_rag.app.main._run_rag_query")
    @patch("codebase_rag.app.main._get_rag_chain")
    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.st")
    def test_with_rag_chain(
        self, mock_st: MagicMock, mock_add: MagicMock, mock_get_chain: MagicMock, mock_run: MagicMock
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_chain = MagicMock()
        mock_get_chain.return_value = mock_chain

        process_user_query("test query")

        mock_run.assert_called_once_with(mock_chain, "test query")

    @patch("codebase_rag.app.main._get_rag_chain")
    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.st")
    def test_no_rag_chain(self, mock_st: MagicMock, mock_add: MagicMock, mock_get_chain: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_get_chain.return_value = None

        process_user_query("test query")

        mock_add.assert_called_once()
        assert "trouble connecting" in mock_add.call_args[0][1].lower()


class TestGetRagChain:
    """Tests for _get_rag_chain."""

    @patch("codebase_rag.app.main.st")
    def test_returns_chain(self, mock_st: MagicMock) -> None:

        mock_chain = MagicMock()
        mock_st.session_state = MagicMock()
        mock_st.session_state.components = {"rag_chain": mock_chain}

        result = _get_rag_chain()
        assert result is mock_chain

    @patch("codebase_rag.app.main.st")
    def test_returns_none_when_no_components(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock(spec=[])  # No 'components' attribute

        result = _get_rag_chain()
        assert result is None


class TestRunRagQuery:
    """Tests for _run_rag_query."""

    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.st")
    def test_successful_query(self, mock_st: MagicMock, mock_add: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a", "sources": []},
            {"role": "user", "content": "current q"},
        ]

        mock_chain = MagicMock()
        mock_chain.run.return_value = {
            "answer": "The answer is 42",
            "sources": [{"id": "1", "file_path": "a.py", "file_name": "a.py"}],
        }

        _run_rag_query(mock_chain, "current q")

        mock_chain.add_user_message.assert_called()
        mock_chain.add_assistant_message.assert_called()
        mock_add.assert_called_once()
        assert mock_add.call_args[0][1] == "The answer is 42"

    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.st")
    def test_query_error(self, mock_st: MagicMock, mock_add: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []

        mock_chain = MagicMock()
        mock_chain.run.side_effect = RuntimeError("LLM error")

        _run_rag_query(mock_chain, "query")

        mock_add.assert_called_once()
        assert "error" in mock_add.call_args[0][1].lower()


class TestTryInitializeComponents:
    """Tests for _try_initialize_components."""

    @patch("codebase_rag.app.main.initialize_app_components")
    @patch("codebase_rag.app.main.st")
    def test_successful_initialization(self, mock_st: MagicMock, mock_init: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        mock_init.return_value = {
            "rag_chain": MagicMock(),
            "vector_retriever": MagicMock(),
            "llm": MagicMock(),
        }

        _try_initialize_components()

        assert mock_st.session_state.initialized is True

    @patch("codebase_rag.app.main.initialize_app_components")
    @patch("codebase_rag.app.main.st")
    def test_initialization_error(self, mock_st: MagicMock, mock_init: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        mock_init.side_effect = ConnectionError("Cannot connect")

        _try_initialize_components()

        assert mock_st.session_state.initialization_error == "Cannot connect"

    @patch("codebase_rag.app.main.initialize_app_components")
    @patch("codebase_rag.app.main.st")
    def test_initialization_partial_failure(self, mock_st: MagicMock, mock_init: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        mock_init.return_value = {
            "rag_chain": None,
            "vector_retriever": None,
            "llm": None,
        }

        _try_initialize_components()

        mock_st.error.assert_called_once()


class TestDisplayInitializationStatus:
    """Tests for _display_initialization_status."""

    @patch("codebase_rag.app.main.st")
    def test_initialized(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.sidebar.__enter__ = MagicMock()
        mock_st.sidebar.__exit__ = MagicMock()

        _display_initialization_status()

    @patch("codebase_rag.app.main.st")
    def test_not_initialized_with_error(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = False
        mock_st.session_state.initialization_error = "some error"
        mock_st.sidebar.__enter__ = MagicMock()
        mock_st.sidebar.__exit__ = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.button.return_value = False

        _display_initialization_status()

        mock_st.error.assert_called()

    @patch("codebase_rag.app.main.st")
    def test_retry_button(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = False
        mock_st.session_state.initialization_error = None
        mock_st.sidebar.__enter__ = MagicMock()
        mock_st.sidebar.__exit__ = MagicMock()
        mock_st.button.return_value = True

        _display_initialization_status()

        assert mock_st.session_state.should_retry is True


class TestDisplaySetupBanner:
    """Tests for _display_setup_banner."""

    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_no_status(self, mock_st: MagicMock, mock_status: MagicMock) -> None:

        mock_status.return_value = None
        _display_setup_banner()
        mock_st.info.assert_not_called()

    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_running_status(self, mock_st: MagicMock, mock_status: MagicMock) -> None:

        mock_status.return_value = {
            "running": True,
            "start_time": time.time() - 10,
            "repo": "https://github.com/test/my-repo",
        }
        _display_setup_banner()
        mock_st.info.assert_called_once()


class TestDisplayChatInterface:
    """Tests for _display_chat_interface."""

    @patch("codebase_rag.app.main.st")
    def test_not_initialized(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = False

        _display_chat_interface()

        mock_st.chat_input.assert_called_once()

    @patch("codebase_rag.app.main.display_chat_history")
    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_initialized_no_auto_ingest(
        self, mock_st: MagicMock, mock_status: MagicMock, mock_display: MagicMock
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.session_state.thinking = False
        mock_st.chat_input.return_value = None
        mock_status.return_value = None

        _display_chat_interface()

        mock_display.assert_called_once()

    @patch("codebase_rag.app.main.display_chat_history")
    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_auto_ingestion_error(self, mock_st: MagicMock, mock_status: MagicMock, mock_display: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.session_state.thinking = False
        mock_st.chat_input.return_value = None
        mock_status.return_value = {"running": False, "error": "Ingestion failed"}

        _display_chat_interface()

        mock_st.warning.assert_called_once()

    @patch("codebase_rag.app.main.add_message")
    @patch("codebase_rag.app.main.display_chat_history")
    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_user_prompt(
        self, mock_st: MagicMock, mock_status: MagicMock, mock_display: MagicMock, mock_add: MagicMock
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.session_state.thinking = False
        mock_st.chat_input.return_value = "What does this do?"
        mock_status.return_value = None

        _display_chat_interface()

        mock_add.assert_called_once_with("user", "What does this do?")
        assert mock_st.session_state.thinking is True

    @patch("codebase_rag.app.main.process_user_query")
    @patch("codebase_rag.app.main.display_chat_history")
    @patch("codebase_rag.app.main.get_auto_ingestion_status")
    @patch("codebase_rag.app.main.st")
    def test_thinking_mode(
        self, mock_st: MagicMock, mock_status: MagicMock, mock_display: MagicMock, mock_process: MagicMock
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.session_state.thinking = True
        mock_st.session_state.query_to_process = "my question"
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()
        mock_status.return_value = None

        _display_chat_interface()

        mock_process.assert_called_once_with("my question")


class TestMain:
    """Tests for main entry point."""

    @patch("codebase_rag.app.main._display_chat_interface")
    @patch("codebase_rag.app.main._display_initialization_status")
    @patch("codebase_rag.app.main.check_and_start_auto_ingestion")
    @patch("codebase_rag.app.main.display_sidebar")
    @patch("codebase_rag.app.main.display_header")
    @patch("codebase_rag.app.main.initialize_chat_history")
    @patch("codebase_rag.app.main.initialize_session_state")
    @patch("codebase_rag.app.main.st")
    def test_main_already_initialized(
        self,
        mock_st: MagicMock,
        mock_init_session: MagicMock,
        mock_init_chat: MagicMock,
        mock_header: MagicMock,
        mock_sidebar: MagicMock,
        mock_auto_ingest: MagicMock,
        mock_display_init: MagicMock,
        mock_display_chat: MagicMock,
    ) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.initialized = True
        mock_st.session_state.initializing = False

        main()

        mock_init_session.assert_called_once()
        mock_init_chat.assert_called_once()
        mock_header.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_auto_ingest.assert_called_once()
        mock_display_chat.assert_called_once()
