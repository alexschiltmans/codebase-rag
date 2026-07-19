"""Unit tests for app/components.py — functions not yet covered."""

from unittest.mock import MagicMock, patch

import codebase_rag.app.components as comp
from codebase_rag.app.components import (
    _delete_chat,
    _display_repo_list,
    _get_chat_title,
    _get_qdrant_store,
    _load_chat_into_session,
    _load_most_recent_chat,
    _load_repo_list,
    _load_saved_chat_histories,
    _switch_to_next_chat,
    add_message,
    display_header,
    display_sidebar,
    initialize_chat_history,
)


class TestAddMessage:
    """Tests for add_message function."""

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_add_user_message(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []
        mock_st.session_state.chat_histories = {"chat1": []}
        mock_st.session_state.current_chat_id = "chat1"

        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr

        add_message("user", "Hello!")

        assert len(mock_st.session_state.messages) == 1
        assert mock_st.session_state.messages[0]["role"] == "user"
        assert mock_st.session_state.messages[0]["content"] == "Hello!"

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_add_assistant_message_with_sources(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []
        mock_st.session_state.chat_histories = {"chat1": []}
        mock_st.session_state.current_chat_id = "chat1"

        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr

        sources = [{"id": "1", "file_path": "test.py", "file_name": "test.py"}]
        add_message("assistant", "Response here", sources=sources)

        msg = mock_st.session_state.messages[0]
        assert msg["role"] == "assistant"
        assert msg["sources"] == sources
        mock_mgr.save_chat_history.assert_called_once()

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_add_message_empty_content_replaced(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []
        mock_st.session_state.chat_histories = {"chat1": []}
        mock_st.session_state.current_chat_id = "chat1"
        mock_get_mgr.return_value = MagicMock()

        add_message("assistant", "")

        msg = mock_st.session_state.messages[0]
        assert "apologize" in msg["content"]

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_add_message_handles_storage_error(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = []
        mock_st.session_state.chat_histories = {"chat1": []}
        mock_st.session_state.current_chat_id = "chat1"

        mock_mgr = MagicMock()
        mock_mgr.save_chat_history.side_effect = OSError("disk full")
        mock_get_mgr.return_value = mock_mgr

        # Should not raise
        add_message("user", "test message")
        assert len(mock_st.session_state.messages) == 1


class TestInitializeChatHistory:
    """Tests for initialize_chat_history."""

    @patch("codebase_rag.app.components._load_saved_chat_histories")
    @patch("codebase_rag.app.components.st")
    def test_initializes_empty_state(self, mock_st: MagicMock, mock_load: MagicMock) -> None:

        # Use a real dict-like object that supports `not in` and attribute access
        class SessionState(dict):
            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = value

        mock_st.session_state = SessionState()

        initialize_chat_history()

        assert mock_st.session_state["messages"] == []
        assert "chat_histories" in mock_st.session_state
        assert "current_chat_id" in mock_st.session_state
        mock_load.assert_called_once()


class TestGetChatTitle:
    """Tests for _get_chat_title."""

    def test_empty_history(self) -> None:

        assert _get_chat_title([]) == "New Chat"

    def test_no_user_messages(self) -> None:

        result = _get_chat_title([{"role": "assistant", "content": "Hello"}])
        assert result == "Empty Chat"

    def test_short_title(self) -> None:

        result = _get_chat_title([{"role": "user", "content": "Hi there"}])
        assert result == "Hi there"

    def test_long_title_truncated(self) -> None:

        long = "A" * 30
        result = _get_chat_title([{"role": "user", "content": long}])
        assert result == "A" * 20 + "..."


class TestGetAutoIngestionStatus:
    """Tests for get_auto_ingestion_status."""

    def test_returns_none_when_not_attempted(self) -> None:

        original = comp._auto_ingest_attempted
        comp._auto_ingest_attempted = False
        try:
            result = comp.get_auto_ingestion_status()
            assert result is None
        finally:
            comp._auto_ingest_attempted = original

    def test_returns_status_when_attempted(self) -> None:

        original_attempted = comp._auto_ingest_attempted
        original_status = dict(comp._ingestion_status)
        original_error = comp._auto_ingest_error

        comp._auto_ingest_attempted = True
        comp._ingestion_status.clear()
        comp._auto_ingest_error = None

        try:
            result = comp.get_auto_ingestion_status()
            assert result == {"running": False}
        finally:
            comp._auto_ingest_attempted = original_attempted
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original_status)
            comp._auto_ingest_error = original_error

    def test_manual_ingestion_does_not_report_as_auto_running(self) -> None:
        """A manual ingest occupying the shared status slot must not make
        the chat interface think auto-ingestion (and its first-boot banner)
        is still in progress."""

        original_attempted = comp._auto_ingest_attempted
        original_status = dict(comp._ingestion_status)
        original_error = comp._auto_ingest_error

        comp._auto_ingest_attempted = True
        comp._ingestion_status.clear()
        comp._ingestion_status.update(running=True, repo="manual-repo", kind="manual")
        comp._auto_ingest_error = None

        try:
            result = comp.get_auto_ingestion_status()
            assert result == {"running": False}
        finally:
            comp._auto_ingest_attempted = original_attempted
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original_status)
            comp._auto_ingest_error = original_error

    def test_returns_error_when_present(self) -> None:

        original_attempted = comp._auto_ingest_attempted
        original_status = dict(comp._ingestion_status)
        original_error = comp._auto_ingest_error

        comp._auto_ingest_attempted = True
        comp._ingestion_status.clear()
        comp._auto_ingest_error = "something failed"

        try:
            result = comp.get_auto_ingestion_status()
            assert result is not None
            assert result["error"] == "something failed"
        finally:
            comp._auto_ingest_attempted = original_attempted
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original_status)
            comp._auto_ingest_error = original_error


class TestCheckAndStartAutoIngestion:
    """Tests for check_and_start_auto_ingestion."""

    @patch("codebase_rag.app.components._run_ingestion")
    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.Config")
    def test_skips_when_already_attempted(
        self, mock_config: MagicMock, mock_store_fn: MagicMock, mock_ingest: MagicMock
    ) -> None:

        original = comp._auto_ingest_attempted
        comp._auto_ingest_attempted = True
        try:
            comp.check_and_start_auto_ingestion()
            mock_ingest.assert_not_called()
        finally:
            comp._auto_ingest_attempted = original

    @patch("codebase_rag.app.components._run_ingestion")
    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.Config")
    def test_skips_when_no_default_repo(
        self, mock_config_cls: MagicMock, mock_store_fn: MagicMock, mock_ingest: MagicMock
    ) -> None:

        original = comp._auto_ingest_attempted
        comp._auto_ingest_attempted = False

        mock_config = MagicMock()
        mock_config.default_repo_url = ""
        mock_config_cls.get_instance.return_value = mock_config

        try:
            comp.check_and_start_auto_ingestion()
            mock_ingest.assert_not_called()
        finally:
            comp._auto_ingest_attempted = original

    @patch("codebase_rag.app.components._run_ingestion")
    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.Config")
    def test_skips_when_repos_exist(
        self, mock_config_cls: MagicMock, mock_store_fn: MagicMock, mock_ingest: MagicMock
    ) -> None:

        original = comp._auto_ingest_attempted
        comp._auto_ingest_attempted = False

        mock_config = MagicMock()
        mock_config.default_repo_url = "https://github.com/test/repo"
        mock_config_cls.get_instance.return_value = mock_config

        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_store.list_repos.return_value = ["existing-repo"]
        mock_store_fn.return_value = mock_store

        try:
            comp.check_and_start_auto_ingestion()
            mock_ingest.assert_not_called()
        finally:
            comp._auto_ingest_attempted = original

    @patch("codebase_rag.app.components._run_ingestion")
    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.Config")
    def test_starts_ingestion_when_empty(
        self, mock_config_cls: MagicMock, mock_store_fn: MagicMock, mock_ingest: MagicMock
    ) -> None:

        original = comp._auto_ingest_attempted
        comp._auto_ingest_attempted = False

        mock_config = MagicMock()
        mock_config.default_repo_url = "https://github.com/test/repo"
        mock_config_cls.get_instance.return_value = mock_config

        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False
        mock_store_fn.return_value = mock_store

        try:
            comp.check_and_start_auto_ingestion()
            mock_ingest.assert_called_once_with("https://github.com/test/repo", kind="auto")
        finally:
            comp._auto_ingest_attempted = original


class TestDisplayHeader:
    """Tests for display_header."""

    @patch("codebase_rag.app.components.st")
    def test_display_header(self, mock_st: MagicMock) -> None:

        display_header()

        mock_st.title.assert_called_once_with("Codebase RAG")
        assert mock_st.markdown.call_count >= 2


class TestLoadSavedChatHistories:
    """Tests for _load_saved_chat_histories."""

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_handles_empty_chat_list(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_mgr = MagicMock()
        mock_mgr.list_chat_histories.return_value = []
        mock_get_mgr.return_value = mock_mgr

        _load_saved_chat_histories()

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_handles_storage_error(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_get_mgr.side_effect = OSError("db error")

        # Should not raise
        _load_saved_chat_histories()

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_loads_chats_from_storage(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {}

        mock_mgr = MagicMock()
        mock_mgr.list_chat_histories.return_value = [
            {"chat_id": "c1"},
            {"chat_id": "c2"},
        ]
        mock_mgr.get_chat_history.side_effect = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "bye"}],
        ]
        mock_get_mgr.return_value = mock_mgr

        _load_saved_chat_histories()

        assert "c1" in mock_st.session_state.chat_histories
        assert "c2" in mock_st.session_state.chat_histories


class TestSwitchToNextChat:
    """Tests for _switch_to_next_chat."""

    @patch("codebase_rag.app.components.st")
    def test_switches_to_existing_chat(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {"c2": [{"role": "user", "content": "hey"}]}

        _switch_to_next_chat()

        assert mock_st.session_state.current_chat_id == "c2"

    @patch("codebase_rag.app.components.st")
    def test_creates_new_when_none_remain(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {}

        _switch_to_next_chat()

        assert mock_st.session_state.messages == []


class TestDisplaySidebar:
    """Tests for display_sidebar."""

    @patch("codebase_rag.app.components._display_chat_history_list")
    @patch("codebase_rag.app.components._display_new_chat_button")
    @patch("codebase_rag.app.components._display_repo_management")
    @patch("codebase_rag.app.components.Config")
    @patch("codebase_rag.app.components.st")
    def test_display_sidebar(
        self,
        mock_st: MagicMock,
        mock_config_cls: MagicMock,
        mock_repo: MagicMock,
        mock_new_chat: MagicMock,
        mock_history: MagicMock,
    ) -> None:

        mock_config = MagicMock()
        mock_config.llm_model_name = "test-model"
        mock_config_cls.get_instance.return_value = mock_config

        mock_st.sidebar = MagicMock()
        mock_st.sidebar.__enter__ = MagicMock()
        mock_st.sidebar.__exit__ = MagicMock()

        display_sidebar()

        mock_st.sidebar.title.assert_called_once()
        mock_st.sidebar.markdown.assert_called()


class TestDisplayIngestionStatus:
    """Tests for _display_ingestion_status."""

    @patch("codebase_rag.app.components.st")
    def test_no_status(self, mock_st: MagicMock) -> None:

        original = dict(comp._ingestion_status)
        comp._ingestion_status.clear()
        try:
            comp._display_ingestion_status()
            mock_st.info.assert_not_called()
        finally:
            comp._ingestion_status.update(original)

    @patch("codebase_rag.app.components.time")
    @patch("codebase_rag.app.components.st")
    def test_running_status(self, mock_st: MagicMock, mock_time: MagicMock) -> None:

        mock_time.time.return_value = 110.0
        original = dict(comp._ingestion_status)
        comp._ingestion_status.update(running=True, repo="test-repo", start_time=100.0)
        try:
            comp._display_ingestion_status()
            mock_st.info.assert_called_once()
            call_arg = mock_st.info.call_args[0][0]
            assert "test-repo" in call_arg
        finally:
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original)

    @patch("codebase_rag.app.components.st")
    def test_error_status(self, mock_st: MagicMock) -> None:

        original = dict(comp._ingestion_status)
        comp._ingestion_status.update(running=False, error="timeout")
        try:
            comp._display_ingestion_status()
            mock_st.error.assert_called_once()
        finally:
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original)

    @patch("codebase_rag.app.components.st")
    def test_success_status(self, mock_st: MagicMock) -> None:

        original = dict(comp._ingestion_status)
        comp._ingestion_status.update(running=False, repo="test-repo", error=None)
        mock_st.cache_resource = MagicMock()
        mock_st.session_state = MagicMock()
        try:
            comp._display_ingestion_status()
            mock_st.success.assert_called_once()
        finally:
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original)


class TestLoadRepoList:
    """Tests for _load_repo_list."""

    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.st")
    def test_returns_repos(self, mock_st: MagicMock, mock_store_fn: MagicMock) -> None:

        mock_store = MagicMock()
        mock_store.list_repos.return_value = ["repo-a", "repo-b"]
        mock_store_fn.return_value = mock_store

        result = _load_repo_list()
        assert result == ["repo-a", "repo-b"]

    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.st")
    def test_handles_connection_error(self, mock_st: MagicMock, mock_store_fn: MagicMock) -> None:

        mock_store_fn.side_effect = ConnectionError("refused")

        result = _load_repo_list()
        assert result == []
        mock_st.warning.assert_called_once()


class TestDisplayRepoList:
    """Tests for _display_repo_list."""

    @patch("codebase_rag.app.components.st")
    def test_empty_repos(self, mock_st: MagicMock) -> None:

        _display_repo_list([])
        mock_st.info.assert_called_once()

    @patch("codebase_rag.app.components._get_qdrant_store")
    @patch("codebase_rag.app.components.st")
    def test_with_repos(self, mock_st: MagicMock, mock_store_fn: MagicMock) -> None:

        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        col1 = MagicMock()
        col2 = MagicMock()
        col1.__enter__ = MagicMock(return_value=col1)
        col1.__exit__ = MagicMock()
        col2.__enter__ = MagicMock(return_value=col2)
        col2.__exit__ = MagicMock()
        mock_st.columns.return_value = [col1, col2]
        mock_st.button.return_value = False

        _display_repo_list(["repo-a"])

        col1.markdown.assert_called()


class TestLoadMostRecentChat:
    """Tests for _load_most_recent_chat."""

    @patch("codebase_rag.app.components.st")
    def test_loads_chat(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {}

        mock_mgr = MagicMock()
        mock_mgr.get_chat_history.return_value = [{"role": "user", "content": "hi"}]

        _load_most_recent_chat(mock_mgr, {"chat_id": "c1"})

        assert mock_st.session_state.current_chat_id == "c1"

    @patch("codebase_rag.app.components.st")
    def test_skips_no_chat_id(self, mock_st: MagicMock) -> None:

        mock_mgr = MagicMock()
        _load_most_recent_chat(mock_mgr, {})
        mock_mgr.get_chat_history.assert_not_called()

    @patch("codebase_rag.app.components.st")
    def test_skips_empty_messages(self, mock_st: MagicMock) -> None:

        mock_mgr = MagicMock()
        mock_mgr.get_chat_history.return_value = []

        _load_most_recent_chat(mock_mgr, {"chat_id": "c1"})


class TestLoadChatIntoSession:
    """Tests for _load_chat_into_session."""

    @patch("codebase_rag.app.components.st")
    def test_loads_chat(self, mock_st: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {}

        mock_mgr = MagicMock()
        mock_mgr.get_chat_history.return_value = [{"role": "user", "content": "msg"}]

        _load_chat_into_session(mock_mgr, {"chat_id": "c2"})

        assert "c2" in mock_st.session_state.chat_histories

    @patch("codebase_rag.app.components.st")
    def test_skips_no_id(self, mock_st: MagicMock) -> None:

        mock_mgr = MagicMock()
        _load_chat_into_session(mock_mgr, {})
        mock_mgr.get_chat_history.assert_not_called()


class TestDeleteChat:
    """Tests for _delete_chat."""

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components._switch_to_next_chat")
    @patch("codebase_rag.app.components.st")
    def test_deletes_current_chat(self, mock_st: MagicMock, mock_switch: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {"c1": [{"role": "user", "content": "hi"}]}
        mock_st.session_state.current_chat_id = "c1"
        mock_get_mgr.return_value = MagicMock()

        _delete_chat("c1")

        mock_switch.assert_called_once()

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_deletes_non_current_chat(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {
            "c1": [{"role": "user", "content": "hi"}],
            "c2": [{"role": "user", "content": "bye"}],
        }
        mock_st.session_state.current_chat_id = "c1"
        mock_get_mgr.return_value = MagicMock()

        _delete_chat("c2")

    @patch("codebase_rag.app.components.get_chat_history_manager")
    @patch("codebase_rag.app.components.st")
    def test_delete_handles_storage_error(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:

        mock_st.session_state = MagicMock()
        mock_st.session_state.chat_histories = {"c1": []}
        mock_st.session_state.current_chat_id = "c2"

        mock_mgr = MagicMock()
        mock_mgr.delete_chat_history.side_effect = OSError("storage error")
        mock_get_mgr.return_value = mock_mgr

        # Should not raise
        _delete_chat("c1")


class TestPreviewLocalFolder:
    """Tests for _preview_local_folder."""

    def test_computes_and_caches_result(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")

        session_state: dict = {}
        with patch("codebase_rag.app.components.st") as mock_st:
            mock_st.session_state = session_state

            dirs, count = comp._preview_local_folder(tmp_path)
            assert count == 1
            assert "src" in dirs
            assert session_state["_folder_preview_cache"]["path"] == str(tmp_path)

            # Second call for the same path should hit the cache instead of
            # re-walking the filesystem.
            with patch("codebase_rag.data_ingestion.pipeline.count_ingestible_files") as mock_count:
                dirs2, count2 = comp._preview_local_folder(tmp_path)
                mock_count.assert_not_called()
                assert (dirs2, count2) == (dirs, count)


class TestGetQdrantStore:
    """Tests for _get_qdrant_store."""

    @patch("codebase_rag.app.components.Config")
    def test_returns_store(self, mock_config_cls: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "test"
        mock_config_cls.get_instance.return_value = mock_config

        with patch("codebase_rag.database.qdrant_store.QdrantStore") as mock_qs:
            mock_store = MagicMock()
            mock_qs.return_value = mock_store

            result = _get_qdrant_store()
            assert result is mock_store


class TestRunIngestion:
    """Tests for _run_ingestion."""

    @patch("codebase_rag.app.components.threading")
    @patch("codebase_rag.app.components.st")
    def test_starts_thread(self, mock_st: MagicMock, mock_threading: MagicMock) -> None:

        original = dict(comp._ingestion_status)
        comp._ingestion_status.clear()
        try:
            comp._run_ingestion("https://github.com/test/repo")

            mock_threading.Thread.assert_called_once()
            mock_threading.Thread.return_value.start.assert_called_once()
            assert comp._ingestion_status["running"] is True
            assert comp._ingestion_status["kind"] == "manual"
        finally:
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original)

    @patch("codebase_rag.app.components.threading")
    @patch("codebase_rag.app.components.st")
    def test_refuses_to_start_when_already_running(self, mock_st: MagicMock, mock_threading: MagicMock) -> None:

        original = dict(comp._ingestion_status)
        comp._ingestion_status.clear()
        comp._ingestion_status.update(running=True, repo="already-running-repo", kind="manual")
        try:
            comp._run_ingestion("https://github.com/test/other-repo")

            mock_threading.Thread.assert_not_called()
            assert comp._ingestion_status["repo"] == "already-running-repo"
        finally:
            comp._ingestion_status.clear()
            comp._ingestion_status.update(original)
