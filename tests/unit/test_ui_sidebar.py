"""Unit tests for app/ui_sidebar.py, with Streamlit mocked out."""

from unittest.mock import MagicMock, patch

from codebase_rag.app.state import SessionState
from codebase_rag.app.ui_sidebar import (
    _delete_chat,
    _display_chat_history_list,
    _display_github_tab,
    _display_ingestion_outcome,
    _display_local_folder_tab,
    _display_new_chat_button,
    _display_repo_list,
    _get_chat_title,
    _ordered_chats,
    _preview_local_folder,
    display_sidebar,
)


def _new_state() -> SessionState:
    state = SessionState(_store={})
    state.ensure_defaults()
    return state


class _AttrDict(dict):
    """A dict that also supports attribute access, like st.session_state."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class TestGetChatTitle:
    def test_empty_history(self) -> None:
        assert _get_chat_title([]) == "New Chat"

    def test_no_user_messages(self) -> None:
        assert _get_chat_title([{"role": "assistant", "content": "hi"}]) == "Empty Chat"

    def test_short_title(self) -> None:
        assert _get_chat_title([{"role": "user", "content": "Hi there"}]) == "Hi there"

    def test_long_title_truncated(self) -> None:
        long = "A" * 30
        assert _get_chat_title([{"role": "user", "content": long}]) == "A" * 20 + "..."


class TestOrderedChats:
    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_orders_by_storage_last_updated_desc(self, mock_get_mgr: MagicMock) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_chat_histories.return_value = [{"chat_id": "c2"}, {"chat_id": "c1"}]
        mock_get_mgr.return_value = mock_mgr

        state = _new_state()
        state.chat_histories["c1"] = [{"role": "user", "content": "old"}]
        state.chat_histories["c2"] = [{"role": "user", "content": "new"}]

        ordered = _ordered_chats(state)
        assert [chat_id for chat_id, _ in ordered] == ["c2", "c1"]

    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_falls_back_to_dict_order_on_storage_error(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.side_effect = OSError("db error")
        state = _new_state()
        state.chat_histories["c1"] = []

        ordered = _ordered_chats(state)
        assert [chat_id for chat_id, _ in ordered] == ["c1"]

    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_new_in_session_chat_not_in_storage_still_appears(self, mock_get_mgr: MagicMock) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_chat_histories.return_value = [{"chat_id": "c1"}]
        mock_get_mgr.return_value = mock_mgr

        state = _new_state()
        state.chat_histories["c1"] = []
        state.chat_histories["c2-not-yet-persisted"] = []

        ordered = _ordered_chats(state)
        assert {chat_id for chat_id, _ in ordered} == {"c1", "c2-not-yet-persisted"}


class TestDeleteChat:
    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_deletes_current_chat_and_switches(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        state = _new_state()
        state.chat_histories["c1"] = [{"role": "user", "content": "hi"}]
        state.chat_histories["c2"] = [{"role": "user", "content": "bye"}]
        state._store["current_chat_id"] = "c1"

        _delete_chat(state, "c1")

        assert "c1" not in state.chat_histories
        assert state.current_chat_id == "c2"

    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_deletes_last_chat_starts_new_one(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        state = _new_state()
        state.chat_histories["c1"] = []
        state._store["current_chat_id"] = "c1"

        _delete_chat(state, "c1")

        assert state.current_chat_id is not None
        assert state.current_chat_id != "c1"

    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    def test_storage_error_does_not_raise(self, mock_get_mgr: MagicMock) -> None:
        mock_mgr = MagicMock()
        mock_mgr.delete_chat_history.side_effect = OSError("boom")
        mock_get_mgr.return_value = mock_mgr

        state = _new_state()
        state.chat_histories["c1"] = []
        state._store["current_chat_id"] = "c2"

        _delete_chat(state, "c1")  # should not raise


class TestDisplayRepoList:
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_empty_shows_info(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        runtime = MagicMock()
        with patch("codebase_rag.app.ui_sidebar.get_repo_list", return_value=[]):
            _display_repo_list(runtime)
        mock_st.info.assert_called_once()

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_lists_repos_with_delete_button(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        col1, col2 = MagicMock(), MagicMock()
        col1.__enter__ = MagicMock(return_value=col1)
        col1.__exit__ = MagicMock()
        col2.button.return_value = False
        mock_st.columns.return_value = [col1, col2]

        runtime = MagicMock()
        with patch("codebase_rag.app.ui_sidebar.get_repo_list", return_value=["repo-a"]):
            _display_repo_list(runtime)

        col1.markdown.assert_called()


class TestGithubTab:
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_invalid_url_sets_error(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        mock_st.text_input.return_value = "not-a-url"
        # First button() call is "Ingest" (True triggers validation); a
        # second "Dismiss" button call only happens once the error is
        # already set, so it must return False or it deletes it immediately.
        mock_st.button.side_effect = [True, False]

        runtime = MagicMock()
        _display_github_tab(runtime, ingestion_running=False)

        assert mock_st.session_state["github_url_error"] == "Please enter a valid GitHub URL"
        runtime.ingestion.start.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_valid_url_starts_ingestion(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        mock_st.text_input.return_value = "https://github.com/owner/repo"
        mock_st.button.return_value = True

        runtime = MagicMock()
        _display_github_tab(runtime, ingestion_running=False)

        runtime.ingestion.start.assert_called_once_with("https://github.com/owner/repo", kind="manual")
        mock_st.rerun.assert_called_once()


class TestPreviewLocalFolder:
    def test_computes_and_caches_result(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")

        session_state: dict = {}
        with patch("codebase_rag.app.ui_sidebar.st") as mock_st:
            mock_st.session_state = session_state
            dirs, count = _preview_local_folder(tmp_path)
            assert count == 1
            assert "src" in dirs
            assert session_state["_folder_preview_cache"]["path"] == str(tmp_path)

            with patch("codebase_rag.data_ingestion.pipeline.count_ingestible_files") as mock_count:
                dirs2, count2 = _preview_local_folder(tmp_path)
                mock_count.assert_not_called()
                assert (dirs2, count2) == (dirs, count)


class TestLocalFolderTab:
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_typed_nonexistent_path_shows_error(self, mock_st: MagicMock) -> None:
        mock_st.session_state = _AttrDict()
        mock_st.button.return_value = False
        mock_st.text_input.return_value = "/does/not/exist"

        runtime = MagicMock()
        runtime.folder_picker.is_open.return_value = False

        _display_local_folder_tab(runtime, ingestion_running=False)

        mock_st.error.assert_any_call("Directory does not exist")

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_browse_already_open_shows_error(self, mock_st: MagicMock) -> None:
        mock_st.session_state = _AttrDict()
        mock_st.button.return_value = True
        mock_st.text_input.return_value = ""

        runtime = MagicMock()
        runtime.folder_picker.is_open.return_value = True
        runtime.folder_picker.open.return_value = None

        _display_local_folder_tab(runtime, ingestion_running=False)

        assert mock_st.session_state["folder_dialog_error"] == "A folder dialog is already open."


class TestDisplayIngestionOutcome:
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_success_toasts_and_acknowledges(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        runtime = MagicMock()
        job = MagicMock(state="succeeded", source="owner/repo")
        runtime.ingestion.last_completed.return_value = job

        _display_ingestion_outcome(runtime)

        mock_st.toast.assert_called_once()
        runtime.ingestion.acknowledge.assert_called_once()
        assert "ingestion_error_banner" not in mock_st.session_state

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_failure_shows_dismissible_banner(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {}
        mock_st.button.return_value = False
        runtime = MagicMock()
        job = MagicMock(state="failed", source="owner/repo", error="boom")
        runtime.ingestion.last_completed.return_value = job

        _display_ingestion_outcome(runtime)

        assert mock_st.session_state["ingestion_error_banner"] == {"source": "owner/repo", "error": "boom"}
        mock_st.error.assert_called_once()

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_dismiss_clears_banner(self, mock_st: MagicMock) -> None:
        mock_st.session_state = {"ingestion_error_banner": {"source": "owner/repo", "error": "boom"}}
        mock_st.button.return_value = True
        runtime = MagicMock()
        runtime.ingestion.last_completed.return_value = None

        _display_ingestion_outcome(runtime)

        assert "ingestion_error_banner" not in mock_st.session_state
        mock_st.rerun.assert_called_once()


class TestNewChatButton:
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_click_starts_a_new_chat(self, mock_st: MagicMock) -> None:
        mock_st.sidebar.button.return_value = True
        state = _new_state()
        state.append_message("user", "hello")
        old_chat_id = state.current_chat_id

        _display_new_chat_button(state)

        assert state.current_chat_id != old_chat_id
        assert state.messages == []
        mock_st.rerun.assert_called_once()

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_no_click_does_nothing(self, mock_st: MagicMock) -> None:
        mock_st.sidebar.button.return_value = False
        state = _new_state()

        _display_new_chat_button(state)

        mock_st.rerun.assert_not_called()


class TestDisplayChatHistoryList:
    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_empty_renders_nothing(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:
        state = _new_state()

        _display_chat_history_list(state)

        mock_st.sidebar.subheader.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar.get_chat_history_manager")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_renders_current_chat_marked(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value.list_chat_histories.return_value = []
        mock_st.session_state = {}
        col1, col2 = MagicMock(), MagicMock()
        col1.button.return_value = False
        col2.button.return_value = False
        mock_st.sidebar.columns.return_value = [col1, col2]

        state = _new_state()
        state.append_message("user", "hello")

        _display_chat_history_list(state)

        mock_st.sidebar.subheader.assert_called_once_with("Chat History")
        assert col1.button.call_args[0][0].startswith("➤")


class TestDisplaySidebar:
    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_renders_about_and_delegates(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        runtime = MagicMock()
        runtime.config.llm_model_name = "test-model"
        state = _new_state()

        display_sidebar(runtime, state)

        mock_st.sidebar.title.assert_called_once_with("About")
        mock_repo_mgmt.assert_called_once_with(runtime)
        mock_new_chat.assert_called_once_with(state)
        mock_history.assert_called_once_with(state)
