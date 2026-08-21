"""Unit tests for app/ui_sidebar.py, with Streamlit mocked out."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from codebase_rag.app.state import SessionState
from codebase_rag.app.ui_sidebar import (
    _about_text,
    _delete_chat,
    _display_chat_history_list,
    _display_github_tab,
    _display_ingestion_outcome,
    _display_local_folder_tab,
    _display_new_chat_button,
    _display_repo_list,
    _folder_dialog_wait_fragment,
    _get_chat_title,
    _ingestion_progress_fragment,
    _ordered_chats,
    _preview_local_folder,
    _retrieval_lines,
    display_sidebar,
)


def _new_state() -> SessionState:
    state = SessionState(_store={})
    state.ensure_defaults()
    return state


class _AttrDict(dict[str, Any]):
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
    @patch("codebase_rag.app.ui_sidebar.list_chat_metadata")
    def test_orders_by_storage_last_updated_desc(self, mock_list_metadata: MagicMock) -> None:
        mock_list_metadata.return_value = [{"chat_id": "c2"}, {"chat_id": "c1"}]

        state = _new_state()
        state.chat_histories["c1"] = [{"role": "user", "content": "old"}]
        state.chat_histories["c2"] = [{"role": "user", "content": "new"}]

        ordered = _ordered_chats(state)
        assert [chat_id for chat_id, _ in ordered] == ["c2", "c1"]

    @patch("codebase_rag.app.ui_sidebar.list_chat_metadata")
    def test_falls_back_to_dict_order_on_storage_error(self, mock_list_metadata: MagicMock) -> None:
        mock_list_metadata.return_value = []
        state = _new_state()
        state.chat_histories["c1"] = []

        ordered = _ordered_chats(state)
        assert [chat_id for chat_id, _ in ordered] == ["c1"]

    @patch("codebase_rag.app.ui_sidebar.list_chat_metadata")
    def test_new_in_session_chat_not_in_storage_still_appears(self, mock_list_metadata: MagicMock) -> None:
        mock_list_metadata.return_value = [{"chat_id": "c1"}]

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
    def test_computes_result(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")

        dirs, count = _preview_local_folder(tmp_path)
        assert count == 1
        assert "src" in dirs

    def test_recomputes_when_folder_contents_change(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        _, count = _preview_local_folder(tmp_path)
        assert count == 1

        (tmp_path / "src" / "second.py").write_text("print('bye')")
        _, count2 = _preview_local_folder(tmp_path)
        assert count2 == 2


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


class TestFolderDialogWaitFragment:
    """Exercises the fragment function directly (via ``__wrapped__``), since
    AppTest never runs a standalone ``run_every`` fragment tick on its own;
    only inline, the one time it's first encountered in a script run."""

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_reruns_when_a_pick_lands_even_if_still_open(self, mock_st: MagicMock) -> None:
        """FolderPicker is process-wide: another session's dialog can keep
        is_open() True for a session whose own pick already landed. The
        fragment must still trigger a rerun so the new path gets redrawn."""
        mock_st.session_state = _AttrDict(folder_dialog_token=object())

        runtime = MagicMock()
        runtime.folder_picker.is_open.return_value = True
        runtime.folder_picker.poll.return_value = MagicMock(path="/picked/dir", error=None)

        cast(Any, _folder_dialog_wait_fragment).__wrapped__(runtime)

        assert mock_st.session_state["typed_folder_path"] == "/picked/dir"
        mock_st.rerun.assert_called_once_with(scope="app")

    @patch("codebase_rag.app.ui_sidebar.st")
    def test_shows_waiting_caption_when_nothing_landed_and_still_open(self, mock_st: MagicMock) -> None:
        mock_st.session_state = _AttrDict(folder_dialog_token=object())

        runtime = MagicMock()
        runtime.folder_picker.is_open.return_value = True
        runtime.folder_picker.poll.return_value = None

        cast(Any, _folder_dialog_wait_fragment).__wrapped__(runtime)

        mock_st.caption.assert_called_once()
        mock_st.rerun.assert_not_called()


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


class TestIngestionProgressFragment:
    """Exercises the fragment directly (via ``__wrapped__``), same pattern as
    ``TestFolderDialogWaitFragment``."""

    @patch("codebase_rag.app.ui_sidebar._display_add_repository")
    @patch("codebase_rag.app.ui_sidebar._display_repo_list")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_renders_proportional_bar_with_phase_and_counts(
        self, mock_st: MagicMock, mock_repo_list: MagicMock, mock_add_repo: MagicMock
    ) -> None:
        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_st.status.return_value = mock_status
        mock_st.button.return_value = False

        runtime = MagicMock()
        job = MagicMock(source="owner/repo", started_at=0, phase="indexing", progress_current=3, progress_total=10)
        runtime.ingestion.current_job.return_value = job

        cast(Any, _ingestion_progress_fragment).__wrapped__(runtime)

        mock_st.progress.assert_called_once_with(0.3)
        mock_st.caption.assert_called_once_with("indexing - 3/10")

    @patch("codebase_rag.app.ui_sidebar._display_add_repository")
    @patch("codebase_rag.app.ui_sidebar._display_repo_list")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_guards_zero_total_by_skipping_the_bar(
        self, mock_st: MagicMock, mock_repo_list: MagicMock, mock_add_repo: MagicMock
    ) -> None:
        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_st.status.return_value = mock_status
        mock_st.button.return_value = False

        runtime = MagicMock()
        job = MagicMock(source="owner/repo", started_at=0, phase="", progress_current=0, progress_total=0)
        runtime.ingestion.current_job.return_value = job

        cast(Any, _ingestion_progress_fragment).__wrapped__(runtime)

        mock_st.progress.assert_not_called()
        mock_st.caption.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar._display_add_repository")
    @patch("codebase_rag.app.ui_sidebar._display_repo_list")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_cancel_button_calls_manager_cancel(
        self, mock_st: MagicMock, mock_repo_list: MagicMock, mock_add_repo: MagicMock
    ) -> None:
        mock_status = MagicMock()
        mock_status.__enter__ = MagicMock(return_value=mock_status)
        mock_status.__exit__ = MagicMock(return_value=False)
        mock_st.status.return_value = mock_status
        mock_st.button.return_value = True

        runtime = MagicMock()
        job = MagicMock(source="owner/repo", started_at=0, phase="processing", progress_current=1, progress_total=2)
        runtime.ingestion.current_job.return_value = job

        cast(Any, _ingestion_progress_fragment).__wrapped__(runtime)

        runtime.ingestion.cancel.assert_called_once()


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


class TestAboutText:
    """Tests for the About block, which reports the configured stack."""

    @staticmethod
    def _runtime(**overrides: Any) -> MagicMock:
        runtime = MagicMock()
        settings: dict[str, Any] = {
            "provider": "ollama",
            "llm_model_name": "test-model",
            "retriever": "bm25",
            "rerank_enabled": False,
            "rerank_model": "a-reranker",
            "rewrite_enabled": False,
        }
        settings.update(overrides)
        runtime.config = SimpleNamespace(**settings)
        runtime.health = {}
        return runtime

    def test_bm25_does_not_claim_hybrid_search(self) -> None:
        """The default retriever is BM25, so the block must not advertise vector search."""
        lines = _retrieval_lines(self._runtime())
        assert any("BM25" in line for line in lines)
        assert not any("Hybrid" in line for line in lines)

    def test_hybrid_names_both_halves(self) -> None:
        """Under hybrid, the vector store is queried rather than only written to."""
        lines = _retrieval_lines(self._runtime(retriever="hybrid"))
        assert "- Hybrid search, fusing vector similarity with BM25" in lines
        assert "- Qdrant vector database, holding the embedded chunks" in lines

    def test_optional_stages_appear_only_when_enabled(self) -> None:
        """Rerank and rewrite are off by default and listed once turned on."""
        off = _retrieval_lines(self._runtime())
        assert not any("rerank" in line.lower() or "rewriting" in line.lower() for line in off)

        on = _retrieval_lines(self._runtime(rerank_enabled=True, rewrite_enabled=True))
        assert any("a-reranker" in line for line in on)
        assert any("Query rewriting" in line for line in on)

    def test_paragraphs_stay_on_one_line_each(self) -> None:
        """Markdown folds a single newline into a space, so a wrapped paragraph would reflow oddly."""
        body, _, bullets = _about_text(self._runtime()).partition("This application uses:")
        assert [line for line in body.splitlines() if line] == [
            (
                "Codebase RAG is a Retrieval-Augmented Generation application for exploring and "
                "understanding codebases locally."
            ),
            "It helps users understand code by providing answers based on ingested documentation and source code.",
        ]
        assert all(line.startswith("- ") for line in bullets.splitlines() if line)


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
        runtime.health = {}
        state = _new_state()

        display_sidebar(runtime, state)

        mock_st.sidebar.title.assert_called_once_with("About")
        mock_repo_mgmt.assert_called_once_with(runtime)
        mock_new_chat.assert_called_once_with(state)
        mock_history.assert_called_once_with(state)

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_renders_model_warning_when_not_found(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        runtime = MagicMock()
        runtime.config.llm_model_name = "my-model"
        runtime.health = {
            "model": {
                "status": "not_found",
                "message": "Model 'my-model' not found",
                "suggested_action": "Run 'docker exec codebase-rag-ollama ollama pull my-model'",
            }
        }
        state = _new_state()

        display_sidebar(runtime, state)

        mock_st.sidebar.warning.assert_called_once()
        warning_text = mock_st.sidebar.warning.call_args[0][0]
        assert "my-model" in warning_text
        assert "docker exec codebase-rag-ollama ollama pull my-model" in warning_text
        assert "refreshes on app restart" in warning_text

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_renders_warning_when_backend_unreachable(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """The most common failure (backend down) previously showed nothing: check_model_availability
        returns status "error" in this case, not "not_found", which the sidebar didn't handle at all.
        """
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.ollama_base_url = "http://localhost:11434"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {"model": {"status": "error", "message": "Cannot connect to Ollama at http://x"}}
        state = _new_state()

        display_sidebar(runtime, state)

        mock_st.sidebar.warning.assert_called_once()
        warning_text = mock_st.sidebar.warning.call_args[0][0]
        assert "http://localhost:11434" in warning_text
        assert "Cannot connect to Ollama at http://x" in warning_text
        assert "refreshes on app restart" in warning_text

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_about_omits_endpoint_before_health_reports(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """The first paint happens before the background check lands and must still render."""
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {}

        display_sidebar(runtime, _new_state())

        about_text = mock_st.sidebar.markdown.call_args[0][0]
        assert "A local LLM via Ollama (**my-model**)" in about_text
        assert "http://" not in about_text

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_about_states_endpoint_and_gpu_placement(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """Which of two reachable backends answered is readable off the screen."""
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {
            "connection": {"status": "connected", "url": "http://127.0.0.1:11434"},
            "model": {"status": "available"},
            "placement": {"placement": "gpu"},
        }

        display_sidebar(runtime, _new_state())

        about_text = mock_st.sidebar.markdown.call_args[0][0]
        assert "http://127.0.0.1:11434" in about_text
        assert "on the GPU" in about_text

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_cpu_placement_is_stated_without_a_warning(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """CPU inference is a legitimate configuration, so it is a fact and not an alert."""
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {
            "connection": {"status": "connected", "url": "http://127.0.0.1:11435"},
            "model": {"status": "available"},
            "placement": {"placement": "cpu"},
        }

        display_sidebar(runtime, _new_state())

        about_text = mock_st.sidebar.markdown.call_args[0][0]
        assert "http://127.0.0.1:11435" in about_text
        assert "on the CPU" in about_text
        mock_st.sidebar.warning.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_unknown_placement_shows_endpoint_alone(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """Nothing loaded yet means no placement claim, not a CPU claim."""
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {
            "connection": {"status": "connected", "url": "http://127.0.0.1:11434"},
            "model": {"status": "available"},
            "placement": {"placement": "unknown"},
        }

        display_sidebar(runtime, _new_state())

        about_text = mock_st.sidebar.markdown.call_args[0][0]
        assert "http://127.0.0.1:11434" in about_text
        assert "on the CPU" not in about_text
        assert "on the GPU" not in about_text

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_endpoint_shown_alongside_the_missing_model_warning(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        """The endpoint is the fact that identifies which backend is missing the model."""
        runtime = MagicMock()
        runtime.config.provider = "ollama"
        runtime.config.llm_model_name = "my-model"
        runtime.health = {
            "connection": {"status": "connected", "url": "http://127.0.0.1:11435"},
            "model": {
                "status": "not_found",
                "message": "Model 'my-model' not found",
                "suggested_action": "Run 'docker exec codebase-rag-ollama ollama pull my-model'",
            },
            "placement": {"placement": "unknown"},
        }

        display_sidebar(runtime, _new_state())

        assert "http://127.0.0.1:11435" in mock_st.sidebar.markdown.call_args[0][0]
        mock_st.sidebar.warning.assert_called_once()

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_no_warning_when_model_available(
        self, mock_st: MagicMock, mock_repo_mgmt: MagicMock, mock_new_chat: MagicMock, mock_history: MagicMock
    ) -> None:
        runtime = MagicMock()
        runtime.config.llm_model_name = "my-model"
        runtime.health = {"model": {"status": "available"}}
        state = _new_state()

        display_sidebar(runtime, state)

        mock_st.sidebar.warning.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar._display_model_health")
    @patch("codebase_rag.app.ui_sidebar._model_health_fragment")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_polls_for_health_until_the_check_reports(
        self,
        mock_st: MagicMock,
        mock_fragment: MagicMock,
        mock_display_health: MagicMock,
        mock_repo_mgmt: MagicMock,
        mock_new_chat: MagicMock,
        mock_history: MagicMock,
    ) -> None:
        """First paint beats the health thread, so something has to bring the banner back.

        Without the fragment a cold start with a missing model shows no banner until an
        unrelated interaction reruns the script, which the change's own capability rules out.
        """
        runtime = MagicMock()
        runtime.config.llm_model_name = "my-model"
        runtime.health = {}

        display_sidebar(runtime, _new_state())

        mock_fragment.assert_called_once_with(runtime)
        mock_display_health.assert_not_called()

    @patch("codebase_rag.app.ui_sidebar._display_chat_history_list")
    @patch("codebase_rag.app.ui_sidebar._display_new_chat_button")
    @patch("codebase_rag.app.ui_sidebar._display_repo_management")
    @patch("codebase_rag.app.ui_sidebar._display_model_health")
    @patch("codebase_rag.app.ui_sidebar._model_health_fragment")
    @patch("codebase_rag.app.ui_sidebar.st")
    def test_stops_polling_once_health_is_populated(
        self,
        mock_st: MagicMock,
        mock_fragment: MagicMock,
        mock_display_health: MagicMock,
        mock_repo_mgmt: MagicMock,
        mock_new_chat: MagicMock,
        mock_history: MagicMock,
    ) -> None:
        """The poll is bounded: no run_every fragment survives the first result."""
        runtime = MagicMock()
        runtime.config.llm_model_name = "my-model"
        runtime.health = {"model": {"status": "available"}}

        display_sidebar(runtime, _new_state())

        mock_fragment.assert_not_called()
        mock_display_health.assert_called_once_with(runtime)
