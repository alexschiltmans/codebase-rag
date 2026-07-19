"""Unit tests for app/main.py, with Streamlit mocked out."""

from unittest.mock import MagicMock, patch

from codebase_rag.app.main import _display_auto_ingest_gate, _restore_saved_chats
from codebase_rag.app.state import SessionState


def _new_state() -> SessionState:
    state = SessionState(_store={})
    state.ensure_defaults()
    return state


class TestDisplayAutoIngestGate:
    @patch("codebase_rag.app.main.st")
    def test_no_job_no_error_does_not_gate(self, mock_st: MagicMock) -> None:
        runtime = MagicMock()
        runtime.ingestion.current_job.return_value = None
        runtime.ingestion.auto_job_error.return_value = None

        assert _display_auto_ingest_gate(runtime) is False
        mock_st.warning.assert_not_called()
        mock_st.info.assert_not_called()

    @patch("codebase_rag.app.main.st")
    def test_running_auto_job_gates_and_shows_banner(self, mock_st: MagicMock) -> None:
        runtime = MagicMock()
        job = MagicMock(kind="auto", source="https://github.com/owner/default-repo")
        runtime.ingestion.current_job.return_value = job

        assert _display_auto_ingest_gate(runtime) is True
        mock_st.info.assert_called_once()

    @patch("codebase_rag.app.main.st")
    def test_running_manual_job_does_not_gate(self, mock_st: MagicMock) -> None:
        """Regression test for FE-2: a running manual job must never show
        the first-boot gate meant for the default repo."""
        runtime = MagicMock()
        job = MagicMock(kind="manual", source="https://github.com/owner/other-repo")
        runtime.ingestion.current_job.return_value = job
        runtime.ingestion.auto_job_error.return_value = None

        assert _display_auto_ingest_gate(runtime) is False
        mock_st.info.assert_not_called()

    @patch("codebase_rag.app.main.st")
    def test_failed_auto_job_shows_warning_not_default_repo_wording_for_manual(self, mock_st: MagicMock) -> None:
        runtime = MagicMock()
        runtime.ingestion.current_job.return_value = None
        runtime.ingestion.auto_job_error.return_value = "connection refused"

        assert _display_auto_ingest_gate(runtime) is False
        mock_st.warning.assert_called_once()
        assert "connection refused" in mock_st.warning.call_args[0][0]


class TestRestoreSavedChats:
    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.get_chat_history_manager")
    def test_skips_if_already_restored_this_session(self, mock_get_mgr: MagicMock, mock_st: MagicMock) -> None:
        state = _new_state()
        mock_st.session_state = {"_chats_restored": True}

        _restore_saved_chats(state)

        mock_get_mgr.assert_not_called()

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.get_chat_history_manager")
    def test_runs_once_even_with_no_saved_chats(self, mock_get_mgr: MagicMock, mock_st: MagicMock) -> None:
        """A user with zero saved chats must still only pay the storage
        scan once per session, not on every rerun."""
        mock_get_mgr.return_value.list_chat_histories.return_value = []
        mock_st.session_state = {}
        state = _new_state()

        _restore_saved_chats(state)
        _restore_saved_chats(state)

        mock_get_mgr.assert_called_once()
        assert mock_st.session_state["_chats_restored"] is True

    @patch("codebase_rag.app.main.st")
    @patch("codebase_rag.app.main.get_chat_history_manager")
    def test_loads_chats_and_selects_most_recent(self, mock_get_mgr: MagicMock, mock_st: MagicMock) -> None:
        mock_mgr = MagicMock()
        mock_mgr.list_chat_histories.return_value = [{"chat_id": "c1"}, {"chat_id": "c2"}]
        mock_mgr.get_chat_history.side_effect = [
            [{"role": "user", "content": "recent"}],
            [{"role": "user", "content": "older"}],
        ]
        mock_get_mgr.return_value = mock_mgr

        state = _new_state()
        mock_st.session_state = state._store
        _restore_saved_chats(state)

        assert state.current_chat_id == "c1"
        assert state.messages == [{"role": "user", "content": "recent"}]
        assert "c2" in state.chat_histories

    @patch("codebase_rag.app.main.get_chat_history_manager")
    def test_storage_error_does_not_raise(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.side_effect = OSError("db error")
        state = _new_state()

        _restore_saved_chats(state)  # should not raise

        assert state.chat_histories == {}
