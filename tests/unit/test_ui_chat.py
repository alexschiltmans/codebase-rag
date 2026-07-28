"""Unit tests for app/ui_chat.py, with Streamlit mocked out."""

from unittest.mock import MagicMock, patch

from codebase_rag.app.state import QueryLifecycle, SessionState
from codebase_rag.app.ui_chat import (
    append_message,
    display_chat_history,
    display_failed_query,
    display_sources,
    process_pending_query,
)


def _new_state() -> SessionState:
    state = SessionState(_store={})
    state.ensure_defaults()
    return state


class TestDisplaySources:
    @patch("codebase_rag.app.ui_chat.st")
    def test_groups_by_file_path(self, mock_st: MagicMock) -> None:
        sources = [
            {"file_path": "a/b.py", "file_name": "b.py"},
            {"file_path": "a/b.py", "file_name": "b.py"},
        ]
        display_sources(sources)
        markdown_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        entries = [c for c in markdown_calls if "b.py" in c]
        assert len(entries) == 1

    @patch("codebase_rag.app.ui_chat.st")
    def test_empty_list_renders_nothing(self, mock_st: MagicMock) -> None:
        display_sources([])
        mock_st.markdown.assert_not_called()


class TestDisplayChatHistory:
    @patch("codebase_rag.app.ui_chat.st")
    def test_renders_every_message(self, mock_st: MagicMock) -> None:
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        state = _new_state()
        state.append_message("user", "hi")
        state.append_message("assistant", "hello")

        display_chat_history(state)

        assert mock_st.chat_message.call_count == 2

    @patch("codebase_rag.app.ui_chat.st")
    def test_flagged_message_renders_the_truncation_warning(self, mock_st: MagicMock) -> None:
        """This re-render is what the user sees; st.rerun() discards anything drawn during process_pending_query."""
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        state = _new_state()
        state.append_message("assistant", "an answer", question_truncated=True)

        display_chat_history(state)

        mock_st.warning.assert_called_once()

    @patch("codebase_rag.app.ui_chat.st")
    def test_unflagged_message_renders_no_warning(self, mock_st: MagicMock) -> None:
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        state = _new_state()
        state.append_message("assistant", "an answer")

        display_chat_history(state)

        mock_st.warning.assert_not_called()


class TestAppendMessage:
    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    def test_persists_to_storage(self, mock_get_mgr: MagicMock) -> None:
        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr
        state = _new_state()

        append_message(state, "user", "hello")

        assert state.messages == [{"role": "user", "content": "hello"}]
        mock_mgr.save_chat_history.assert_called_once()

    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    def test_empty_content_replaced(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        state = _new_state()

        append_message(state, "assistant", "")

        assert "apologize" in state.messages[0]["content"]

    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    def test_storage_error_does_not_raise(self, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.side_effect = OSError("disk full")
        state = _new_state()

        append_message(state, "user", "hello")  # should not raise

        assert len(state.messages) == 1


class TestProcessPendingQuery:
    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    @patch("codebase_rag.app.ui_chat.st")
    def test_successful_stream_moves_to_idle(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.write_stream.return_value = "42"

        runtime = MagicMock()
        mock_chain = MagicMock()
        mock_chain.last_result = {"sources": []}
        runtime.new_rag_chain.return_value = mock_chain

        state = _new_state()
        state.append_message("user", "what is the answer?")
        state.submit_query("what is the answer?")

        process_pending_query(runtime, state)

        assert state.query_state == QueryLifecycle.IDLE
        assert any("42" in m["content"] for m in state.messages)
        mock_st.warning.assert_not_called()

    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    @patch("codebase_rag.app.ui_chat.st")
    def test_truncated_question_renders_warning(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.write_stream.return_value = "42"

        runtime = MagicMock()
        mock_chain = MagicMock()
        mock_chain.last_result = {"sources": [], "metrics": {"question_truncated_chars": 37}}
        runtime.new_rag_chain.return_value = mock_chain

        state = _new_state()
        state.append_message("user", "what is the answer?")
        state.submit_query("what is the answer?")

        process_pending_query(runtime, state)

        assert state.query_state == QueryLifecycle.IDLE
        # Persisted on the message, not rendered immediately, so it survives the st.rerun() that follows.
        assert state.messages[-1]["question_truncated"] is True

    @patch("codebase_rag.app.ui_chat.get_chat_history_manager")
    @patch("codebase_rag.app.ui_chat.st")
    def test_exception_moves_to_failed(self, mock_st: MagicMock, mock_get_mgr: MagicMock) -> None:
        mock_get_mgr.return_value = MagicMock()
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        # A real chat_message context manager doesn't suppress exceptions
        # raised inside it; MagicMock's default __exit__ is truthy and
        # would, so pin it.
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.write_stream.side_effect = RuntimeError("boom")

        runtime = MagicMock()
        runtime.new_rag_chain.return_value = MagicMock()

        state = _new_state()
        state.append_message("user", "q")
        state.submit_query("q")

        process_pending_query(runtime, state)

        assert state.query_state == QueryLifecycle.FAILED
        assert state.query_error == "boom"
        # A stuck query never re-triggers the LLM on the next call.
        assert state.pending_query == "q"


class TestDisplayFailedQuery:
    @patch("codebase_rag.app.ui_chat.st")
    def test_retry_resubmits(self, mock_st: MagicMock) -> None:
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.columns.return_value[0].button.return_value = True
        mock_st.columns.return_value[1].button.return_value = False

        state = _new_state()
        state.submit_query("q")
        state.query_failed("boom")

        display_failed_query(state)

        assert state.query_state == QueryLifecycle.PENDING
        mock_st.rerun.assert_called_once()

    @patch("codebase_rag.app.ui_chat.st")
    def test_dismiss_clears_state(self, mock_st: MagicMock) -> None:
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.columns.return_value[0].button.return_value = False
        mock_st.columns.return_value[1].button.return_value = True

        state = _new_state()
        state.submit_query("q")
        state.query_failed("boom")

        display_failed_query(state)

        assert state.query_state == QueryLifecycle.IDLE
        assert state.pending_query is None
