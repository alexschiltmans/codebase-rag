"""Unit tests for the QueryLifecycle state machine in app/state.py."""

from codebase_rag.app.state import QueryLifecycle, SessionState


def _new_state() -> SessionState:
    """A SessionState backed by a plain dict instead of st.session_state,
    so these tests run with no Streamlit runtime."""
    state = SessionState(_store={})
    state.ensure_defaults()
    return state


class TestQueryLifecycle:
    def test_starts_idle(self) -> None:
        state = _new_state()
        assert state.query_state == QueryLifecycle.IDLE
        assert state.pending_query is None

    def test_submit_moves_to_pending(self) -> None:
        state = _new_state()
        state.submit_query("how does this work?")
        assert state.query_state == QueryLifecycle.PENDING
        assert state.pending_query == "how does this work?"

    def test_success_returns_to_idle_and_clears_query(self) -> None:
        state = _new_state()
        state.submit_query("q")
        state.query_succeeded()
        assert state.query_state == QueryLifecycle.IDLE
        assert state.pending_query is None
        assert state.query_error is None

    def test_failure_moves_to_failed_and_keeps_query(self) -> None:
        state = _new_state()
        state.submit_query("q")
        state.query_failed("boom")
        assert state.query_state == QueryLifecycle.FAILED
        assert state.query_error == "boom"
        assert state.pending_query == "q"

    def test_dismiss_failure_returns_to_idle_without_resubmitting(self) -> None:
        state = _new_state()
        state.submit_query("q")
        state.query_failed("boom")
        state.dismiss_failure()
        assert state.query_state == QueryLifecycle.IDLE
        assert state.pending_query is None
        assert state.query_error is None

    def test_retry_resubmits_the_same_query_once(self) -> None:
        state = _new_state()
        state.submit_query("q")
        state.query_failed("boom")
        state.retry_failed_query()
        assert state.query_state == QueryLifecycle.PENDING
        assert state.pending_query == "q"
        assert state.query_error is None


class TestChatLifecycle:
    def test_no_phantom_chat_before_first_message(self) -> None:
        state = _new_state()
        assert state.current_chat_id is None
        assert state.chat_histories == {}

    def test_ensure_current_chat_mints_lazily(self) -> None:
        state = _new_state()
        chat_id = state.ensure_current_chat()
        assert chat_id is not None
        assert chat_id in state.chat_histories
        assert state.ensure_current_chat() == chat_id

    def test_append_message_creates_chat_on_first_message(self) -> None:
        state = _new_state()
        assert state.current_chat_id is None
        state.append_message("user", "hello")
        assert state.current_chat_id is not None
        assert state.messages == [{"role": "user", "content": "hello"}]
        assert state.chat_history_for(state.current_chat_id) == [{"role": "user", "content": "hello"}]

    def test_start_new_chat_resets_messages(self) -> None:
        state = _new_state()
        state.append_message("user", "hello")
        old_chat_id = state.current_chat_id
        state.start_new_chat()
        assert state.current_chat_id != old_chat_id
        assert state.messages == []

    def test_switch_chat_restores_messages(self) -> None:
        state = _new_state()
        state.append_message("user", "hello")
        first_chat_id = state.current_chat_id
        state.start_new_chat()
        state.switch_chat(first_chat_id)
        assert state.current_chat_id == first_chat_id
        assert state.messages == [{"role": "user", "content": "hello"}]


class TestGetSession:
    def test_ensure_defaults_is_idempotent(self) -> None:
        """get_session() should not clobber existing session_state values
        on a rerun."""
        store = {"query_state": QueryLifecycle.PENDING, "pending_query": "in flight"}
        state = SessionState(_store=store)
        state.ensure_defaults()
        assert state.query_state == QueryLifecycle.PENDING
        assert state.pending_query == "in flight"
