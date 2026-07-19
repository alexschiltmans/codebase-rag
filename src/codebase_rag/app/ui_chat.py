"""Chat surface: history, input, streaming, and the query error state."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from codebase_rag.app.runtime import MAX_CONVERSATION_HISTORY, AppRuntime, list_chat_metadata
from codebase_rag.app.state import QueryLifecycle, SessionState
from codebase_rag.database.chat_storage import get_chat_history_manager

logger = logging.getLogger(__name__)


def display_sources(sources: list[dict[str, str]]) -> None:
    """Display the sources used for a response, grouped by file path."""
    if not sources:
        return

    st.markdown("### Sources")
    grouped: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        grouped.setdefault(source.get("file_path", "Unknown"), []).append(source)

    for file_path, source_list in grouped.items():
        file_name = source_list[0].get("file_name", "Unknown")
        st.markdown(f"- `{file_path}` — {file_name}")


def _format_message(message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "")):
        st.markdown(message.get("content", ""))
        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources"):
                display_sources(sources)


def display_chat_history(state: SessionState) -> None:
    for message in state.messages:
        _format_message(message)


def append_message(state: SessionState, role: str, content: str, sources: list[dict[str, str]] | None = None) -> None:
    """Append a message to the current chat and persist it to storage."""
    if not content or not content.strip():
        content = "I apologize, but I wasn't able to generate a response. Please try rephrasing your question."

    state.append_message(role, content, sources)

    chat_id = state.current_chat_id
    if chat_id is None:
        return
    try:
        chat_manager = get_chat_history_manager()
        chat_manager.save_chat_history(chat_id, state.chat_history_for(chat_id))
        list_chat_metadata.clear()  # type: ignore[attr-defined]
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to save chat history: %s", e)


def _apply_conversation_history(rag_chain: Any, state: SessionState) -> None:
    """Replay this session's prior turns onto a fresh per-session RAGChain.

    Excludes the last message (the current user query), since
    ``rag_chain.stream()`` adds it itself via ``add_user_message``.
    """
    rag_chain.conversation_history = []
    tail = state.messages[-(2 * MAX_CONVERSATION_HISTORY + 1) : -1]
    for msg in tail:
        if msg["role"] == "user":
            rag_chain.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            rag_chain.add_assistant_message(msg["content"], msg.get("sources"))


def process_pending_query(runtime: AppRuntime, state: SessionState) -> None:
    """Run the query lifecycle's PENDING -> IDLE/FAILED transition.

    The catch here is deliberately broad: this is the state-machine
    boundary, and any exception crossing it must become a FAILED card
    rather than a Streamlit traceback page.
    """
    query = state.pending_query
    if query is None:
        state.query_succeeded()
        return

    try:
        rag_chain = runtime.new_rag_chain()
        _apply_conversation_history(rag_chain, state)
        with st.chat_message("assistant"):
            answer = st.write_stream(rag_chain.stream(query))
        if not isinstance(answer, str):
            answer = "".join(str(part) for part in answer)
        sources = (rag_chain.last_result or {}).get("sources", [])
        append_message(state, "assistant", answer, sources)
        state.query_succeeded()
    except Exception as e:  # noqa: BLE001 - state-machine boundary, see docstring
        logger.error("Error generating response: %s", e)
        state.query_failed(str(e))


def display_failed_query(state: SessionState) -> None:
    """Render the FAILED state as a dismissible assistant-slot error card
    with a Retry button that resubmits the same query once."""
    with st.chat_message("assistant"):
        st.error(f"I ran into an error answering that: {state.query_error}")
        cols = st.columns(2)
        if cols[0].button("Retry", key="btn_retry_query"):
            state.retry_failed_query()
            st.rerun()
        if cols[1].button("Dismiss", key="btn_dismiss_query_error"):
            state.dismiss_failure()
            st.rerun()


def display_chat_interface(runtime: AppRuntime, state: SessionState, *, chat_gated: bool) -> None:
    """Render history, the pending/failed query states, and the input box."""
    if chat_gated:
        st.chat_input("Getting ready — please wait…", disabled=True)
        return

    display_chat_history(state)

    if state.query_state == QueryLifecycle.FAILED:
        display_failed_query(state)
    elif state.query_state == QueryLifecycle.PENDING:
        process_pending_query(runtime, state)
        st.rerun()

    if prompt := st.chat_input("Ask about your codebase"):
        append_message(state, "user", prompt)
        state.submit_query(prompt)
        st.rerun()
