"""Entry point: page config and page assembly.

All process-wide resources come from ``AppRuntime`` (``runtime.py``); all
per-session state comes from ``SessionState`` (``state.py``). This module
wires them into the page and nothing else.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from codebase_rag.app.runtime import AppRuntime, get_runtime
from codebase_rag.app.state import SessionState, get_session
from codebase_rag.app.ui_chat import display_chat_interface
from codebase_rag.app.ui_sidebar import display_sidebar
from codebase_rag.database.chat_storage import get_chat_history_manager

_LOGO_PATH = str(Path(__file__).parent / "logo.png")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="Codebase RAG", page_icon=_LOGO_PATH, layout="wide")


def _display_header() -> None:
    st.title("Codebase RAG")
    st.markdown(
        "Ask questions about your ingested codebases. "
        "The assistant will provide answers based on the documentation and code."
    )
    st.markdown("---")


def _restore_saved_chats(state: SessionState) -> None:
    """Load any previously persisted chats into this session, once."""
    if state.chat_histories:
        return
    try:
        chat_manager = get_chat_history_manager()
        chat_list = chat_manager.list_chat_histories()
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to load saved chat histories: %s", e)
        return

    for chat_metadata in chat_list:
        chat_id = chat_metadata.get("chat_id")
        if not chat_id:
            continue
        messages = chat_manager.get_chat_history(chat_id)
        if messages:
            state.chat_histories[chat_id] = list(messages)

    if chat_list:
        most_recent_id = chat_list[0].get("chat_id")
        if most_recent_id and most_recent_id in state.chat_histories:
            st.session_state["current_chat_id"] = most_recent_id
            st.session_state["messages"] = list(state.chat_histories[most_recent_id])
        logger.info("Loaded %d saved chats from storage", len(chat_list))


def _display_auto_ingest_gate(runtime: AppRuntime) -> bool:
    """Render the first-boot banner while the default repo auto-ingests.

    Returns True while the chat surface should stay gated. A running
    *manual* job never gates the chat; only a running auto job does.
    """
    job = runtime.ingestion.current_job()
    if job is None or job.kind != "auto":
        error = runtime.ingestion.auto_job_error()
        if error:
            st.warning(
                f"Default repository ingestion failed: {error}\n\nYou can add a repository manually using the sidebar."
            )
        return False

    repo_name = job.source.rstrip("/").rsplit("/", 1)[-1]
    st.info(
        f"🚀 **Getting ready…**\n\nPreparing **{repo_name}** so you can start exploring right away. "
        "This usually takes a few minutes on first startup.",
        icon="🔄",
    )
    return True


def main() -> None:
    runtime = get_runtime()
    state = get_session()
    _restore_saved_chats(state)

    _display_header()
    display_sidebar(runtime, state)

    chat_gated = _display_auto_ingest_gate(runtime)
    display_chat_interface(runtime, state, chat_gated=chat_gated)


if __name__ == "__main__":
    main()
