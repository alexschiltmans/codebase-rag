"""Sidebar: logo, about, repository manager, and chat list."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st

from codebase_rag.app.runtime import AppRuntime, get_repo_list
from codebase_rag.app.state import SessionState
from codebase_rag.database.chat_storage import get_chat_history_manager

logger = logging.getLogger(__name__)

_LOGO_PATH = Path(__file__).parent / "logo.png"


def display_sidebar(runtime: AppRuntime, state: SessionState) -> None:
    """Render the full sidebar: logo, about, repos, and chat history."""
    try:
        st.logo(str(_LOGO_PATH), size="large")
    except Exception as e:  # FileNotFoundError, RuntimeError, etc.
        logger.debug("Skipping sidebar logo due to %s", e)

    st.sidebar.title("About")
    st.sidebar.markdown(
        f"""
        Codebase RAG is a Retrieval-Augmented Generation application for exploring and understanding codebases locally.

        It helps users understand code by providing answers based on ingested documentation and source code.

        This application uses:
        - A local LLM via Ollama (**{runtime.config.llm_model_name}**)
        - Hybrid search combining vector and BM25
        - Qdrant vector database
        """
    )

    with st.sidebar:
        _display_repo_management(runtime)

    _display_new_chat_button(state)
    _display_chat_history_list(state)


def _display_repo_management(runtime: AppRuntime) -> None:
    """Repo status, list, and the add-repository controls.

    Only wrapped in a live-updating fragment while a job is actually
    running; the idle sidebar renders no ``run_every`` fragment at all.
    """
    st.subheader("Repositories")

    _display_ingestion_outcome(runtime)

    job = runtime.ingestion.current_job()
    if job is not None:
        _ingestion_progress_fragment(runtime)
    else:
        _display_repo_list(runtime)
        _display_add_repository(runtime, ingestion_running=False)


@st.fragment(run_every=2)
def _ingestion_progress_fragment(runtime: AppRuntime) -> None:
    """Live progress while an ingestion is running. Torn down (not just
    idle) the instant the job finishes, per the "fragments are for live
    regions only, and only while live" rule.
    """
    job = runtime.ingestion.current_job()
    if job is None:
        st.rerun(scope="app")
        return

    with st.status(f"Ingesting {job.source}…", expanded=True) as status:
        elapsed = int(time.time() - job.started_at)
        st.write(f"⏳ {elapsed}s elapsed")
        status.update(label=f"Ingesting {job.source}… ({elapsed}s)")

    _display_repo_list(runtime)
    _display_add_repository(runtime, ingestion_running=True)


def _display_ingestion_outcome(runtime: AppRuntime) -> None:
    """Show the most recent finished job as a toast (success) or a
    dismissible banner (failure), per the notification policy in §6.5.
    """
    job = runtime.ingestion.last_completed()
    if job is not None:
        if job.state == "succeeded":
            st.toast(f"✅ Ingested **{job.source}** successfully!")
            runtime.ingestion.acknowledge()
        else:
            st.session_state["ingestion_error_banner"] = {"source": job.source, "error": job.error}
            runtime.ingestion.acknowledge()

    banner = st.session_state.get("ingestion_error_banner")
    if banner:
        st.error(f"Ingestion of **{banner['source']}** failed: {banner['error']}")
        if st.button("Dismiss", key="btn_dismiss_ingestion_error"):
            del st.session_state["ingestion_error_banner"]
            st.rerun()


def _display_repo_list(runtime: AppRuntime) -> None:
    repos = get_repo_list(runtime.qdrant_store)
    if not repos:
        st.info("No repositories ingested yet.")
        return

    for repo_name in repos:
        cols = st.columns([6, 1])
        cols[0].markdown(f"📦 **{repo_name}**")
        if cols[1].button(
            "", icon=":material/delete:", type="tertiary", key=f"del_repo_{repo_name}", help=f"Remove {repo_name}"
        ):
            st.session_state["confirm_delete_repo"] = repo_name
            st.rerun()

    pending = st.session_state.get("confirm_delete_repo")
    if pending in repos:
        _confirm_delete_repo_dialog(runtime, pending)


@st.dialog("Remove repository")
def _confirm_delete_repo_dialog(runtime: AppRuntime, repo_name: str) -> None:
    st.write(f"Remove **{repo_name}** and all of its indexed chunks? This can't be undone.")
    cols = st.columns(2)
    if cols[0].button("Remove", type="primary", key="btn_confirm_delete_repo"):
        with st.spinner(f"Removing {repo_name}..."):
            deleted = runtime.delete_repo(repo_name)
        st.session_state.pop("confirm_delete_repo", None)
        st.toast(f"Removed {repo_name} ({deleted} chunks)")
        st.rerun()
    if cols[1].button("Cancel", key="btn_cancel_delete_repo"):
        st.session_state.pop("confirm_delete_repo", None)
        st.rerun()


def _display_add_repository(runtime: AppRuntime, *, ingestion_running: bool) -> None:
    with st.expander("Add Repository"):
        tab_github, tab_local = st.tabs(["GitHub URL", "Local Folder"])
        with tab_github:
            _display_github_tab(runtime, ingestion_running)
        with tab_local:
            _display_local_folder_tab(runtime, ingestion_running)


def _display_github_tab(runtime: AppRuntime, ingestion_running: bool) -> None:
    new_repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo", key="new_repo_url")
    if st.button("Ingest", key="btn_ingest_repo", disabled=bool(not new_repo_url or ingestion_running)):
        if new_repo_url and new_repo_url.startswith("https://github.com/"):
            st.session_state.pop("github_url_error", None)
            runtime.ingestion.start(new_repo_url, kind="manual")
            st.rerun()
        else:
            st.session_state["github_url_error"] = "Please enter a valid GitHub URL"

    if st.session_state.get("github_url_error"):
        st.error(st.session_state["github_url_error"])
        if st.button("Dismiss", key="btn_dismiss_github_url_error"):
            del st.session_state["github_url_error"]
            st.rerun()


def _display_local_folder_tab(runtime: AppRuntime, ingestion_running: bool) -> None:
    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = ""

    _poll_folder_dialog(runtime)

    dialog_open = runtime.folder_picker.is_open()
    if st.button("Browse…", key="btn_browse_folder", disabled=ingestion_running or dialog_open):
        st.session_state.pop("folder_dialog_error", None)
        token = runtime.folder_picker.open()
        st.session_state["folder_dialog_token"] = token
        if token is None:
            st.session_state["folder_dialog_error"] = "A folder dialog is already open."

    if dialog_open:
        st.caption("⏳ Folder dialog opened — waiting for your selection…")

    if st.session_state.get("folder_dialog_error"):
        st.error(st.session_state["folder_dialog_error"])

    st.caption("Or, if no native dialog is available (e.g. inside Docker), type a path directly:")
    typed_path = st.text_input("Folder path", key="typed_folder_path", label_visibility="collapsed")
    if typed_path:
        st.session_state.selected_folder = typed_path

    if st.session_state.selected_folder:
        _display_selected_folder(runtime, ingestion_running)


def _poll_folder_dialog(runtime: AppRuntime) -> None:
    token = st.session_state.get("folder_dialog_token")
    if token is None:
        return
    result = runtime.folder_picker.poll(token)
    if result is None:
        return
    st.session_state["folder_dialog_token"] = None
    if result.path:
        st.session_state.selected_folder = result.path
        st.session_state.pop("folder_dialog_error", None)
    elif result.error:
        st.session_state["folder_dialog_error"] = result.error


def _display_selected_folder(runtime: AppRuntime, ingestion_running: bool) -> None:
    st.markdown(f"📂 `{st.session_state.selected_folder}`")
    folder_path = Path(st.session_state.selected_folder).resolve()
    if not folder_path.is_dir():
        st.error("Directory does not exist")
        return

    included_dirs, file_count = _preview_local_folder(folder_path)
    if file_count == 0:
        st.warning(
            "No ingestible files found in this folder "
            "(after skipping node_modules, venv, dist, and similar directories)."
        )
    else:
        dirs_label = ", ".join(included_dirs) if included_dirs else "(root)"
        st.caption(f"📄 {file_count} file(s) found in: {dirs_label}")

    if st.button("Ingest", key="btn_ingest_local", disabled=ingestion_running or file_count == 0):
        st.session_state.selected_folder = ""
        st.session_state.pop("_folder_preview_cache", None)
        runtime.ingestion.start(str(folder_path), kind="manual")
        st.rerun()


def _preview_local_folder(folder_path: Path) -> tuple[list[str], int]:
    """Return (and cache) the discovered dirs and file count for a folder,
    so this isn't re-walked on every rerun while the expander is open."""
    from codebase_rag.data_ingestion.pipeline import count_ingestible_files

    cache_key = str(folder_path)
    cached = st.session_state.get("_folder_preview_cache")
    if cached and cached.get("path") == cache_key:
        return cached["dirs"], cached["count"]

    included_dirs, file_count = count_ingestible_files(folder_path)
    st.session_state["_folder_preview_cache"] = {"path": cache_key, "dirs": included_dirs, "count": file_count}
    return included_dirs, file_count


def _display_new_chat_button(state: SessionState) -> None:
    if st.sidebar.button("Start New Chat", use_container_width=True):
        state.start_new_chat()
        st.rerun()


def _get_chat_title(chat_history: list[dict[str, Any]]) -> str:
    if not chat_history:
        return "New Chat"
    user_messages = [msg for msg in chat_history if msg.get("role") == "user"]
    if not user_messages:
        return "Empty Chat"
    content = str(user_messages[0].get("content", ""))
    return content[:20] + "..." if len(content) > 20 else content


def _display_chat_history_list(state: SessionState) -> None:
    if not state.chat_histories:
        return

    st.sidebar.subheader("Chat History")

    for chat_id, chat_history in _ordered_chats(state):
        chat_title = _get_chat_title(chat_history)
        if state.current_chat_id == chat_id:
            chat_title = f"➤ {chat_title}"

        cols = st.sidebar.columns([6, 1])
        if cols[0].button(chat_title, key=f"btn_{chat_id}"):
            state.switch_chat(chat_id)
            st.rerun()
        if cols[1].button("", icon=":material/delete:", type="tertiary", key=f"del_{chat_id}"):
            st.session_state["confirm_delete_chat"] = chat_id
            st.rerun()

    pending = st.session_state.get("confirm_delete_chat")
    if pending in state.chat_histories:
        _confirm_delete_chat_dialog(state, pending)


def _ordered_chats(state: SessionState) -> list[tuple[str, list[dict[str, Any]]]]:
    """Order sidebar chats by ``last_updated`` DESC, straight from storage
    metadata, instead of the old insertion-order-plus-reverse() scheme.
    """
    try:
        metadata = get_chat_history_manager().list_chat_histories()
        order = [m["chat_id"] for m in metadata if m.get("chat_id") in state.chat_histories]
    except (OSError, RuntimeError, ValueError):
        order = []

    ordered_ids = order + [cid for cid in state.chat_histories if cid not in order]
    return [(cid, state.chat_histories[cid]) for cid in ordered_ids]


@st.dialog("Delete chat")
def _confirm_delete_chat_dialog(state: SessionState, chat_id: str) -> None:
    st.write("Delete this chat? This can't be undone.")
    cols = st.columns(2)
    if cols[0].button("Delete", type="primary", key="btn_confirm_delete_chat"):
        _delete_chat(state, chat_id)
        st.session_state.pop("confirm_delete_chat", None)
        st.rerun()
    if cols[1].button("Cancel", key="btn_cancel_delete_chat"):
        st.session_state.pop("confirm_delete_chat", None)
        st.rerun()


def _delete_chat(state: SessionState, chat_id: str) -> None:
    del state.chat_histories[chat_id]

    if state.current_chat_id == chat_id:
        if state.chat_histories:
            state.switch_chat(next(iter(state.chat_histories)))
        else:
            state.start_new_chat()

    try:
        get_chat_history_manager().delete_chat_history(chat_id)
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to delete chat from persistent storage: %s", e)
