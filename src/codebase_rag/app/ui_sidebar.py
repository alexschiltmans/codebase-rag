"""Sidebar: logo, about, repository manager, and chat list."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st

from codebase_rag.app.runtime import AppRuntime, get_repo_list, list_chat_metadata
from codebase_rag.app.state import SessionState
from codebase_rag.database.chat_storage import get_chat_history_manager

logger = logging.getLogger(__name__)

_LOGO_PATH = Path(__file__).parent / "logo.png"
_INGEST_REFUSED_MESSAGE = "An ingestion is already running, wait for it to finish."
# Stated as fact, not as a remedy: CPU inference is the only option on plenty of machines.
_PLACEMENT_TEXT = {"gpu": ", on the GPU", "cpu": ", on the CPU"}


def _backend_line(runtime: AppRuntime) -> str:
    """The About block's backend bullet, naming the endpoint once health has reported.

    Renders without the endpoint before the background check lands, so the first paint is
    the existing line rather than an error. Placement appears only once it is known, which
    on a cold start means after the first answer has loaded the model.
    """
    backend_label = "Ollama" if runtime.config.provider == "ollama" else "an OpenAI-compatible server"
    line = f"- A local LLM via {backend_label} (**{runtime.config.llm_model_name}**)"

    endpoint = runtime.health.get("connection", {}).get("url")
    if not endpoint:
        return line
    placement = runtime.health.get("placement", {}).get("placement")
    return f"{line} at `{endpoint}`{_PLACEMENT_TEXT.get(placement, '')}"


def _retrieval_lines(runtime: AppRuntime) -> list[str]:
    """The About block's retrieval bullets, read from config: the fixed list claimed hybrid, the default is BM25."""
    config = runtime.config
    if config.retriever == "hybrid":
        lines = [
            "- Hybrid search, fusing vector similarity with BM25",
            "- Qdrant vector database, holding the embedded chunks",
        ]
    else:
        lines = [
            "- Keyword search over the indexed chunks (BM25)",
            "- Qdrant vector database, where ingestion stores the embedded chunks",
        ]
    if config.rerank_enabled:
        lines.append(f"- Cross-encoder reranking of the top candidates (**{config.rerank_model}**)")
    if config.rewrite_enabled:
        lines.append("- Query rewriting, expanding the question with likely identifiers before retrieval")
    return lines


def _about_text(runtime: AppRuntime) -> str:
    """The whole About body, assembled here so no line depends on a literal's indentation."""
    return "\n".join(
        [
            (
                "Codebase RAG is a Retrieval-Augmented Generation application for exploring and "
                "understanding codebases locally."
            ),
            "",
            "It helps users understand code by providing answers based on ingested documentation and source code.",
            "",
            "This application uses:",
            _backend_line(runtime),
            *_retrieval_lines(runtime),
        ]
    )


def display_sidebar(runtime: AppRuntime, state: SessionState) -> None:
    """Render the full sidebar: logo, about, repos, and chat history."""
    try:
        st.logo(str(_LOGO_PATH), size="large")
    except Exception as e:  # FileNotFoundError, RuntimeError, etc.
        logger.debug("Skipping sidebar logo due to %s", e)

    st.sidebar.title("About")
    st.sidebar.markdown(_about_text(runtime))

    if runtime.health:
        _display_model_health(runtime)
    else:
        # Health runs on a background thread, and nothing else triggers a rerun when it lands,
        # so on a cold start with no auto-ingest the first paint would show no banner at all
        # until some unrelated interaction. Torn down the moment health is populated, so this
        # is not the standing poll the design ruled out.
        _model_health_fragment(runtime)

    with st.sidebar:
        _display_repo_management(runtime)

    _display_new_chat_button(state)
    _display_chat_history_list(state)


@st.fragment(run_every=2)
def _model_health_fragment(runtime: AppRuntime) -> None:
    """Wait for the health thread's first result, then hand off to a plain render."""
    if runtime.health:
        st.rerun(scope="app")
        return


def _display_model_health(runtime: AppRuntime) -> None:
    """Warn about the configured model, once the health check has reported."""
    model_status = runtime.health.get("model", {})
    model_health_status = model_status.get("status")
    if model_health_status == "not_found":
        st.sidebar.warning(
            f"""
            Model **{runtime.config.llm_model_name}** not found.

            {model_status.get("suggested_action", "Please pull the model.")}

            The check refreshes on app restart.
            """
        )
    elif model_health_status == "error":
        # The most common failure (backend unreachable) previously showed nothing at all:
        # check_model_availability returns {"status": "error", ...} on any failure to check,
        # not "not_found", so this case fell through the check above silently. This covers
        # both an actually-unreachable server and one that reached but rejected the request
        # (e.g. a 401 on a misconfigured LLM_API_KEY), so the heading can't claim "can't
        # reach" specifically; the check's own message carries whichever it was.
        is_ollama = runtime.config.provider == "ollama"
        backend_url = runtime.config.ollama_base_url if is_ollama else runtime.config.llm_base_url
        st.sidebar.warning(
            f"""
            Problem with the configured LLM backend at **{backend_url}**.

            {model_status.get("message", "Connection failed.")}

            The check refreshes on app restart.
            """
        )


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
        if job.progress_total > 0:
            st.progress(min(job.progress_current / job.progress_total, 1.0))
            st.caption(f"{job.phase} - {job.progress_current}/{job.progress_total}")
        st.write(f"⏳ {elapsed}s elapsed")
        status.update(label=f"Ingesting {job.source}… ({elapsed}s)")
        if st.button("Cancel", key="btn_cancel_ingestion"):
            runtime.ingestion.cancel()

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
        elif job.state == "cancelled":
            st.session_state["ingestion_cancelled_banner"] = {"source": job.source}
            runtime.ingestion.acknowledge()
        else:
            st.session_state["ingestion_error_banner"] = {"source": job.source, "error": job.error}
            runtime.ingestion.acknowledge()

    banner = st.session_state.get("ingestion_error_banner")
    if banner:
        _display_dismissible_error(
            f"Ingestion of **{banner['source']}** failed: {banner['error']}",
            "ingestion_error_banner",
            "btn_dismiss_ingestion_error",
        )

    cancelled_banner = st.session_state.get("ingestion_cancelled_banner")
    if cancelled_banner:
        _display_dismissible_info(
            f"Ingestion of **{cancelled_banner['source']}** was cancelled - the repo is partially ingested. "
            "Re-run ingest to complete it, or remove the repo.",
            "ingestion_cancelled_banner",
            "btn_dismiss_ingestion_cancelled",
        )


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

    # Read from the flag every run, not just the click run, so a finishing ingest can't close an open dialog.
    pending = st.session_state.get("confirm_delete_repo")
    if pending in repos:
        _confirm_delete_repo_dialog(runtime, pending)


def _clear_confirm_delete_repo() -> None:
    st.session_state.pop("confirm_delete_repo", None)


@st.dialog("Remove repository", on_dismiss=_clear_confirm_delete_repo)
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


def _display_dismissible_error(message: str, session_key: str, dismiss_key: str) -> None:
    st.error(message)
    if st.button("Dismiss", key=dismiss_key):
        st.session_state.pop(session_key, None)
        st.rerun()


def _display_dismissible_info(message: str, session_key: str, dismiss_key: str) -> None:
    st.info(message)
    if st.button("Dismiss", key=dismiss_key):
        st.session_state.pop(session_key, None)
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
            if runtime.ingestion.start(new_repo_url, kind="manual"):
                st.session_state.pop("github_ingest_refused_error", None)
                st.rerun()
            else:
                st.session_state["github_ingest_refused_error"] = _INGEST_REFUSED_MESSAGE
                st.rerun()
        else:
            st.session_state.pop("github_ingest_refused_error", None)
            st.session_state["github_url_error"] = "Please enter a valid GitHub URL"

    if st.session_state.get("github_url_error"):
        _display_dismissible_error(
            st.session_state["github_url_error"], "github_url_error", "btn_dismiss_github_url_error"
        )

    if st.session_state.get("github_ingest_refused_error"):
        _display_dismissible_error(
            st.session_state["github_ingest_refused_error"],
            "github_ingest_refused_error",
            "btn_dismiss_github_ingest_refused_error",
        )


def _display_local_folder_tab(runtime: AppRuntime, ingestion_running: bool) -> None:
    if "_pending_clear_typed_folder_path" in st.session_state:
        st.session_state["typed_folder_path"] = st.session_state.pop("_pending_clear_typed_folder_path")

    just_picked_path = _poll_folder_dialog(runtime)

    if st.button("Browse…", key="btn_browse_folder", disabled=ingestion_running or runtime.folder_picker.is_open()):
        st.session_state.pop("folder_dialog_error", None)
        st.session_state.pop("_folder_dialog_stale_poll", None)
        token = runtime.folder_picker.open()
        st.session_state["folder_dialog_token"] = token
        if token is None:
            st.session_state["folder_dialog_error"] = "A folder dialog is already open."

    # Gated on the token, not just is_open(): the picker thread can store its result and exit
    # between the poll above and this check, leaving the token set with is_open() already False.
    if st.session_state.get("folder_dialog_token") is not None or runtime.folder_picker.is_open():
        _folder_dialog_wait_fragment(runtime)

    if st.session_state.get("folder_dialog_error"):
        st.error(st.session_state["folder_dialog_error"])

    st.caption("Or, if no native dialog is available (e.g. inside Docker), type a path directly:")
    typed_path = st.text_input("Folder path", key="typed_folder_path", label_visibility="collapsed")

    # `_poll_folder_dialog` already wrote a fresh pick into `typed_path`'s widget state above.
    st.session_state.selected_folder = just_picked_path if just_picked_path is not None else typed_path

    if st.session_state.selected_folder:
        _display_selected_folder(runtime, ingestion_running)

    if st.session_state.get("local_ingest_refused_error"):
        _display_dismissible_error(
            st.session_state["local_ingest_refused_error"],
            "local_ingest_refused_error",
            "btn_dismiss_local_ingest_refused_error",
        )


@st.fragment(run_every=1)
def _folder_dialog_wait_fragment(runtime: AppRuntime) -> None:
    """Poll the open native dialog while nothing else is driving a rerun.

    Without this, a picked path only appears once the user happens to
    interact with some other widget, since a background thread can't
    trigger a Streamlit rerun on its own.
    """
    picked_path = _poll_folder_dialog(runtime, allow_self_terminate=True)
    # A pick can land while `is_open()` is still True: FolderPicker is process-wide, so
    # another session's dialog can keep it True for a session whose own pick already landed.
    # Rerun on that too, not just "no longer open": the widget holding the new path
    # otherwise never gets redrawn until some unrelated interaction triggers a rerun.
    if picked_path is None and runtime.folder_picker.is_open():
        st.caption("⏳ Folder dialog opened — waiting for your selection…")
    else:
        st.rerun(scope="app")


def _poll_folder_dialog(runtime: AppRuntime, *, allow_self_terminate: bool = False) -> str | None:
    """Poll the picker; return the path if one just landed this render.

    Writes the result into the typed-path widget's state directly. Safe even from the wait
    fragment's own auto-rerun, where that widget already exists from the parent run, because
    Streamlit resets widget_ids_this_run per fragment run.

    ``allow_self_terminate`` lets a token that got no result and finds the dialog no longer
    open (stranded, e.g. cancelled) be cleared, but only on the *second* such miss: a result
    can still land between one poll and the next (the wait fragment's own poll can be the
    first real attempt on a token the main body's poll ran too early to see), so a single miss
    isn't enough signal: it's tracked with ``_folder_dialog_stale_poll`` across fragment ticks.
    """
    token = st.session_state.get("folder_dialog_token")
    if token is None:
        return None
    result = runtime.folder_picker.poll(token)
    if result is None:
        if allow_self_terminate and not runtime.folder_picker.is_open():
            if st.session_state.get("_folder_dialog_stale_poll"):
                st.session_state["folder_dialog_token"] = None
                st.session_state.pop("_folder_dialog_stale_poll", None)
                st.session_state["folder_dialog_error"] = "The folder dialog closed without a selection."
            else:
                st.session_state["_folder_dialog_stale_poll"] = True
        return None
    st.session_state["folder_dialog_token"] = None
    st.session_state.pop("_folder_dialog_stale_poll", None)
    if result.path:
        st.session_state.pop("folder_dialog_error", None)
        st.session_state["typed_folder_path"] = result.path
        return result.path
    if result.error:
        st.session_state["folder_dialog_error"] = result.error
    return None


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
        if runtime.ingestion.start(str(folder_path), kind="manual"):
            st.session_state.pop("local_ingest_refused_error", None)
            # Queued instead of set directly: the widget's already instantiated this
            # render, so touching its key now would raise; the next render's top-of-
            # function pending-path check applies it before the widget is recreated.
            st.session_state["_pending_clear_typed_folder_path"] = ""
            st.rerun()
        else:
            st.session_state["local_ingest_refused_error"] = _INGEST_REFUSED_MESSAGE
            st.rerun()


def _preview_local_folder(folder_path: Path) -> tuple[list[str], int]:
    """Return the discovered dirs and file count for a folder, re-walked on every render.

    Deliberately uncached: added/removed files must show up on the very next rerun, which a TTL cache would delay.
    """
    from codebase_rag.data_ingestion.pipeline import count_ingestible_files

    return count_ingestible_files(folder_path)


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
    metadata = list_chat_metadata()
    order = [m["chat_id"] for m in metadata if m.get("chat_id") in state.chat_histories]

    ordered_ids = order + [cid for cid in state.chat_histories if cid not in order]
    return [(cid, state.chat_histories[cid]) for cid in ordered_ids]


def _clear_confirm_delete_chat() -> None:
    st.session_state.pop("confirm_delete_chat", None)


@st.dialog("Delete chat", on_dismiss=_clear_confirm_delete_chat)
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
        list_chat_metadata.clear()
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to delete chat from persistent storage: %s", e)
