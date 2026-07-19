from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from codebase_rag.config import Config

if TYPE_CHECKING:
    from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.database.chat_storage import get_chat_history_manager

logger = logging.getLogger(__name__)

# Module-level dict for ingestion status — threads cannot write to
# st.session_state, so we use a plain dict that the main thread reads.
_ingestion_status: dict[str, object] = {}
_ingestion_lock = threading.Lock()

# Module-level dict for the folder dialog result — the background thread
# writes the selected path here, and the main thread polls it on each
# fragment refresh so the UI never blocks.
_folder_dialog_result: dict[str, object] = {}
_folder_dialog_lock = threading.Lock()
# The currently running dialog thread, if any — checked so repeated
# "Browse…" clicks don't stack native dialogs whose results would
# overwrite each other.
_folder_dialog_thread: threading.Thread | None = None

# Tracks whether auto-ingestion of the default repo has already been
# attempted in this process lifetime (prevents re-triggering).
_auto_ingest_attempted = False
_auto_ingest_error: str | None = None


def _set_ingestion_status(**kwargs: object) -> None:
    """Update ingestion status fields under the lock."""
    with _ingestion_lock:
        _ingestion_status.update(**kwargs)


def _get_ingestion_status() -> dict[str, object]:
    """Return a snapshot of the current ingestion status."""
    with _ingestion_lock:
        return dict(_ingestion_status)


def _clear_ingestion_status() -> None:
    """Clear all ingestion status fields under the lock."""
    with _ingestion_lock:
        _ingestion_status.clear()


def display_header() -> None:
    """Display the application header."""
    st.markdown(
        """
        <style>
        /* Compact delete buttons */
        button[kind="secondary"] {
            padding: 0.25rem 0.5rem;
            min-height: 0;
            line-height: 1;
        }
        /* Reduce gap above sidebar logo */
        [data-testid="stSidebar"] [data-testid="stImage"] {
            margin-top: -3rem;
            margin-bottom: -1rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Codebase RAG")
    st.markdown(
        "Ask questions about your ingested codebases. "
        "The assistant will provide answers based on the documentation and code."
    )
    st.markdown("---")


def display_sources(sources: list[dict[str, str]]) -> None:
    """Display the sources used for the response as file paths.

    Args:
        sources: List of source information dictionaries.
    """
    if not sources:
        return

    st.markdown("### Sources")

    # Group sources by file_path to avoid duplicates
    grouped_sources: dict[str, list[dict[str, str]]] = {}
    for source in sources:
        file_path = source.get("file_path", "Unknown")
        if file_path in grouped_sources:
            grouped_sources[file_path].append(source)
        else:
            grouped_sources[file_path] = [source]

    for file_path, source_list in grouped_sources.items():
        primary_source = source_list[0]
        file_name = primary_source.get("file_name", "Unknown")

        # Show file path with repo context
        st.markdown(f"- `{file_path}` — {file_name}")


def format_message(message: dict[str, Any]) -> None:
    """Format and display a chat message with chat_message UI component.

    Args:
        message: Chat message dictionary.
    """
    role = message.get("role", "")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        # Display sources if available
        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources"):
                display_sources(sources)


def initialize_chat_history() -> None:
    """Initialize the chat history in the session state if it doesn't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.chat_histories[st.session_state.current_chat_id] = []
        st.session_state.chat_counter = 1
        _load_saved_chat_histories()


def _load_saved_chat_histories() -> None:
    """Load saved chat histories from persistent storage into session state."""
    try:
        chat_manager = get_chat_history_manager()
        chat_list = chat_manager.list_chat_histories()
        if not chat_list:
            return

        _load_most_recent_chat(chat_manager, chat_list[0])

        for chat_metadata in chat_list[1:]:
            _load_chat_into_session(chat_manager, chat_metadata)

        logger.info("Loaded %d saved chats from storage", len(chat_list))

    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to load saved chat histories: %s", e)


def _load_most_recent_chat(chat_manager: Any, chat_metadata: dict[str, Any]) -> None:
    """Load the most recent chat and set it as the current chat."""
    chat_id = chat_metadata.get("chat_id")
    if not chat_id:
        return
    messages = chat_manager.get_chat_history(chat_id)
    if messages:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = list(messages)
        st.session_state.chat_histories[chat_id] = list(messages)


def _load_chat_into_session(chat_manager: Any, chat_metadata: dict[str, Any]) -> None:
    """Load a single chat history into session state."""
    chat_id = chat_metadata.get("chat_id")
    if not chat_id:
        return
    messages = chat_manager.get_chat_history(chat_id)
    if messages:
        st.session_state.chat_histories[chat_id] = messages


def display_chat_history() -> None:
    """Display the chat history from the session state."""
    for message in st.session_state.messages:
        format_message(message)


def add_message(role: str, content: str, sources: list[dict[str, str]] | None = None) -> None:
    """Add a message to the chat history.

    Args:
        role: The role of the message sender (user or assistant)
        content: The message content
        sources: Optional list of sources for assistant messages
    """
    if not hasattr(st.session_state, "messages"):
        st.session_state.messages = []

    if not content or content.strip() == "":
        content = "I apologize, but I wasn't able to generate a response. Please try rephrasing your question."
    message = {"role": role, "content": content}

    if role == "assistant" and sources:
        message["sources"] = sources  # type: ignore

    st.session_state.messages.append(message)

    # Also update the current chat in the histories
    if (
        hasattr(st.session_state, "chat_histories")
        and hasattr(st.session_state, "current_chat_id")
        and st.session_state.current_chat_id in st.session_state.chat_histories
    ):
        st.session_state.chat_histories[st.session_state.current_chat_id].append(message)

        # Persist chat history to storage
        try:
            chat_manager = get_chat_history_manager()
            chat_id = st.session_state.current_chat_id
            messages = st.session_state.chat_histories[chat_id]
            chat_manager.save_chat_history(chat_id, messages)
            logger.info("Saved chat %s to persistent storage", chat_id)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to save chat history: %s", e)


def _get_qdrant_store() -> QdrantStore:
    """Get a QdrantStore instance for repo management."""
    from codebase_rag.database.qdrant_store import QdrantStore  # Avoid circular import

    config = Config.get_instance()
    return QdrantStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.collection_name,
    )


def get_auto_ingestion_status() -> dict[str, object] | None:
    """Return the current auto-ingestion status, or None if not applicable.

    ``_ingestion_status`` is shared with manual ingestion, so a status
    whose ``kind`` isn't ``"auto"`` describes a manual run in progress —
    from this function's perspective that means auto-ingestion itself
    isn't running, even though the shared dict is currently occupied.
    """
    if not _auto_ingest_attempted:
        return None
    status = _get_ingestion_status()
    result = status if status and status.get("kind") == "auto" else {"running": False}
    if _auto_ingest_error:
        result["error"] = _auto_ingest_error
    return result


def check_and_start_auto_ingestion() -> None:
    """Check if auto-ingestion is needed and start it.

    Called from main() after initialization completes. Checks whether
    Qdrant is empty and a default repo URL is configured. If so, kicks
    off background ingestion and returns immediately.
    """
    global _auto_ingest_attempted  # noqa: PLW0603

    if _auto_ingest_attempted or _get_ingestion_status().get("running"):
        return

    config = Config.get_instance()
    default_repo = config.default_repo_url
    if not default_repo:
        return

    store = _get_qdrant_store()
    if store.collection_exists():
        try:
            repos = store.list_repos()
            if repos:
                return
        except Exception:  # noqa: BLE001
            logger.debug("Could not list repos for auto-ingestion check", exc_info=True)

    _auto_ingest_attempted = True
    logger.info("No data found. Auto-ingesting default repo: %s", default_repo)
    _run_ingestion(default_repo, kind="auto")


def _start_ingestion_status(repo_url: str, kind: str) -> bool:
    """Atomically claim the shared ingestion-status slot for a new run.

    Returns False without changing anything if an ingestion is already
    running, so callers can refuse to start a second, overlapping
    ``IngestPipeline`` (which would race on the BM25 pickle/JSON cache
    and interleave Qdrant writes).
    """
    with _ingestion_lock:
        if _ingestion_status.get("running"):
            return False
        _ingestion_status.clear()
        _ingestion_status.update(running=True, repo=repo_url, error=None, start_time=time.time(), kind=kind)
        return True


def _run_ingestion(repo_url: str, kind: str = "manual") -> None:
    """Run the ingestion pipeline for a repository in a background thread.

    Runs the ``IngestPipeline`` directly in a daemon thread so
    Streamlit's UI loop is not blocked. Progress is tracked via
    ``_ingestion_status``, tagged with ``kind`` ("auto" or "manual") so
    ``get_auto_ingestion_status()`` can tell the two apart. Refuses to
    start if another ingestion is already running.
    """
    from codebase_rag.data_ingestion.pipeline import IngestPipeline  # Avoid circular import

    if not _start_ingestion_status(repo_url, kind):
        logger.warning("Ingestion already in progress; ignoring request to ingest %s", repo_url)
        return

    logger.info("Starting ingestion for %s", repo_url)

    def _run() -> None:
        global _auto_ingest_error  # noqa: PLW0603
        try:
            pipeline = IngestPipeline(repo_urls=[repo_url], use_cache=False)
            pipeline.run()
            logger.info("Ingestion completed for %s", repo_url)
            _set_ingestion_status(running=False, error=None)
        except Exception as exc:
            logger.error("Ingestion error for %s: %s", repo_url, exc)
            _set_ingestion_status(running=False, error=str(exc))
            if kind == "auto":
                _auto_ingest_error = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@st.fragment(run_every=5)
def _display_repo_management() -> None:
    """Display repository management UI in the sidebar.

    Decorated with ``@st.fragment(run_every=5)`` so this section
    auto-refreshes every 5 seconds *independently* of the main page,
    allowing the user to keep chatting while a repository is being
    ingested.

    Must be called inside a ``with st.sidebar:`` context manager
    because ``@st.fragment`` does not allow direct ``st.sidebar`` calls.
    """
    st.subheader("Repositories")

    _display_ingestion_status()

    repos = _load_repo_list()
    _display_repo_list(repos)

    ingestion_running = bool(_get_ingestion_status().get("running"))

    with st.expander("Add Repository"):
        tab_github, tab_local = st.tabs(["GitHub URL", "Local Folder"])

        with tab_github:
            _display_github_tab(ingestion_running)

        with tab_local:
            _display_local_folder_tab(ingestion_running)


def _display_github_tab(ingestion_running: bool) -> None:
    """Render the GitHub URL input tab."""
    new_repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
        key="new_repo_url",
    )
    if st.button("Ingest", key="btn_ingest_repo", disabled=bool(not new_repo_url or ingestion_running)):
        if new_repo_url and new_repo_url.startswith("https://github.com/"):
            st.session_state.pop("github_url_error", None)
            _run_ingestion(new_repo_url, kind="manual")
            st.rerun()
        elif new_repo_url:
            st.session_state["github_url_error"] = "Please enter a valid GitHub URL"

    if st.session_state.get("github_url_error"):
        st.error(st.session_state["github_url_error"])
        if st.button("Dismiss", key="btn_dismiss_github_url_error"):
            del st.session_state["github_url_error"]
            st.rerun()


def _display_local_folder_tab(ingestion_running: bool) -> None:
    """Render the local folder picker tab."""
    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = ""

    # Poll for a result from the background folder-dialog thread.
    with _folder_dialog_lock:
        dialog_result = dict(_folder_dialog_result)
        _folder_dialog_result.clear()
    if dialog_result.get("path"):
        st.session_state.selected_folder = dialog_result["path"]
        st.session_state.pop("folder_dialog_error", None)
    elif dialog_result.get("error"):
        st.session_state["folder_dialog_error"] = dialog_result["error"]

    dialog_open = _folder_dialog_thread is not None and _folder_dialog_thread.is_alive()
    if st.button("Browse…", key="btn_browse_folder", disabled=ingestion_running or dialog_open):
        st.session_state.pop("folder_dialog_error", None)
        _open_folder_dialog()

    if _folder_dialog_thread is not None and _folder_dialog_thread.is_alive():
        st.caption("⏳ Folder dialog opened — waiting for your selection…")

    if st.session_state.get("folder_dialog_error"):
        st.error(st.session_state["folder_dialog_error"])

    if st.session_state.selected_folder:
        st.markdown(f"📂 `{st.session_state.selected_folder}`")
        folder_path = Path(st.session_state.selected_folder).resolve()
        if folder_path.is_dir():
            included_dirs, file_count = _preview_local_folder(folder_path)
            if file_count == 0:
                st.warning(
                    "No ingestible files found in this folder "
                    "(after skipping node_modules, venv, dist, and similar directories)."
                )
            else:
                dirs_label = ", ".join(included_dirs) if included_dirs else "(root)"
                st.caption(f"📄 {file_count} file(s) found in: {dirs_label}")

        if st.button("Ingest", key="btn_ingest_local", disabled=ingestion_running):
            if folder_path.is_dir():
                st.session_state.selected_folder = ""
                st.session_state.pop("_folder_preview_cache", None)
                _run_ingestion(str(folder_path), kind="manual")
                st.rerun()
            else:
                st.error("Directory does not exist")


def _preview_local_folder(folder_path: Path) -> tuple[list[str], int]:
    """Return (and cache) the discovered dirs and file count for a folder.

    Recomputing this on every 5-second fragment refresh would mean
    re-walking the filesystem repeatedly while the user just sits on the
    "Add Repository" expander, so the result is cached in session state
    until the selected folder changes.
    """
    from codebase_rag.data_ingestion.pipeline import count_ingestible_files  # Avoid circular import

    cache_key = str(folder_path)
    cached = st.session_state.get("_folder_preview_cache")
    if cached and cached.get("path") == cache_key:
        return cached["dirs"], cached["count"]

    included_dirs, file_count = count_ingestible_files(folder_path)
    st.session_state["_folder_preview_cache"] = {"path": cache_key, "dirs": included_dirs, "count": file_count}
    return included_dirs, file_count


def _open_folder_dialog() -> None:
    """Launch a native OS folder-picker dialog in a background thread.

    The dialog runs in a separate thread so Streamlit's main thread
    never blocks. The result is written to ``_folder_dialog_result`` and
    picked up by the main thread on the next fragment refresh. If a
    dialog is already open, this is a no-op instead of stacking a second
    one on top of it.
    """
    global _folder_dialog_thread
    if _folder_dialog_thread is not None and _folder_dialog_thread.is_alive():
        return

    def _run_dialog() -> None:
        path, error = _pick_folder_path()
        with _folder_dialog_lock:
            _folder_dialog_result.clear()
            if path:
                _folder_dialog_result["path"] = path
            elif error:
                _folder_dialog_result["error"] = error

    _folder_dialog_thread = threading.Thread(target=_run_dialog, daemon=True)
    _folder_dialog_thread.start()


# A bare `choose folder` run from osascript belongs to a faceless
# background process: the panel opens without keyboard focus, never
# raises above the browser, and with a fullscreen browser it lands on a
# different Space entirely — the click looks like a no-op. Routing it
# through System Events and activating first makes the dialog take
# focus and switch to the user's Space. First use prompts once for
# Automation permission (osascript → System Events).
_MACOS_CHOOSE_FOLDER_SCRIPT = (
    'tell application "System Events"\n'
    "activate\n"
    'return POSIX path of (choose folder with prompt "Select a codebase folder")\n'
    "end tell"
)


def _pick_folder_path() -> tuple[str | None, str | None]:
    """Show the native folder picker and return ``(path, error)``.

    ``(path, None)`` on selection, ``(None, None)`` on cancel/timeout,
    and ``(None, message)`` for real failures the user should see.
    """
    try:
        if sys.platform == "darwin":
            result = subprocess.run(  # noqa: S603
                ["osascript", "-e", _MACOS_CHOOSE_FOLDER_SCRIPT],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        elif sys.platform == "win32":
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$d.Description = 'Select a codebase folder'; "
                "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath } else { '' }"
            )
            result = subprocess.run(  # noqa: S603
                ["powershell", "-NoProfile", "-Command", ps_script],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        elif shutil.which("zenity"):
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title=Select a codebase folder"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        elif shutil.which("kdialog"):
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", "."],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        else:
            logger.warning("No folder dialog tool available (install zenity or kdialog)")
            return None, "No folder dialog tool available — install zenity or kdialog."
        path = result.stdout.strip().rstrip("/\\")
        if path:
            return path, None
        stderr = result.stderr.strip()
        if result.returncode == 0 or not stderr or "cancel" in stderr.lower():
            return None, None
        logger.warning("Folder dialog failed: %s", stderr)
        if "-1743" in stderr or "Not authorized" in stderr:
            return None, (
                "The folder dialog needs Automation permission: allow your "
                "terminal to control System Events under System Settings → "
                "Privacy & Security → Automation, then try again."
            )
        return None, f"Folder dialog failed: {stderr}"
    except subprocess.TimeoutExpired:
        logger.warning("Folder dialog timed out waiting for a selection")
        return None, None
    except OSError as exc:
        logger.warning("Folder dialog failed: %s", exc)
        return None, f"Folder dialog failed: {exc}"


def _display_ingestion_status() -> None:
    """Show ingestion progress in the sidebar, and keep the most recent
    outcome banner (success or failure) visible until the user dismisses
    it, instead of it disappearing on the next 5-second fragment tick.
    """
    ingestion = _get_ingestion_status()

    if ingestion.get("running"):
        elapsed = int(time.time() - ingestion.get("start_time", time.time()))  # type: ignore[operator]
        st.info(f"⏳ Ingesting {ingestion['repo']}… ({elapsed}s elapsed)")
    elif ingestion.get("error") or "repo" in ingestion:
        st.session_state["ingestion_outcome"] = dict(ingestion)
        _clear_ingestion_status()
        if "repo" in ingestion and not ingestion.get("error"):
            st.cache_resource.clear()
            st.session_state.initialized = False
            st.session_state.initializing = False
            # Force a full app rerun so the main page picks up the new
            # index; the banner itself survives the rerun via session_state.
            st.rerun(scope="app")

    outcome = st.session_state.get("ingestion_outcome")
    if not outcome:
        return

    if outcome.get("error"):
        st.error(f"Ingestion failed: {outcome['error']}")
    else:
        st.success(f"✅ Ingested **{outcome['repo']}** successfully!")
    if st.button("Dismiss", key="btn_dismiss_ingestion_outcome"):
        del st.session_state["ingestion_outcome"]
        st.rerun()


def _load_repo_list() -> list[str]:
    """Load the list of ingested repositories from Qdrant."""
    try:
        store = _get_qdrant_store()
        return store.list_repos()
    except Exception as e:
        logger.warning("Could not connect to Qdrant: %s", e)
        st.warning("Could not connect to vector database")
        return []


def _remove_repo_from_bm25_index(repo_name: str) -> None:
    """Drop a repo's BM25 corpus and rebuild the combined index so deleted
    content stops being retrievable via keyword search."""
    from codebase_rag.retrieval.bm25_search import delete_bm25_corpus, rebuild_bm25_index

    cache_dir = Path("data/cache")
    delete_bm25_corpus(cache_dir / "bm25_corpus", repo_name)
    rebuild_bm25_index(cache_dir)


def _display_repo_list(repos: list[str]) -> None:
    """Render the list of ingested repositories with delete buttons."""
    if not repos:
        st.info("No repositories ingested yet.")
        return

    store = _get_qdrant_store()
    for repo_name in repos:
        cols = st.columns([6, 1])
        cols[0].markdown(f"📦 **{repo_name}**")
        if cols[1].button("✕", key=f"del_repo_{repo_name}", help=f"Remove {repo_name}"):
            with st.spinner(f"Removing {repo_name}..."):
                deleted = store.delete_by_repo(repo_name)
                _remove_repo_from_bm25_index(repo_name)
                st.success(f"Removed {repo_name} ({deleted} chunks)")
                st.cache_resource.clear()
                st.session_state.initialized = False
                st.session_state.initializing = False
                st.rerun()


def display_sidebar() -> None:
    """Display the sidebar with additional information."""
    # Guard Streamlit sidebar UI so importing this module in pytest doesn't
    # execute UI code that requires a running Streamlit runtime or files.
    logo_path = Path(__file__).parent / "logo.png"
    try:
        with st.sidebar:
            _, col, _ = st.columns([1, 2, 1])
            col.image(str(logo_path), use_container_width=True)
    except Exception as e:  # FileNotFoundError, RuntimeError, etc.
        logger.debug("Skipping sidebar image due to %s", e)
        # If Streamlit runtime isn't available (e.g., during pytest import),
        # avoid executing any further sidebar UI code.
        return
    st.sidebar.title("About")

    config = Config.get_instance()

    st.sidebar.markdown(
        f"""
        Codebase RAG is a Retrieval-Augmented Generation application for exploring and understanding codebases locally.

        It helps users understand code by providing answers based on ingested documentation and source code.

        This application uses:
        - A local LLM via Ollama (**{config.llm_model_name}**)
        - Hybrid search combining vector and BM25
        - Qdrant vector database
        """
    )

    # Repositories section, called inside sidebar context because
    # @st.fragment does not allow direct st.sidebar usage.
    with st.sidebar:
        _display_repo_management()

    # Chat history management section
    _display_new_chat_button()
    _display_chat_history_list()


def _display_new_chat_button() -> None:
    """Display the 'Start New Chat' button and handle its click."""
    if st.sidebar.button("Start New Chat", use_container_width=True):
        if hasattr(st.session_state, "chat_counter"):
            st.session_state.chat_counter += 1
        else:
            st.session_state.chat_counter = 1

        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_histories[new_chat_id] = []
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
        st.rerun()


def _get_chat_title(chat_history: list[dict[str, Any]]) -> str:
    """Derive a display title from a chat's message history."""
    if not chat_history:
        return "New Chat"
    user_messages = [msg for msg in chat_history if msg.get("role") == "user"]
    if not user_messages:
        return "Empty Chat"
    content = str(user_messages[0].get("content", ""))
    return content[:20] + "..." if len(content) > 20 else content


def _display_chat_history_list() -> None:
    """Display the list of available chat histories in the sidebar."""
    if not (hasattr(st.session_state, "chat_histories") and st.session_state.chat_histories):
        return

    st.sidebar.subheader("Chat History")

    all_chats = list(st.session_state.chat_histories.items())
    # Reverse so that the most recently added/loaded chats appear first.
    # Storage returns chats sorted by last_updated DESC so insertion
    # order is already meaningful; new in-session chats are appended.
    all_chats.reverse()

    for chat_id, chat_history in all_chats:
        chat_title = _get_chat_title(chat_history)
        if st.session_state.current_chat_id == chat_id:
            chat_title = f"➤ {chat_title}"

        cols = st.sidebar.columns([6, 1])
        if cols[0].button(chat_title, key=f"btn_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat_history.copy()
            st.rerun()
        if cols[1].button("✕", key=f"del_{chat_id}"):
            _delete_chat(chat_id)


def _delete_chat(chat_id: str) -> None:
    """Delete a chat from session state and persistent storage, then rerun."""
    try:
        del st.session_state.chat_histories[chat_id]

        if st.session_state.current_chat_id == chat_id:
            _switch_to_next_chat()

        try:
            chat_manager = get_chat_history_manager()
            chat_manager.delete_chat_history(chat_id)
            logger.info("Deleted chat %s from persistent storage", chat_id)
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Failed to delete chat from persistent storage: %s", e)

        st.rerun()
    except (KeyError, ValueError) as e:
        logger.error("Failed to delete chat: %s", e)


def _switch_to_next_chat() -> None:
    """Switch to the next available chat, or create a new one if none remain."""
    if st.session_state.chat_histories:
        new_current = next(iter(st.session_state.chat_histories.keys()))
        st.session_state.current_chat_id = new_current
        st.session_state.messages = st.session_state.chat_histories[new_current].copy()
    else:
        new_chat_id = str(uuid.uuid4())
        st.session_state.chat_histories[new_chat_id] = []
        st.session_state.current_chat_id = new_chat_id
        st.session_state.messages = []
