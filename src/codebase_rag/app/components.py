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
    """Return the current auto-ingestion status, or None if not applicable."""
    if not _auto_ingest_attempted:
        return None
    status = _get_ingestion_status()
    result = status if status else {"running": False}
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
    _run_ingestion(default_repo)


def _run_ingestion(repo_url: str) -> None:
    """Run the ingestion pipeline for a repository in a background thread.

    Runs the ``IngestPipeline`` directly in a daemon thread so
    Streamlit's UI loop is not blocked. Progress is tracked via
    ``_ingestion_status``.
    """
    from codebase_rag.data_ingestion.pipeline import IngestPipeline  # Avoid circular import

    logger.info("Starting ingestion for %s", repo_url)

    # Track status in module-level dict (threads can't write st.session_state)
    _set_ingestion_status(running=True, repo=repo_url, error=None, start_time=time.time())

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
            _run_ingestion(new_repo_url)
            st.rerun()
        elif new_repo_url:
            st.error("Please enter a valid GitHub URL")


def _display_local_folder_tab(ingestion_running: bool) -> None:
    """Render the local folder picker tab."""
    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = ""

    if st.button("Browse…", key="btn_browse_folder", disabled=ingestion_running):
        folder = _open_folder_dialog()
        if folder:
            st.session_state.selected_folder = folder

    if st.session_state.selected_folder:
        st.markdown(f"📂 `{st.session_state.selected_folder}`")
        if st.button("Ingest", key="btn_ingest_local", disabled=ingestion_running):
            resolved = Path(st.session_state.selected_folder).resolve()
            if resolved.is_dir():
                st.session_state.selected_folder = ""
                _run_ingestion(str(resolved))
                st.rerun()
            else:
                st.error("Directory does not exist")


def _open_folder_dialog() -> str | None:
    """Open a native OS folder-picker dialog and return the selected path.

    Uses AppleScript on macOS, PowerShell on Windows, and zenity/kdialog on Linux.
    """
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select a codebase folder")'],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=120,
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
            return None
        path = result.stdout.strip().rstrip("/\\")
        return path if path else None
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Folder dialog failed: %s", exc)
        return None


def _display_ingestion_status() -> None:
    """Show ingestion progress/result banner in the sidebar."""
    ingestion = _get_ingestion_status()
    if not ingestion:
        return

    if ingestion.get("running"):
        elapsed = int(time.time() - ingestion.get("start_time", time.time()))  # type: ignore[operator]
        st.info(f"⏳ Ingesting {ingestion['repo']}… ({elapsed}s elapsed)")
    elif ingestion.get("error"):
        st.error(f"Ingestion failed: {ingestion['error']}")
        _clear_ingestion_status()
    elif "repo" in ingestion:
        st.success(f"✅ Ingested **{ingestion['repo']}** successfully!")
        _clear_ingestion_status()
        st.cache_resource.clear()
        st.session_state.initialized = False
        st.session_state.initializing = False
        # Force a full app rerun so the main page picks up the new state
        st.rerun(scope="app")


def _load_repo_list() -> list[str]:
    """Load the list of ingested repositories from Qdrant."""
    try:
        store = _get_qdrant_store()
        return store.list_repos()
    except Exception as e:
        logger.warning("Could not connect to Qdrant: %s", e)
        st.warning("Could not connect to vector database")
        return []


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
