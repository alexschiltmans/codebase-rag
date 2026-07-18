"""Main Streamlit application for the Codebase RAG interface."""

import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from codebase_rag.app.components import (
    add_message,
    check_and_start_auto_ingestion,
    display_chat_history,
    display_header,
    display_sidebar,
    get_auto_ingestion_status,
    initialize_chat_history,
)
from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever

_LOGO_PATH = str(Path(__file__).parent / "logo.png")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Configure the page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Codebase RAG",
    page_icon=_LOGO_PATH,
    layout="wide",
)


def initialize_session_state() -> None:
    """Initialize session state variables."""
    session_defaults = {
        "initialized": False,
        "initializing": False,
        "initialization_error": None,
        "should_retry": False,
        "retry_count": 0,
        "processing_query": False,
        "thinking": False,
        "query_to_process": None,
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def load_or_create_bm25_retriever() -> BM25Retriever:
    """Load BM25 retriever from cache or create a new one."""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    bm25_file = cache_dir / "bm25_retriever.json"

    if bm25_file.exists():
        retriever = BM25Retriever.load_json(bm25_file)
        logger.info("Loaded BM25 retriever from cache")
        return retriever

    sample_docs = [
        Document(
            page_content="This is a sample document for initializing the search index.",
            metadata={"source": "sample"},
        ),
        Document(
            page_content="Codebase RAG provides a RAG interface to explore codebases.",
            metadata={"source": "sample"},
        ),
        Document(
            page_content="You can ask questions about your code and get AI-powered responses.",
            metadata={"source": "sample"},
        ),
    ]

    logger.info("No BM25 cache found; using an in-memory sample retriever until the first ingest")
    return BM25Retriever(sample_docs)


def initialize_vector_store(config: Config) -> QdrantStore:
    """Initialize and validate Qdrant vector store.

    If the collection does not yet exist it will be created on the first
    ingestion run.  The app is still usable — the sidebar will show
    "No repositories ingested yet" and the user can trigger ingestion
    from the UI.
    """
    qdrant_store = QdrantStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.collection_name,
    )

    if not qdrant_store.collection_exists():
        logger.warning(
            "Qdrant collection '%s' does not exist yet. It will be created when a repository is ingested.",
            config.collection_name,
        )
    else:
        logger.info("Successfully initialized Qdrant store with collection '%s'", config.collection_name)

    return qdrant_store


def initialize_llm(config: Config) -> OllamaClient:
    """Initialize and validate LLM."""
    llm = OllamaClient(
        model_name=config.llm_model_name,
        base_url=config.ollama_base_url,
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_tokens=1024,
        timeout=120,
    )

    llm_status = llm.check_connection()
    if llm_status["status"] != "connected":
        logger.warning("LLM connection issue: %s", llm_status["message"])

    model_status = llm.check_model_availability()
    if model_status["status"] != "available":
        logger.warning("Model availability issue: %s", model_status["message"])

    logger.info("Successfully initialized LLM")
    return llm


def warm_up_vector_store(vector_retriever: VectorRetriever) -> None:
    """Warm up the vector store with a test query."""
    try:
        logger.info("Warming up vector store with a test query...")
        warm_up_query = "What does this codebase do?"
        vector_retriever.search(warm_up_query, k=1)
        logger.info("Vector store warm-up successful")
    except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.warning("Vector store warm-up failed: %s", e)


@st.cache_resource
def initialize_app_components() -> dict:
    """Initialize all application components.

    Returns:
        dict: Dictionary containing all initialized components or None values if initialization failed
    """
    components: dict[str, Any] = {
        "rag_chain": None,
        "vector_retriever": None,
        "bm25_retriever": None,
        "hybrid_retriever": None,
        "llm": None,
    }

    config = Config.get_instance()

    qdrant_store = initialize_vector_store(config)

    components["vector_retriever"] = VectorRetriever(qdrant_store)
    logger.info("Successfully initialized vector retriever")

    components["bm25_retriever"] = load_or_create_bm25_retriever()

    if components["vector_retriever"] and components["bm25_retriever"]:
        components["hybrid_retriever"] = HybridRetriever(
            vector_retriever=components["vector_retriever"],
            bm25_retriever=components["bm25_retriever"],
            vector_weight=0.7,
            bm25_weight=0.3,
            min_score_threshold=0.1,
        )
        logger.info("Successfully initialized hybrid retriever")

    components["llm"] = initialize_llm(config)

    retriever = components["hybrid_retriever"] if components["hybrid_retriever"] else components["vector_retriever"]
    components["rag_chain"] = RAGChain(
        retriever=retriever, llm=components["llm"], use_conversation_memory=True, max_conversation_history=10
    )
    logger.info("Successfully initialized RAG chain with conversation memory enabled")

    # Warm up the vector store
    if components["vector_retriever"] is not None:
        warm_up_vector_store(components["vector_retriever"])

    # Mark as successful initialization
    st.session_state.initialization_error = None
    return components


def process_user_query(query: str) -> None:
    """Process a user query and generate a response.

    Args:
        query: The user's query text
    """
    if not st.session_state.initialized:
        add_message("assistant", "I'm still initializing. Please wait a moment and try again.")
        st.session_state.thinking = False
        st.session_state.query_to_process = None
        return

    st.session_state.processing_query = True

    rag_chain = _get_rag_chain()

    if rag_chain:
        _run_rag_query(rag_chain, query)
    else:
        add_message("assistant", "I'm having trouble connecting to the knowledge base. Please try again later.")

    st.session_state.processing_query = False
    st.session_state.thinking = False
    st.session_state.query_to_process = None


def _get_rag_chain() -> Any:
    """Get the RAG chain from session state components."""
    if hasattr(st.session_state, "components") and st.session_state.components:
        return st.session_state.components.get("rag_chain")
    return None


def _run_rag_query(rag_chain: Any, query: str) -> None:
    """Execute a query against the RAG chain and add the response."""
    if hasattr(st.session_state, "messages") and st.session_state.messages:
        rag_chain.conversation_history = []
        # Exclude the last message (the current user query) because
        # rag_chain.run() will add it via add_user_message.
        history_messages = st.session_state.messages[:-1]
        for msg in history_messages:
            if msg["role"] == "user":
                rag_chain.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                rag_chain.add_assistant_message(msg["content"], msg.get("sources"))

    try:
        response = rag_chain.run(query)
        add_message("assistant", response["answer"], response["sources"])
    except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.error("Error generating response: %s", e)
        add_message("assistant", f"I encountered an error while processing your question: {e}")


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    initialize_chat_history()

    display_header()
    display_sidebar()

    if not st.session_state.initialized and not st.session_state.initializing:
        _try_initialize_components()

    if st.session_state.initialized:
        check_and_start_auto_ingestion()

    _display_initialization_status()
    _display_chat_interface()


def _try_initialize_components() -> None:
    """Attempt to initialize all app components."""
    st.session_state.initializing = True

    with st.spinner("Initializing application components..."):
        try:
            components = initialize_app_components()
            st.session_state.components = components

            if (
                components["rag_chain"] is not None
                and components["vector_retriever"] is not None
                and components["llm"] is not None
            ):
                st.session_state.initialized = True
                st.session_state.initializing = False
                st.session_state.initialization_error = None
                logger.info("Application initialized successfully")
                st.rerun()
            else:
                st.session_state.initializing = False
                st.error("Failed to initialize essential components")
        except (ConnectionError, ValueError, RuntimeError, ImportError) as e:
            st.session_state.initializing = False
            st.session_state.initialization_error = str(e)
            logger.error("Initialization failed: %s", e)


def _display_initialization_status() -> None:
    """Show initialization status and retry button in the sidebar."""
    with st.sidebar:
        st.divider()
        if st.session_state.initialized:
            return

        st.error("✗ Application failed to initialize")
        if st.session_state.initialization_error:
            with st.expander("Error details"):
                st.code(st.session_state.initialization_error)

        if st.button("Retry Initialization"):
            st.session_state.should_retry = True
            st.session_state.initializing = False
            st.rerun()


def _display_setup_banner() -> None:
    """Show a prominent banner while the default repo is being ingested."""
    status = get_auto_ingestion_status()
    if not status or not status.get("running"):
        return

    elapsed = int(time.time() - float(str(status.get("start_time", time.time()))))
    repo = str(status.get("repo", "default repository"))
    repo_name = repo.rstrip("/").rsplit("/", 1)[-1] if "/" in repo else repo

    st.info(
        f"🚀 **Getting ready…**\n\n"
        f"Preparing **{repo_name}** so you can start exploring right away. "
        f"This usually takes a few minutes on first startup.\n\n"
        f"⏳ Elapsed: {elapsed}s",
        icon="🔄",
    )


def _display_chat_interface() -> None:
    """Display the main chat interface."""
    if not st.session_state.initialized:
        st.chat_input("Initializing application...", disabled=True)
        return

    # Show setup progress if auto-ingestion of the default repo is running
    auto_status = get_auto_ingestion_status()
    if auto_status and auto_status.get("running"):
        _auto_ingestion_progress()
        return

    # Show warning if auto-ingestion failed
    if auto_status and auto_status.get("error"):
        st.warning(
            f"Default repository ingestion failed: {auto_status['error']}\n\n"
            "You can add a repository manually using the sidebar.",
        )

    display_chat_history()

    if st.session_state.thinking and st.session_state.query_to_process:
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            process_user_query(st.session_state.query_to_process)
            st.rerun()

    if prompt := st.chat_input("Ask about your codebase"):
        add_message("user", prompt)
        st.session_state.thinking = True
        st.session_state.query_to_process = prompt
        st.rerun()


@st.fragment(run_every=3)
def _auto_ingestion_progress() -> None:
    """Self-refreshing fragment that shows ingestion progress.

    Refreshes every 3 seconds independently of the main page.
    When ingestion completes, triggers a full app rerun.
    """
    status = get_auto_ingestion_status()

    if not status or not status.get("running"):
        # Ingestion finished — trigger full page rerun to unlock the chat
        st.rerun(scope="app")
        return

    _display_setup_banner()
    st.chat_input("Getting ready \u2014 please wait\u2026", disabled=True)


if __name__ == "__main__":
    main()
