"""Process-wide resources shared by every session.

``AppRuntime`` is the single ``@st.cache_resource`` root: it owns the one
``QdrantClient``, the one LLM client, the retrievers, and the
``IngestionManager``. UI modules read from it and never construct clients,
stores, or pipelines themselves.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import streamlit as st

from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever
from codebase_rag.services.folder_picker import FolderPicker

logger = logging.getLogger(__name__)

IngestKind = Literal["auto", "manual"]
IngestState = Literal["running", "succeeded", "failed"]


@dataclass
class IngestJob:
    kind: IngestKind
    source: str
    state: IngestState = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    acknowledged: bool = False


class IngestionManager:
    """Single owner of background ingestion: one job, one status object.

    Replaces the module-level ``_ingestion_status`` / ``_auto_ingest_*``
    globals and their ad-hoc locks. ``start()`` is a compare-and-set under
    the manager's own lock. That's the actual concurrency guard, since a
    render-time ``disabled=`` snapshot on a button can't stop a second
    click from racing the first between renders.
    """

    def __init__(self, on_success: Callable[[IngestJob], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._job: IngestJob | None = None
        self._on_success = on_success

    def start(self, source: str, kind: IngestKind) -> bool:
        """Claim the single ingestion slot, or refuse if one is running."""
        with self._lock:
            if self._job is not None and self._job.state == "running":
                return False
            job = IngestJob(kind=kind, source=source)
            self._job = job

        def _run() -> None:
            from codebase_rag.data_ingestion.pipeline import IngestPipeline

            try:
                pipeline = IngestPipeline(repo_urls=[source], use_cache=False)
                pipeline.run()
            except Exception as exc:  # noqa: BLE001 - surfaced via IngestJob.error, not swallowed
                logger.error("Ingestion error for %s: %s", source, exc)
                with self._lock:
                    job.state = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()
                return

            with self._lock:
                job.state = "succeeded"
                job.finished_at = time.time()
            logger.info("Ingestion completed for %s", source)
            if self._on_success:
                try:
                    self._on_success(job)
                except Exception as exc:  # noqa: BLE001 - a hook failure must not undo a real success
                    logger.error("Post-ingest hook failed for %s: %s", source, exc)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def current_job(self) -> IngestJob | None:
        """Return the running job, if any."""
        with self._lock:
            if self._job is not None and self._job.state == "running":
                return self._job
            return None

    def last_completed(self) -> IngestJob | None:
        """Return the most recent finished job until it's acknowledged."""
        with self._lock:
            if self._job is not None and self._job.state != "running" and not self._job.acknowledged:
                return self._job
            return None

    def acknowledge(self) -> None:
        with self._lock:
            if self._job is not None:
                self._job.acknowledged = True

    def auto_job_error(self) -> str | None:
        """Error from the most recent auto job, kept visible past acknowledgement.

        Manual jobs never gate the chat surface, so only an auto failure
        needs to survive past the banner being dismissed (the chat-gating
        check runs on every rerun, independent of the banner's lifecycle).
        """
        with self._lock:
            if self._job is not None and self._job.kind == "auto" and self._job.state == "failed":
                return self._job.error
            return None


def _load_or_create_bm25_retriever() -> BM25Retriever:
    """Load BM25 retriever from cache or create a new (empty) one."""
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    bm25_file = cache_dir / "bm25_retriever.json"

    if bm25_file.exists():
        logger.info("Loaded BM25 retriever from cache")
        return BM25Retriever.load_json(bm25_file)

    logger.info("No BM25 cache found; starting with an empty retriever until the first ingest")
    return BM25Retriever([])


def _warm_up_vector_store(vector_retriever: VectorRetriever) -> None:
    try:
        vector_retriever.search("What does this codebase do?", k=1)
        logger.info("Vector store warm-up successful")
    except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.warning("Vector store warm-up failed: %s", e)


def _run_health_checks(llm: OllamaClient, vector_retriever: VectorRetriever) -> None:
    """Best-effort connectivity checks, logged only. Run off the main
    thread so a slow/unreachable Ollama never blocks the first render.
    """
    llm_status = llm.check_connection()
    if llm_status["status"] != "connected":
        logger.warning("LLM connection issue: %s", llm_status["message"])
    model_status = llm.check_model_availability()
    if model_status["status"] != "available":
        logger.warning("Model availability issue: %s", model_status["message"])
    _warm_up_vector_store(vector_retriever)


MAX_CONVERSATION_HISTORY = 10


class AppRuntime:
    """Process-wide resource root: one Qdrant client, one LLM client, one
    set of retrievers, and the ingestion manager, all sharing a single
    ``@st.cache_resource`` lifetime across every session.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.qdrant_store = QdrantStore(
            host=config.qdrant_host, port=config.qdrant_port, collection_name=config.collection_name
        )
        if not self.qdrant_store.collection_exists():
            logger.warning(
                "Qdrant collection '%s' does not exist yet; it will be created on first ingestion.",
                config.collection_name,
            )

        self.vector_retriever = VectorRetriever(self.qdrant_store)
        self.bm25_retriever = _load_or_create_bm25_retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            vector_weight=0.7,
            bm25_weight=0.3,
            min_score_threshold=0.1,
        )

        self.llm = OllamaClient(
            model_name=config.llm_model_name,
            base_url=config.ollama_base_url,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            timeout=120,
        )
        threading.Thread(target=_run_health_checks, args=(self.llm, self.vector_retriever), daemon=True).start()

        self.folder_picker = FolderPicker()
        self.ingestion = IngestionManager(on_success=self._on_ingest_success)

        self._auto_ingest_checked = False
        self._check_auto_ingest()

    def new_rag_chain(self) -> RAGChain:
        """Build a fresh, per-session ``RAGChain`` sharing this runtime's retriever.

        A new instance per session keeps conversation history isolated
        without needing a new retriever, LLM client, or Qdrant connection.
        """
        return RAGChain(
            retriever=self.hybrid_retriever,
            llm=self.llm,
            use_conversation_memory=True,
            max_conversation_history=MAX_CONVERSATION_HISTORY,
        )

    def swap_bm25(self, index: BM25Retriever) -> None:
        """Atomically replace the hybrid retriever's BM25 component.

        Called after a successful ingest instead of clearing
        ``st.cache_resource`` and rebuilding everything: the embedding
        model, Qdrant client, and LLM client survive untouched, and every
        open session sees the new index on its next rerun because they
        share this runtime.
        """
        self.bm25_retriever = index
        self.hybrid_retriever.bm25_retriever = index

    def _on_ingest_success(self, _job: IngestJob) -> None:
        get_repo_list.clear()  # type: ignore[attr-defined]
        from codebase_rag.retrieval.bm25_search import rebuild_bm25_index

        cache_dir = Path("data/cache")
        self.swap_bm25(rebuild_bm25_index(cache_dir))

    def _check_auto_ingest(self) -> None:
        """Check once, at construction time, whether the default repo needs
        auto-ingesting. ``@st.cache_resource`` already gives the
        once-per-process semantics the old ``_auto_ingest_attempted``
        global reimplemented by hand.
        """
        default_repo = self.config.default_repo_url
        if not default_repo:
            return
        if self.qdrant_store.collection_exists():
            try:
                if self.qdrant_store.list_repos():
                    return
            except Exception:  # noqa: BLE001
                logger.debug("Could not list repos for auto-ingestion check", exc_info=True)

        logger.info("No data found. Auto-ingesting default repo: %s", default_repo)
        self.ingestion.start(default_repo, kind="auto")

    def delete_repo(self, repo_name: str) -> int:
        """Delete a repo from Qdrant and rebuild BM25 so it stops being
        retrievable via keyword search too (fixes the AI-2 leak)."""
        from codebase_rag.retrieval.bm25_search import delete_bm25_corpus, rebuild_bm25_index

        deleted = self.qdrant_store.delete_by_repo(repo_name)
        cache_dir = Path("data/cache")
        delete_bm25_corpus(cache_dir / "bm25_corpus", repo_name)
        self.swap_bm25(rebuild_bm25_index(cache_dir))
        get_repo_list.clear()  # type: ignore[attr-defined]
        return deleted


@st.cache_resource
def get_runtime() -> AppRuntime:
    """The one process-wide ``AppRuntime``, built once per process."""
    return AppRuntime(Config.get_instance())


@st.cache_data(ttl=30)
def get_repo_list(_qdrant_store: QdrantStore) -> list[str]:
    """Cached repo list: zero Qdrant calls between invalidations/TTL,
    versus a client construction and a call on every poll under the old
    always-on fragment. Explicitly invalidated on ingest completion and
    repo deletion via ``get_repo_list.clear()``.
    """
    try:
        return _qdrant_store.list_repos()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not connect to Qdrant: %s", e)
        return []


@st.cache_data(ttl=30)
def list_chat_metadata() -> list[dict[str, Any]]:
    """Cached chat-storage listing used for sidebar ordering: avoids a disk
    scan on every rerun. Explicitly invalidated on save/delete via
    ``list_chat_metadata.clear()``.
    """
    from codebase_rag.database.chat_storage import get_chat_history_manager

    try:
        return get_chat_history_manager().list_chat_histories()
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not list chat histories: %s", e)
        return []
