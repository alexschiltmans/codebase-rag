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
from typing import Any, Literal

import streamlit as st

from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.retriever_protocol import RetrieverProtocol
from codebase_rag.retrieval.vector_search import VectorRetriever, resolve_score_threshold
from codebase_rag.services.folder_picker import FolderPicker

logger = logging.getLogger(__name__)

IngestKind = Literal["auto", "manual"]
IngestState = Literal["running", "succeeded", "failed", "cancelled"]


@dataclass
class IngestJob:
    kind: IngestKind
    source: str
    state: IngestState = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    acknowledged: bool = False
    phase: str = ""
    progress_current: int = 0
    progress_total: int = 0


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
        self._cancel_event: threading.Event | None = None
        self._on_success = on_success

    def start(self, source: str, kind: IngestKind) -> bool:
        """Claim the single ingestion slot, or refuse if one is running."""
        from codebase_rag.data_ingestion.pipeline import IngestCancelled

        with self._lock:
            if self._job is not None and self._job.state == "running":
                return False
            job = IngestJob(kind=kind, source=source)
            self._job = job
            cancel_event = threading.Event()
            self._cancel_event = cancel_event

        def _progress_callback(phase: str, current: int, total: int) -> None:
            with self._lock:
                job.phase = phase
                job.progress_current = current
                job.progress_total = total

        def _run() -> None:
            from codebase_rag.data_ingestion.pipeline import IngestPipeline

            try:
                pipeline = IngestPipeline(
                    repo_urls=[source],
                    use_cache=False,
                    progress_callback=_progress_callback,
                    cancel_event=cancel_event,
                )
                pipeline.run()
            except IngestCancelled:
                logger.info("Ingestion cancelled for %s", source)
                with self._lock:
                    job.state = "cancelled"
                    job.finished_at = time.time()
                return
            except Exception as exc:
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
                except Exception as exc:
                    logger.error("Post-ingest hook failed for %s: %s", source, exc)

        threading.Thread(target=_run, daemon=True).start()
        return True

    def cancel(self) -> None:
        """Signal the running job's cancel event, if any."""
        with self._lock:
            if self._cancel_event is not None and self._job is not None and self._job.state == "running":
                self._cancel_event.set()

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

    def auto_job_cancelled(self) -> bool:
        """Whether the most recent auto job ended cancelled, kept visible
        past acknowledgement for the same reason as ``auto_job_error``.

        Kept separate from ``auto_job_error`` (rather than folding a synthetic
        message into it) so the ungating gate can word a cancellation without
        borrowing "failed" phrasing.
        """
        with self._lock:
            return self._job is not None and self._job.kind == "auto" and self._job.state == "cancelled"


def _load_or_create_bm25_retriever() -> BM25Retriever:
    """Load BM25 retriever from cache or create a new (empty) one."""
    cache_dir = Config.get_instance().cache_dir
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


def _check_placement(runtime: AppRuntime) -> dict[str, Any]:
    """Ask the backend where the model is running, degrading to unknown on any failure.

    Swallows everything the client can raise: placement is the least important of the three
    results, and losing the endpoint and the model status over it would cost more than it
    is worth.
    """
    try:
        return runtime.llm.check_runtime_placement()
    except Exception as e:
        logger.warning("Could not determine runtime placement: %s", e)
        # Same shape as the clients return, so a caller never has to special-case the failure.
        return {"placement": "unknown", "url": getattr(runtime.llm, "base_url", None)}


def _run_health_checks(runtime: AppRuntime) -> None:
    """Best-effort connectivity checks, logged only. Run off the
    main thread so a slow/unreachable server never blocks the first render.
    """
    try:
        llm_status = runtime.llm.check_connection()
        if llm_status["status"] != "connected":
            logger.warning("LLM connection issue: %s", llm_status["message"])
        model_status = runtime.llm.check_model_availability()
        if model_status["status"] != "available":
            logger.warning("Model availability issue: %s", model_status["message"])
        placement_status = _check_placement(runtime)
        # One assignment, because the sidebar treats any truthy health dict as complete.
        runtime.health = {
            "connection": llm_status,
            "model": model_status,
            "placement": placement_status,
            "checked_at": time.time(),
        }
        logger.info(
            "Generation backend resolved: %s (model '%s', placement: %s)",
            llm_status.get("url", "unknown"),
            runtime.config.llm_model_name,
            placement_status.get("placement", "unknown"),
        )
    except Exception as e:
        # Deliberately not `return`-ing here: the LLM check and the vector-store
        # warm-up are independent, so a failure in one must not skip the other.
        logger.warning("Health checks failed: %s", e)
    try:
        _warm_up_vector_store(runtime.vector_retriever)
    except Exception as e:
        logger.warning("Vector store warm-up failed: %s", e)


MAX_CONVERSATION_HISTORY = 10


class AppRuntime:
    """Process-wide resource root: one Qdrant client, one LLM client, one
    set of retrievers, and the ingestion manager, all sharing a single
    ``@st.cache_resource`` lifetime across every session.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.health: dict[str, Any] = {}
        self.qdrant_store = QdrantStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
        )
        if not self.qdrant_store.collection_exists():
            logger.warning(
                "Qdrant collection '%s' does not exist yet; it will be created on first ingestion.",
                config.collection_name,
            )

        self.vector_retriever = VectorRetriever(
            self.qdrant_store, score_threshold=resolve_score_threshold(config.embedding_model)
        )
        self.bm25_retriever = _load_or_create_bm25_retriever()

        self.llm = create_llm_client(
            model_name=config.llm_model_name,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            timeout=120,
            num_ctx=config.ollama_num_ctx,
        )
        # Composed once here, not per query: the rerank stage caches a ~2GB model on the
        # instance, so rebuilding the stack for every question would reload it from disk
        # each time. Rebuilt only when the base index is swapped (see swap_bm25).
        self.retriever = self._build_retrieval_stack()
        threading.Thread(target=_run_health_checks, args=(self,), daemon=True).start()

        self.folder_picker = FolderPicker()
        self.ingestion = IngestionManager(on_success=self._on_ingest_success)

        self._auto_ingest_checked = False
        self._check_auto_ingest()

    def new_rag_chain(self) -> RAGChain:
        """Build a fresh, per-session ``RAGChain`` sharing this runtime's retriever.

        A new instance per session keeps conversation history isolated
        without needing a new retriever, LLM client, or Qdrant connection.
        The composed retrieval stack (base + optional rerank/rewrite stages)
        is built once in ``__init__`` and shared here, so the reranker model
        is not reloaded per query.
        """
        return RAGChain(
            retriever=self.retriever,
            llm=self.llm,
            use_conversation_memory=True,
            max_conversation_history=MAX_CONVERSATION_HISTORY,
            prompt_budget_chars=self.llm.prompt_budget_chars,
        )

    def _build_retrieval_stack(self) -> RetrieverProtocol:
        """Compose the base BM25 retriever with the optional rerank/rewrite stages.

        Delegates to the shared ``apply_stages`` so the runtime, the HTTP API,
        and the eval harness cannot drift on stage ordering or enablement.
        """
        from codebase_rag.retrieval.retrieval_stack import apply_stages

        return apply_stages(self.bm25_retriever, self.config, self.llm)

    def swap_bm25(self, index: BM25Retriever) -> None:
        """Atomically replace the runtime's BM25 retriever and rebuild the stack.

        Called after a successful ingest instead of clearing
        ``st.cache_resource`` and rebuilding everything: the embedding
        model, Qdrant client, and LLM client survive untouched, and every
        open session sees the new index on its next rerun because they
        share this runtime.

        The composed stack is rebuilt so the new base index feeds the rerank and
        rewrite stages. The rebuild constructs new stage instances, so with
        reranking enabled the cached cross-encoder is dropped and the next
        question pays a full model load, roughly 2GB from the local cache. That
        is the opposite of what building the stack once in ``__init__`` is for,
        and it is worth carrying the loaded model across a swap if ingests ever
        become frequent; with reranking off by default nothing is reloaded.
        """
        from codebase_rag.retrieval.retrieval_stack import close_stages

        close_stages(getattr(self, "retriever", None))
        self.bm25_retriever = index
        self.retriever = self._build_retrieval_stack()

    def _on_ingest_success(self, _job: IngestJob) -> None:
        get_repo_list.clear()
        from codebase_rag.retrieval.bm25_search import rebuild_bm25_index

        cache_dir = self.config.cache_dir
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
            except Exception:
                logger.debug("Could not list repos for auto-ingestion check", exc_info=True)

        logger.info("No data found. Auto-ingesting default repo: %s", default_repo)
        self.ingestion.start(default_repo, kind="auto")

    def delete_repo(self, repo_name: str) -> int:
        """Delete a repo from Qdrant and rebuild BM25 so it stops being
        retrievable via keyword search too."""
        from codebase_rag.retrieval.bm25_search import delete_bm25_corpus, rebuild_bm25_index

        deleted = self.qdrant_store.delete_by_repo(repo_name)
        cache_dir = self.config.cache_dir
        delete_bm25_corpus(cache_dir / "bm25_corpus", repo_name)
        self.swap_bm25(rebuild_bm25_index(cache_dir))
        get_repo_list.clear()
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
    except Exception as e:
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
