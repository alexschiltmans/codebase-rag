"""Single in-process ingest job slot, guarding concurrent `/ingest` requests.

Mirrors the Streamlit app's `IngestionManager` (see `app/runtime.py`), but
triggers `IngestPipeline.process_repo_incremental` instead of a full `.run()`,
and keeps the job model minimal (one in-memory job, no task queue) per the
design's non-goals.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

JobState = Literal["running", "succeeded", "failed"]


@dataclass
class IngestJob:
    source: str
    state: JobState = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None


class ApiIngestionManager:
    """One job at a time; a second request while one is running gets refused."""

    def __init__(self, on_success: Callable[[IngestJob], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._job: IngestJob | None = None
        self._on_success = on_success

    def start(self, source: str) -> IngestJob | None:
        """Claim the single ingestion slot, or return None if one is already running."""
        with self._lock:
            if self._job is not None and self._job.state == "running":
                return None
            job = IngestJob(source=source)
            self._job = job

        def _run() -> None:
            from codebase_rag.data_ingestion.pipeline import IngestPipeline

            try:
                pipeline = IngestPipeline()
                result = pipeline.process_repo_incremental(source)
                with self._lock:
                    # result before state: routers read job fields without the
                    # lock, so a poll that sees "succeeded" must also see result.
                    job.result = {
                        "repo_name": result.repo_name,
                        "files_changed": result.files_changed,
                        "files_deleted": result.files_deleted,
                        "files_unchanged": result.files_unchanged,
                        "chunks_indexed": result.chunks_indexed,
                        "head_sha": result.head_sha,
                    }
                    job.finished_at = time.time()
                    job.state = "succeeded"
            except Exception as exc:  # noqa: BLE001 - surfaced via IngestJob.error, not swallowed
                logger.error("Ingestion error for %s: %s", source, exc)
                with self._lock:
                    job.error = str(exc)
                    job.finished_at = time.time()
                    job.state = "failed"
                return

            if self._on_success:
                try:
                    self._on_success(job)
                except Exception as exc:  # noqa: BLE001 - a hook failure must not undo a real success
                    logger.error("Post-ingest hook failed for %s: %s", source, exc)

        threading.Thread(target=_run, daemon=True).start()
        return job

    def current_job(self) -> IngestJob | None:
        with self._lock:
            return self._job
