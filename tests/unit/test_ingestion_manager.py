"""Unit tests for IngestionManager: the CAS start() guard and auto/manual separation."""

import threading
import time
from unittest.mock import MagicMock, patch

from codebase_rag.app.runtime import IngestionManager


def _blocking_pipeline_cls(release: threading.Event, calls: list[str]) -> MagicMock:
    """A fake IngestPipeline whose .run() blocks until `release` is set."""

    def _run(self: MagicMock) -> None:
        calls.append("run")
        release.wait(timeout=5)

    pipeline_cls = MagicMock()
    pipeline_cls.return_value.run.side_effect = lambda: _run(pipeline_cls.return_value)
    return pipeline_cls


class TestStartCAS:
    def test_second_start_while_running_returns_false(self) -> None:
        """Regression test for FE-3: two overlapping start() calls must not
        both launch a pipeline invocation."""
        release = threading.Event()
        calls: list[str] = []
        pipeline_cls = _blocking_pipeline_cls(release, calls)

        manager = IngestionManager()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            assert manager.start("repo-a", kind="manual") is True
            assert manager.start("repo-b", kind="manual") is False
            release.set()
            time.sleep(0.2)

        assert calls == ["run"]

    def test_start_after_completion_succeeds(self) -> None:
        manager = IngestionManager()
        pipeline_cls = MagicMock()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            assert manager.start("repo-a", kind="manual") is True
            for _ in range(50):
                if manager.current_job() is None:
                    break
                time.sleep(0.05)
            assert manager.start("repo-b", kind="manual") is True


class TestAutoManualSeparation:
    def test_failed_manual_job_does_not_set_auto_error(self) -> None:
        """Regression test for FE-2: a failed manual ingest must never be
        reported as a default-repository failure."""
        manager = IngestionManager()
        pipeline_cls = MagicMock()
        pipeline_cls.return_value.run.side_effect = RuntimeError("manual boom")
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("repo-a", kind="manual")
            for _ in range(50):
                if manager.last_completed() is not None:
                    break
                time.sleep(0.05)

        assert manager.auto_job_error() is None
        job = manager.last_completed()
        assert job is not None
        assert job.state == "failed"
        assert job.kind == "manual"

    def test_failed_auto_job_sets_auto_error_and_survives_acknowledge(self) -> None:
        manager = IngestionManager()
        pipeline_cls = MagicMock()
        pipeline_cls.return_value.run.side_effect = RuntimeError("auto boom")
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("default-repo", kind="auto")
            for _ in range(50):
                if manager.last_completed() is not None:
                    break
                time.sleep(0.05)

        assert manager.auto_job_error() == "auto boom"
        manager.acknowledge()
        assert manager.last_completed() is None
        # The chat-gating check (auto_job_error) survives past acknowledgement,
        # independent of the banner's own dismiss lifecycle.
        assert manager.auto_job_error() == "auto boom"

    def test_running_manual_job_does_not_gate_via_current_job_kind(self) -> None:
        release = threading.Event()
        pipeline_cls = _blocking_pipeline_cls(release, [])
        manager = IngestionManager()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("repo-a", kind="manual")
            job = manager.current_job()
            assert job is not None
            assert job.kind == "manual"
            release.set()
