"""Unit tests for IngestionManager: the CAS start() guard and auto/manual separation."""

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from codebase_rag.app.runtime import IngestionManager
from codebase_rag.data_ingestion.pipeline import IngestCancelled


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


class TestCancel:
    def _cancel_aware_pipeline_cls(self) -> MagicMock:
        """A fake IngestPipeline whose .run() raises IngestCancelled once the
        cancel_event passed to its constructor is set, mimicking the real
        pipeline's cooperative-cancellation boundary check."""

        def _init(self: Any, **kwargs: object) -> None:
            self._cancel_event = kwargs.get("cancel_event")

        def _run(self: Any) -> None:
            event = self._cancel_event
            assert isinstance(event, threading.Event)
            event.wait(timeout=5)
            raise IngestCancelled("cancelled")

        pipeline_cls = MagicMock(side_effect=lambda **kwargs: _FakePipeline(kwargs))

        class _FakePipeline:
            def __init__(self, kwargs: dict[str, Any]) -> None:
                _init(self, **kwargs)

            def run(self) -> None:
                _run(self)

        return pipeline_cls

    def test_cancel_before_finish_marks_job_cancelled(self) -> None:
        pipeline_cls = self._cancel_aware_pipeline_cls()
        manager = IngestionManager()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("repo-a", kind="manual")
            job = manager.current_job()
            assert job is not None

            manager.cancel()

            for _ in range(50):
                if manager.last_completed() is not None:
                    break
                time.sleep(0.05)

        job = manager.last_completed()
        assert job is not None
        assert job.state == "cancelled"
        assert manager.current_job() is None

    def test_cancel_with_no_running_job_is_a_no_op(self) -> None:
        manager = IngestionManager()
        manager.cancel()  # must not raise
        assert manager.current_job() is None

    def test_acknowledge_after_cancel_clears_last_completed(self) -> None:
        pipeline_cls = self._cancel_aware_pipeline_cls()
        manager = IngestionManager()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("repo-a", kind="manual")
            manager.cancel()
            for _ in range(50):
                if manager.last_completed() is not None:
                    break
                time.sleep(0.05)

        manager.acknowledge()
        assert manager.last_completed() is None

    def test_cancelled_auto_job_reports_auto_cancelled_not_auto_error(self) -> None:
        pipeline_cls = self._cancel_aware_pipeline_cls()
        manager = IngestionManager()
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline", pipeline_cls):
            manager.start("default-repo", kind="auto")
            manager.cancel()
            for _ in range(50):
                if manager.last_completed() is not None:
                    break
                time.sleep(0.05)

        assert manager.auto_job_cancelled() is True
        assert manager.auto_job_error() is None
        manager.acknowledge()
        assert manager.auto_job_cancelled() is True
