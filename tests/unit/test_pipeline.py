"""Unit tests for data_ingestion/pipeline.py."""

import hashlib
import json
import logging
import pickle
import sys
import tempfile
import threading
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from codebase_rag.data_ingestion.pipeline import (
    IngestCancelled,
    IngestPipeline,
    _teardown_logging,
    count_ingestible_files,
    discover_included_dirs,
    display_progress,
    load_documents_cache,
    save_documents_cache,
    setup_logging,
)
from codebase_rag.retrieval.bm25_search import BM25Retriever
from tests.conftest import stub_embedding_manager


class TestDiscoverIncludedDirs:
    """Tests for discover_included_dirs."""

    def test_skips_default_excluded_dirs(self, tmp_path: Path) -> None:
        for name in ["src", "node_modules", "venv", "dist", "docs"]:
            (tmp_path / name).mkdir()

        result = discover_included_dirs(tmp_path, fallback=["docs", "src", "tests"])

        assert set(result) == {"src", "docs"}

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / ".git").mkdir()

        result = discover_included_dirs(tmp_path, fallback=[])

        assert result == ["src"]

    def test_returns_fallback_when_path_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"

        result = discover_included_dirs(missing, fallback=["docs", "src", "tests"])

        assert result == ["docs", "src", "tests"]


class TestCountIngestibleFiles:
    """Tests for count_ingestible_files."""

    def test_counts_files_excluding_denylisted_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("module.exports = {}")

        included_dirs, count = count_ingestible_files(tmp_path)

        assert "node_modules" not in included_dirs
        assert count == 1

    def test_zero_files_for_empty_folder(self, tmp_path: Path) -> None:
        included_dirs, count = count_ingestible_files(tmp_path)

        assert count == 0


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_valid_log_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger, log_file = setup_logging("DEBUG")
        assert logger.name == "codebase_rag"
        assert log_file.exists()
        assert "ingest-" in log_file.name

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("BANANA")

    def test_file_created_with_root_handlers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.StreamHandler())

        logger, log_file = setup_logging("INFO", add_console=False)

        assert log_file.exists()
        assert (tmp_path / "logs").exists()

    def test_calling_twice_leaves_one_handler(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import logging as log_module
        import time as time_module

        monkeypatch.chdir(tmp_path)
        log_module.getLogger("codebase_rag").handlers.clear()

        logger1, log_file1 = setup_logging("INFO")
        logger1.info("First run message")
        time_module.sleep(0.01)
        logger2, log_file2 = setup_logging("INFO")
        logger2.info("Second run message")

        ingest_handlers = [h for h in logger2.handlers if h.name == "codebase_rag.ingest_file"]
        assert len(ingest_handlers) == 1
        assert log_file1.exists()
        assert log_file2.exists()
        assert log_file1 != log_file2

        content1 = log_file1.read_text()
        content2 = log_file2.read_text()
        assert "First run message" in content1
        assert "Second run message" in content2
        assert "First run message" not in content2

    def test_console_handler_explicit_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger = logging.getLogger("codebase_rag")
        logger.handlers.clear()

        log_logger, log_file = setup_logging("INFO", add_console=True)

        console_handlers = [
            h
            for h in log_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 1
        assert console_handlers[0].name == "codebase_rag.ingest_console"

    def test_no_console_handler_explicit_false(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger = logging.getLogger("codebase_rag")
        logger.handlers.clear()

        log_logger, log_file = setup_logging("INFO", add_console=False)

        console_handlers = [
            h
            for h in log_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 0

    def test_default_no_console_when_root_has_handlers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The `add_console=None` default probes the root logger: app case (handlers present)."""
        monkeypatch.chdir(tmp_path)
        logger = logging.getLogger("codebase_rag")
        logger.handlers.clear()
        root_logger = logging.getLogger()
        root_logger.addHandler(logging.StreamHandler())

        log_logger, log_file = setup_logging("INFO")

        console_handlers = [
            h
            for h in log_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 0

    def test_default_adds_console_when_root_has_no_handlers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `add_console=None` default probes the root logger: CLI case (root cleared)."""
        monkeypatch.chdir(tmp_path)
        logger = logging.getLogger("codebase_rag")
        logger.handlers.clear()
        root_logger = logging.getLogger()
        saved_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()
        try:
            log_logger, log_file = setup_logging("INFO")

            console_handlers = [
                h
                for h in log_logger.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            assert len(console_handlers) == 1
            assert console_handlers[0].name == "codebase_rag.ingest_console"
        finally:
            root_logger.handlers[:] = saved_handlers

    def test_teardown_detaches_handlers_and_stops_capturing_other_loggers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After teardown, records from unrelated package loggers no longer land in the run's file."""
        monkeypatch.chdir(tmp_path)
        codebase_rag_logger = logging.getLogger("codebase_rag")
        codebase_rag_logger.handlers.clear()

        logger, log_file = setup_logging("DEBUG")
        logger.info("run record")
        _teardown_logging(logger)

        other_logger = logging.getLogger("codebase_rag.app.ui_chat")
        other_logger.warning("unrelated app log after ingest")

        content = log_file.read_text()
        assert "run record" in content
        assert "unrelated app log after ingest" not in content
        assert not any(h.name in ("codebase_rag.ingest_file", "codebase_rag.ingest_console") for h in logger.handlers)

    def test_teardown_restores_prior_logger_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_logging must not leave its DEBUG level bleeding into all app logging after the run ends."""
        monkeypatch.chdir(tmp_path)
        codebase_rag_logger = logging.getLogger("codebase_rag")
        codebase_rag_logger.handlers.clear()
        codebase_rag_logger.setLevel(logging.NOTSET)

        logger, _ = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

        _teardown_logging(logger)

        assert logger.level == logging.NOTSET


class TestDocumentCache:
    """Tests for save/load document cache."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "sub" / "cache.pkl"
        docs = [
            Document(page_content="doc1", metadata={"source": "a.py"}),
            Document(page_content="doc2", metadata={"source": "b.py"}),
        ]

        save_documents_cache(docs, cache_path)
        loaded = load_documents_cache(cache_path)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].page_content == "doc1"

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_documents_cache(tmp_path / "nonexistent.pkl")
        assert result is None


class TestDisplayProgress:
    """Tests for display_progress."""

    def test_progress_bar_output(self) -> None:
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            display_progress(5, 10, prefix="Test: ")
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "50.0%" in output
        assert "Test: " in output

    def test_progress_bar_complete(self) -> None:
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            display_progress(10, 10, prefix="Done: ")
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "100.0%" in output


class TestIngestPipeline:
    """Tests for IngestPipeline."""

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_repo_name_from_url(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline(repo_url="https://github.com/owner/my-repo.git")

        assert pipeline._repo_name_from_url("https://github.com/owner/my-repo.git") == "my-repo"
        assert pipeline._repo_name_from_url("https://github.com/owner/my-repo/") == "my-repo"
        assert pipeline._repo_name_from_url("https://github.com/owner/my-repo") == "my-repo"

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_cache_path_for_repo(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config.cache_dir = Path("/tmp/cache")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline()
        path = pipeline._cache_path_for_repo("my-repo")
        assert "processed_documents_my-repo.pkl" in str(path)

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_index_documents(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        mock_store = MagicMock()
        stub_embedding_manager(mock_store)
        mock_qdrant_cls.return_value = mock_store

        pipeline = IngestPipeline()

        docs = [
            Document(page_content="content1", metadata={"source": "a.py", "chunk_index": 0, "repo": "my-repo"}),
            Document(page_content="content2", metadata={"source": "b.py", "chunk_index": 0, "repo": "my-repo"}),
        ]

        pipeline.index_documents(docs)

        mock_store.delete_by_repo.assert_called_once_with("my-repo")
        mock_store.add_documents.assert_called()
        assert pipeline.stats["chunks_indexed"] == 2

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_index_documents_reports_monotonic_progress(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        calls: list[tuple[str, int, int]] = []
        pipeline = IngestPipeline(progress_callback=lambda phase, current, total: calls.append((phase, current, total)))
        stub_embedding_manager(cast(MagicMock, pipeline.vector_store))

        docs = [
            Document(page_content=f"content{i}", metadata={"source": f"{i}.py", "repo": "my-repo"}) for i in range(250)
        ]

        pipeline.index_documents(docs)

        assert calls == [("indexing", 1, 3), ("indexing", 2, 3), ("indexing", 3, 3)]

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_index_documents_raises_ingest_cancelled_when_event_set(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = IngestPipeline(cancel_event=cancel_event)
        docs = [Document(page_content="content", metadata={"source": "a.py", "repo": "my-repo"})]

        with pytest.raises(IngestCancelled):
            pipeline.index_documents(docs)

        cast(MagicMock, pipeline.vector_store.add_documents).assert_not_called()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_index_documents_cancelled_before_delete_leaves_existing_chunks_untouched(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        """A cancel that lands right as indexing starts must not run
        delete_by_repo — otherwise a re-ingest's existing chunks are wiped
        with nothing indexed to replace them."""
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = IngestPipeline(cancel_event=cancel_event)
        docs = [Document(page_content="content", metadata={"source": "a.py", "repo": "my-repo"})]

        with pytest.raises(IngestCancelled):
            pipeline.index_documents(docs)

        cast(MagicMock, pipeline.vector_store.delete_by_repo).assert_not_called()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_index_documents_stops_at_next_batch_boundary_once_cancelled(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        cancel_event = threading.Event()

        def _maybe_cancel(phase: str, current: int, total: int) -> None:
            if current == 1:
                cancel_event.set()

        pipeline = IngestPipeline(progress_callback=_maybe_cancel, cancel_event=cancel_event)
        stub_embedding_manager(cast(MagicMock, pipeline.vector_store))
        docs = [
            Document(page_content=f"content{i}", metadata={"source": f"{i}.py", "repo": "my-repo"}) for i in range(250)
        ]

        with pytest.raises(IngestCancelled):
            pipeline.index_documents(docs)

        assert cast(MagicMock, pipeline.vector_store.add_documents).call_count == 1

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_save_bm25_index(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))
        mock_qdrant_cls.return_value.embedding_manager.max_seq_length = 384

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            docs = [
                Document(page_content="hello world", metadata={"source": "a.py"}),
                Document(page_content="from test import something", metadata={"source": "b.py"}),
            ]

            pipeline.save_bm25_index(docs)

            bm25_path = Path(tmpdir) / "bm25_retriever.json"
            assert bm25_path.exists()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_save_bm25_index_records_the_chunking_it_wrote(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        """The application's own corpus is the one every default benchmark invocation scores
        against. Left unrecorded it is the one corpus nothing can name a chunk size for, and two
        runs either side of a re-ingest collide on the same collection name."""
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))
        mock_qdrant_cls.return_value.embedding_manager.max_seq_length = 384

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            pipeline.save_bm25_index([Document(page_content="hello world", metadata={"source": "a.py"})])

            sidecar = json.loads((Path(tmpdir) / "bm25_corpus" / "_meta" / "chunking.json").read_text())
            assert sidecar == {"chunk_size": 614, "chunk_overlap": 122, "chunk_max_seq_length": 384}

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_save_stats(self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)
            pipeline.stats = {"chunks_created": 10, "chunks_indexed": 10}

            pipeline.save_stats()

            stats_path = Path(tmpdir) / "ingest_stats.json"
            assert stats_path.exists()
            with open(stats_path) as f:
                loaded = json.load(f)
            assert loaded["chunks_created"] == 10

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_process_documents_no_urls_raises(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline()

        with pytest.raises(ValueError, match="No repository URLs provided"):
            pipeline.process_documents()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_process_documents_with_urls(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline(repo_urls=["https://github.com/test/repo1"])

        with patch.object(pipeline, "_process_single_repo") as mock_process:
            mock_process.return_value = [Document(page_content="code", metadata={"source": "a.py", "repo": "repo1"})]
            result = pipeline.process_documents()

        assert len(result) == 1
        assert pipeline.stats["chunks_created"] == 1

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_run_orchestrates_pipeline(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            docs = [Document(page_content="test", metadata={"source": "file.py"})]

            with (
                patch.object(pipeline, "process_documents", return_value=docs),
                patch.object(pipeline, "index_documents") as mock_index,
                patch.object(pipeline, "save_bm25_index") as mock_bm25,
                patch.object(pipeline, "verify_hybrid_search") as mock_verify,
            ):
                pipeline.run()

            mock_index.assert_called_once_with(docs)
            mock_bm25.assert_called_once_with(docs)
            mock_verify.assert_called_once()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_run_raises_on_error(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline()

        with (
            patch.object(pipeline, "process_documents", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            pipeline.run()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_run_detaches_ingest_handlers_on_success(
        self, mock_config_cls: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must tear down the ingest handlers itself; a real (unmocked) logger proves it."""
        monkeypatch.chdir(tmp_path)
        logging.getLogger("codebase_rag").handlers.clear()

        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config

        pipeline = IngestPipeline()
        docs = [Document(page_content="test", metadata={"source": "file.py"})]

        with (
            patch.object(pipeline, "process_documents", return_value=docs),
            patch.object(pipeline, "index_documents"),
            patch.object(pipeline, "save_bm25_index"),
            patch.object(pipeline, "verify_hybrid_search"),
        ):
            pipeline.run()

        assert not any(
            h.name in ("codebase_rag.ingest_file", "codebase_rag.ingest_console") for h in pipeline.logger.handlers
        )

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_run_detaches_ingest_handlers_on_failure(
        self, mock_config_cls: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The teardown must run even when the pipeline raises, not just on the success path."""
        monkeypatch.chdir(tmp_path)
        logging.getLogger("codebase_rag").handlers.clear()

        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config

        pipeline = IngestPipeline()

        with (
            patch.object(pipeline, "process_documents", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            pipeline.run()

        assert not any(
            h.name in ("codebase_rag.ingest_file", "codebase_rag.ingest_console") for h in pipeline.logger.handlers
        )

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_init_detaches_ingest_handlers_when_construction_fails(
        self, mock_config_cls: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A constructor failure after setup_logging (e.g. QdrantStore unreachable) must not
        leave the handler attached, since run()'s finally block is never reached in that case.
        """
        monkeypatch.chdir(tmp_path)
        logging.getLogger("codebase_rag").handlers.clear()

        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_qdrant_cls.side_effect = RuntimeError("qdrant unreachable")

        with pytest.raises(RuntimeError, match="qdrant unreachable"):
            IngestPipeline()

        logger = logging.getLogger("codebase_rag")
        assert not any(h.name in ("codebase_rag.ingest_file", "codebase_rag.ingest_console") for h in logger.handlers)
        assert logger.level == logging.NOTSET

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_verify_hybrid_search_success(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            # Create a mock BM25 cache

            bm25 = BM25Retriever([Document(page_content="test doc", metadata={"source": "x.py"})])
            bm25_path = Path(tmpdir) / "bm25_retriever.json"
            bm25.save_json(bm25_path)

            with patch("codebase_rag.data_ingestion.pipeline.VectorRetriever") as mock_vr_cls:
                mock_vr = MagicMock()
                mock_vr_cls.return_value = mock_vr

                with patch("codebase_rag.data_ingestion.pipeline.HybridRetriever") as mock_hr_cls:
                    mock_hr = MagicMock()
                    mock_hr.search.return_value = [(Document(page_content="result", metadata={"source": "a.py"}), 0.85)]
                    mock_hr_cls.return_value = mock_hr

                    pipeline.verify_hybrid_search("test query")

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_verify_hybrid_search_no_results(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            bm25 = BM25Retriever([Document(page_content="test", metadata={"source": "x.py"})])
            bm25_path = Path(tmpdir) / "bm25_retriever.json"
            bm25.save_json(bm25_path)

            with (
                patch("codebase_rag.data_ingestion.pipeline.VectorRetriever"),
                patch("codebase_rag.data_ingestion.pipeline.HybridRetriever") as mock_hr_cls,
            ):
                mock_hr = MagicMock()
                mock_hr.search.return_value = []
                mock_hr_cls.return_value = mock_hr

                # Should not raise
                pipeline.verify_hybrid_search()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_verify_hybrid_search_error(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline()
            pipeline.cache_dir = Path(tmpdir)

            # No bm25 file, so it should hit the except branch
            pipeline.verify_hybrid_search()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_init_with_repo_urls(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline(repo_urls=["https://github.com/a/b", "https://github.com/c/d"])
        assert len(pipeline._repo_urls) == 2

    @patch("codebase_rag.data_ingestion.pipeline.DocumentProcessor")
    @patch("codebase_rag.data_ingestion.pipeline.GitLoader")
    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_process_single_repo_from_cache(
        self,
        mock_config_cls: MagicMock,
        mock_logging: MagicMock,
        mock_qdrant_cls: MagicMock,
        mock_git_loader_cls: MagicMock,
        mock_doc_proc_cls: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        # Set up mock git loader to return a repo with a known HEAD SHA
        mock_git_loader = MagicMock()
        mock_git_loader.repo.head.commit.hexsha = "abc123"
        mock_git_loader_cls.return_value = mock_git_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline(repo_url="https://github.com/test/myrepo")
            pipeline.cache_dir = Path(tmpdir)
            stub_embedding_manager(cast(MagicMock, pipeline.vector_store))

            # Create cached docs
            cached_docs = [
                Document(page_content="cached", metadata={"source": "a.py"}),
            ]
            cache_path = Path(tmpdir) / "processed_documents_myrepo.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(cached_docs, f)

            # Create matching cache metadata so cache is considered fresh
            meta_path = Path(tmpdir) / "myrepo_cache_meta.json"
            with open(meta_path, "w") as f:
                json.dump({"commit_sha": "abc123", "timestamp": 0}, f)

            # The cache also has to have been chunked the way this run chunks.
            pipeline._cache_chunking_path_for_repo("myrepo").write_text(
                json.dumps({"chunking": pipeline._chunking_fingerprint(pipeline._build_chunker())})
            )

            result = pipeline._process_single_repo("https://github.com/test/myrepo")

        assert len(result) == 1
        assert result[0].metadata.get("repo") == "myrepo"
        # DocumentProcessor should NOT have been called — cache was fresh
        mock_doc_proc_cls.assert_not_called()

    @patch("codebase_rag.data_ingestion.pipeline.DocumentProcessor")
    @patch("codebase_rag.data_ingestion.pipeline.GitLoader")
    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_process_single_repo_rejects_cache_from_an_older_commit(
        self,
        mock_config_cls: MagicMock,
        mock_logging: MagicMock,
        mock_qdrant_cls: MagicMock,
        mock_git_loader_cls: MagicMock,
        mock_doc_proc_cls: MagicMock,
    ) -> None:
        """A second full `run()` after further upstream commits must not treat
        an existing pickle as fresh just because it exists on disk: the
        recorded HEAD SHA has to match too, or a stale document cache would
        be re-indexed as if it were current."""
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        # HEAD has advanced past the commit the cache was built from.
        mock_git_loader = MagicMock()
        mock_git_loader.repo.head.commit.hexsha = "def456"
        mock_git_loader_cls.return_value = mock_git_loader
        mock_doc_proc_cls.return_value.process.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline(repo_url="https://github.com/test/myrepo")
            pipeline.cache_dir = Path(tmpdir)
            stub_embedding_manager(cast(MagicMock, pipeline.vector_store))

            cache_path = Path(tmpdir) / "processed_documents_myrepo.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump([Document(page_content="stale", metadata={"source": "a.py"})], f)

            freshness_path = Path(tmpdir) / "myrepo_freshness.json"
            with open(freshness_path, "w") as f:
                json.dump({"last_ingest_time": 0, "head_sha": "abc123"}, f)

            pipeline._process_single_repo("https://github.com/test/myrepo")

        # The stale cache didn't match HEAD, so the pipeline had to reprocess.
        mock_doc_proc_cls.return_value.process.assert_called_once()


class TestTruncationReporting:
    """Tests for the truncation report and the model-sized chunker."""

    @staticmethod
    def _make_pipeline(
        mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, max_seq_length: int = 384
    ) -> tuple[IngestPipeline, MagicMock]:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        logger = MagicMock()
        mock_logging.return_value = (logger, Path("/tmp/ingest.log"))

        mock_store = MagicMock()
        stub_embedding_manager(mock_store, max_seq_length=max_seq_length)
        mock_qdrant_cls.return_value = mock_store

        return IngestPipeline(), logger

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_chunker_is_sized_from_the_configured_model(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        small, _ = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, max_seq_length=384)
        large, _ = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, max_seq_length=2048)

        assert large._build_chunker().chunk_size > small._build_chunker().chunk_size

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_indexing_reports_over_length_chunks_per_file_type(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        pipeline, logger = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls)

        # The stubbed tokenizer counts one token per four characters.
        pipeline.index_documents(
            [
                Document(page_content="x" * 4000, metadata={"source": "a.json", "file_type": ".json", "repo": "r"}),
                Document(page_content="short", metadata={"source": "b.py", "file_type": ".py", "repo": "r"}),
            ]
        )

        logged = " ".join(str(call.args[0]) for call in logger.info.call_args_list)
        assert "1 of 2 chunks" in logged
        assert "384-token" in logged
        assert ".json: 1/1" in logged

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_a_clean_run_still_reports(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        pipeline, logger = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls)

        pipeline.index_documents(
            [Document(page_content="short", metadata={"source": "b.py", "file_type": ".py", "repo": "r"})]
        )

        logged = " ".join(str(call.args[0]) for call in logger.info.call_args_list)
        assert "nothing was truncated" in logged


class TestChunkingInvalidatesCaches:
    """Tests that a chunk-size or model change is not hidden by a content-keyed cache."""

    @staticmethod
    def _make_pipeline(
        mock_config_cls: MagicMock,
        mock_logging: MagicMock,
        mock_qdrant_cls: MagicMock,
        cache_dir: Path,
        max_seq_length: int = 384,
    ) -> IngestPipeline:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = (MagicMock(), Path("/tmp/ingest.log"))

        pipeline = IngestPipeline()
        pipeline.cache_dir = cache_dir
        store = cast(MagicMock, pipeline.vector_store)
        stub_embedding_manager(store, max_seq_length=max_seq_length)
        store.embedding_manager.model_name = "some/model"
        return pipeline

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_resized_chunks_re_chunk_every_file_not_just_changed_ones(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("print('a')")
        (tmp_path / "src" / "b.py").write_text("print('b')")
        cache_dir = tmp_path / "cache"

        first = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
        first._explicit_included_dirs = ["src"]
        first.process_repo_incremental(str(tmp_path))

        # Nothing on disk changed, only the model's window did.
        second = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir, max_seq_length=2048)
        second._explicit_included_dirs = ["src"]
        result = second.process_repo_incremental(str(tmp_path))

        assert result.files_changed == 2
        assert result.files_unchanged == 0

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_unchanged_chunking_still_skips_unchanged_files(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("print('a')")
        cache_dir = tmp_path / "cache"

        first = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
        first._explicit_included_dirs = ["src"]
        first.process_repo_incremental(str(tmp_path))

        second = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
        second._explicit_included_dirs = ["src"]
        result = second.process_repo_incremental(str(tmp_path))

        assert (result.files_changed, result.files_unchanged) == (0, 1)

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_a_manifest_without_a_recorded_chunking_is_not_trusted(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        source = tmp_path / "src" / "a.py"
        source.write_text("print('a')")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # The flat mapping written before chunking was recorded.
        legacy = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        (cache_dir / f"{tmp_path.name}_file_hashes.json").write_text(json.dumps(legacy))

        pipeline = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
        pipeline._explicit_included_dirs = ["src"]
        result = pipeline.process_repo_incremental(str(tmp_path))

        assert result.files_changed == 1

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_document_cache_is_stale_when_chunking_changed_at_the_same_commit(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pipeline = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)

        save_documents_cache(
            [Document(page_content="cached", metadata={"source": "a.py"})], pipeline._cache_path_for_repo("myrepo")
        )
        (cache_dir / "myrepo_freshness.json").write_text(json.dumps({"last_ingest_time": 0, "head_sha": "abc123"}))
        pipeline._cache_chunking_path_for_repo("myrepo").write_text(json.dumps({"chunking": "some/model|384|1000|200"}))

        assert pipeline._is_cache_fresh("myrepo", "abc123") is False

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_document_cache_is_fresh_when_commit_and_chunking_both_match(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pipeline = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)

        save_documents_cache(
            [Document(page_content="cached", metadata={"source": "a.py"})], pipeline._cache_path_for_repo("myrepo")
        )
        (cache_dir / "myrepo_freshness.json").write_text(json.dumps({"last_ingest_time": 0, "head_sha": "abc123"}))
        pipeline._cache_chunking_path_for_repo("myrepo").write_text(
            json.dumps({"chunking": pipeline._chunking_fingerprint(pipeline._build_chunker())})
        )

        assert pipeline._is_cache_fresh("myrepo", "abc123") is True

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_a_cache_with_no_recorded_chunking_is_not_reused(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        pipeline = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)

        save_documents_cache(
            [Document(page_content="cached", metadata={"source": "a.py"})], pipeline._cache_path_for_repo("myrepo")
        )
        (cache_dir / "myrepo_freshness.json").write_text(json.dumps({"last_ingest_time": 0, "head_sha": "abc123"}))

        assert pipeline._is_cache_fresh("myrepo", "abc123") is False

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_a_failing_truncation_check_does_not_stop_the_ingest(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        pipeline = self._make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, tmp_path)
        cast(MagicMock, pipeline.vector_store).embedding_manager.count_tokens.side_effect = RuntimeError("tokenizer")

        pipeline.index_documents([Document(page_content="x", metadata={"source": "a.py", "repo": "r"})])

        cast(MagicMock, pipeline.vector_store).add_documents.assert_called_once()
        warnings = cast(MagicMock, pipeline.logger).warning.call_args_list
        assert any("Truncation check failed" in str(call.args[0]) for call in warnings)
