"""Unit tests for data_ingestion/pipeline.py."""

import json
import pickle
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from codebase_rag.data_ingestion.pipeline import (
    IngestPipeline,
    display_progress,
    load_documents_cache,
    save_documents_cache,
    setup_logging,
)
from codebase_rag.retrieval.bm25_search import BM25Retriever


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_valid_log_level(self, tmp_path: Path) -> None:
        with patch("codebase_rag.data_ingestion.pipeline.Path", return_value=tmp_path / "logs"):
            logger = setup_logging("DEBUG")
        assert logger.name == "codebase_rag.ingest"

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("BANANA")


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
        mock_logging.return_value = MagicMock()

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
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

        mock_store = MagicMock()
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
    def test_save_bm25_index(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock
    ) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = MagicMock()

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
    def test_save_stats(self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.qdrant_host = "localhost"
        mock_config.qdrant_port = 6333
        mock_config.collection_name = "docs"
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

        pipeline = IngestPipeline()

        with (
            patch.object(pipeline, "process_documents", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            pipeline.run()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

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
        mock_logging.return_value = MagicMock()

        # Set up mock git loader to return a repo with a known HEAD SHA
        mock_git_loader = MagicMock()
        mock_git_loader.repo.head.commit.hexsha = "abc123"
        mock_git_loader_cls.return_value = mock_git_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = IngestPipeline(repo_url="https://github.com/test/myrepo")
            pipeline.cache_dir = Path(tmpdir)

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

            result = pipeline._process_single_repo("https://github.com/test/myrepo")

        assert len(result) == 1
        assert result[0].metadata.get("repo") == "myrepo"
        # DocumentProcessor should NOT have been called — cache was fresh
        mock_doc_proc_cls.assert_not_called()
