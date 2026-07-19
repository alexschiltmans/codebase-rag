"""Unit tests for IngestPipeline.process_repo_incremental (content-hash diffing)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_rag.data_ingestion.pipeline import IngestPipeline


def _make_pipeline(
    mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmpdir: str
) -> IngestPipeline:
    mock_config = MagicMock()
    mock_config.qdrant_host = "localhost"
    mock_config.qdrant_port = 6333
    mock_config.collection_name = "docs"
    mock_config.repo_local_path = Path("/tmp/repos")
    mock_config_cls.get_instance.return_value = mock_config
    mock_logging.return_value = MagicMock()

    pipeline = IngestPipeline()
    pipeline.cache_dir = Path(tmpdir)
    return pipeline


class TestProcessRepoIncremental:
    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_first_ingest_embeds_every_file(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("print('a')")
        (tmp_path / "src" / "b.py").write_text("print('b')")

        with tempfile.TemporaryDirectory() as cache_dir:
            pipeline = _make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
            pipeline._explicit_included_dirs = ["src"]

            result = pipeline.process_repo_incremental(str(tmp_path))

            assert result.files_changed == 2
            assert result.files_deleted == 0
            assert result.chunks_indexed == 2
            pipeline.vector_store.add_documents.assert_called_once()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_reingest_with_one_changed_file_only_reembeds_it(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        a_path = tmp_path / "src" / "a.py"
        b_path = tmp_path / "src" / "b.py"
        a_path.write_text("print('a')")
        b_path.write_text("print('b')")

        with tempfile.TemporaryDirectory() as cache_dir:
            pipeline = _make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
            pipeline._explicit_included_dirs = ["src"]
            pipeline.process_repo_incremental(str(tmp_path))
            pipeline.vector_store.reset_mock()

            a_path.write_text("print('a changed')")
            result = pipeline.process_repo_incremental(str(tmp_path))

            assert result.files_changed == 1
            assert result.files_unchanged == 1
            assert result.files_deleted == 0

            deleted_sources = [call.args[0] for call in pipeline.vector_store.delete_by_source.call_args_list]
            assert deleted_sources == [str(a_path)]

            added_docs = pipeline.vector_store.add_documents.call_args[0][0]
            assert all(doc.metadata["source"] == str(a_path) for doc in added_docs)

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_reingest_with_no_changes_reembeds_nothing(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("print('a')")

        with tempfile.TemporaryDirectory() as cache_dir:
            pipeline = _make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
            pipeline._explicit_included_dirs = ["src"]
            pipeline.process_repo_incremental(str(tmp_path))
            pipeline.vector_store.reset_mock()

            result = pipeline.process_repo_incremental(str(tmp_path))

            assert result.files_changed == 0
            assert result.files_unchanged == 1
            assert result.chunks_indexed == 0
            pipeline.vector_store.add_documents.assert_not_called()
            pipeline.vector_store.delete_by_source.assert_not_called()

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_deleted_file_removes_its_chunks(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        a_path = tmp_path / "src" / "a.py"
        b_path = tmp_path / "src" / "b.py"
        a_path.write_text("print('a')")
        b_path.write_text("print('b')")

        with tempfile.TemporaryDirectory() as cache_dir:
            pipeline = _make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
            pipeline._explicit_included_dirs = ["src"]
            pipeline.process_repo_incremental(str(tmp_path))
            pipeline.vector_store.reset_mock()

            b_path.unlink()
            result = pipeline.process_repo_incremental(str(tmp_path))

            assert result.files_deleted == 1
            deleted_sources = [call.args[0] for call in pipeline.vector_store.delete_by_source.call_args_list]
            assert str(b_path) in deleted_sources

    @patch("codebase_rag.data_ingestion.pipeline.QdrantStore")
    @patch("codebase_rag.data_ingestion.pipeline.setup_logging")
    @patch("codebase_rag.data_ingestion.pipeline.Config")
    def test_saves_freshness_metadata(
        self, mock_config_cls: MagicMock, mock_logging: MagicMock, mock_qdrant_cls: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("print('a')")

        with tempfile.TemporaryDirectory() as cache_dir:
            pipeline = _make_pipeline(mock_config_cls, mock_logging, mock_qdrant_cls, cache_dir)
            pipeline._explicit_included_dirs = ["src"]

            result = pipeline.process_repo_incremental(str(tmp_path))

            freshness_path = Path(cache_dir) / f"{result.repo_name}_freshness.json"
            assert freshness_path.exists()
            data = json.loads(freshness_path.read_text())
            assert "last_ingest_time" in data
