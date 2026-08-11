"""End-to-end tests for the data ingestion pipeline."""

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest

# Ensure scripts/ is importable for the CLI main() function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from codebase_rag.data_ingestion.pipeline import IngestPipeline
from scripts.ingest import main


@pytest.fixture
def temp_data_dir() -> Iterator[Path]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(temp_dir)

    data_dir = Path(temp_dir) / "data"
    cache_dir = data_dir / "cache"
    logs_dir = Path(temp_dir) / "logs"

    for directory in [data_dir, cache_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    yield Path(temp_dir)

    os.chdir(old_cwd)
    shutil.rmtree(temp_dir)


@pytest.mark.e2e
@patch("codebase_rag.data_ingestion.git_loader.git.Repo.clone_from")
@patch("codebase_rag.database.qdrant_store.QdrantStore.add_documents")
@patch("codebase_rag.database.qdrant_store.QdrantStore.delete_by_repo")
@patch("codebase_rag.retrieval.hybrid_search.HybridRetriever.search")
def test_ingest_pipeline(
    mock_hybrid_search: MagicMock,
    mock_delete_by_repo: MagicMock,
    mock_add_documents: MagicMock,
    mock_clone_from: MagicMock,
    temp_data_dir: Path,
) -> None:
    """Test the complete IngestPipeline runs its real process/index/bm25 steps."""
    repo_url = "https://github.com/test/test-repo"
    repo_name = "test-repo"

    repo_dir = temp_data_dir / "data" / "repos" / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    readme = repo_dir / "README.md"
    readme.write_text("# Codebase\nA code analysis tool.")

    repo = git.Repo.init(repo_dir)
    repo.index.add(["README.md"])
    repo.index.commit("initial commit")

    mock_hybrid_search.return_value = []

    pipeline = IngestPipeline(
        repo_urls=[repo_url],
        included_dirs=[],
        included_files=["README.md"],
        use_cache=False,
        debug=True,
    )

    pipeline.run()

    stats_path = temp_data_dir / "data" / "cache" / "ingest_stats.json"
    assert stats_path.exists()

    with open(stats_path) as f:
        stats = json.load(f)

    assert stats["chunks_created"] > 0
    assert stats["chunks_indexed"] > 0
    assert stats["elapsed_time"] > 0

    assert mock_add_documents.call_count > 0


@pytest.mark.e2e
@patch("argparse.ArgumentParser.parse_args")
@patch("scripts.ingest.IngestPipeline")
def test_main_script(mock_pipeline_class: MagicMock, mock_parse_args: MagicMock, temp_data_dir: Path) -> None:
    """Test the main function of the ingest script."""
    args = mock_parse_args.return_value
    args.dirs = ["docs"]
    args.files = ["README.md"]
    args.drop = True
    args.force = False
    args.no_cache = True
    args.debug = True
    args.repos = ["https://github.com/test/repo"]
    args.all_repos = False

    pipeline_instance = mock_pipeline_class.return_value

    main()

    mock_pipeline_class.assert_called_once_with(
        included_dirs=["docs"],
        included_files=["README.md"],
        drop_existing=True,
        use_cache=False,
        debug=True,
        repo_urls=["https://github.com/test/repo"],
    )

    pipeline_instance.run.assert_called_once()
