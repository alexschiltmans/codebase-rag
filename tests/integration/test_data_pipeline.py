"""Integration tests for the data ingestion pipeline."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest

from codebase_rag.data_ingestion.document_processor import DocumentProcessor
from codebase_rag.data_ingestion.git_loader import GitLoader
from codebase_rag.database.qdrant_store import QdrantStore


@pytest.fixture
def temp_repo_dir():
    """Create a temporary directory for test repository."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_repo(temp_repo_dir):
    """Create a sample repository structure for testing."""
    docs_dir = temp_repo_dir / "docs"
    src_dir = temp_repo_dir / "src" / "mypackage"
    tests_dir = temp_repo_dir / "tests"

    for directory in [docs_dir, src_dir, tests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    readme = temp_repo_dir / "README.md"
    readme.write_text("""
# Sample Project
This is a sample README for testing.
## Installation
Install with pip.
    """)

    pyproject = temp_repo_dir / "pyproject.toml"
    pyproject.write_text("""
[project]
name = "mypackage"
version = "0.1.0"
description = "A sample project"
    """)

    docs_file = docs_dir / "getting_started.md"
    docs_file.write_text("""
# Getting Started
This guide helps you get started with codebase.
## Quick Start
```python
import mypackage
grid = mypackage.load_network()
```
    """)

    code_file = src_dir / "core.py"
    code_file.write_text("""
def load_network():
    \"\"\"Load a power grid network.

    Returns:
        Network: A power grid network object.
    \"\"\"
    return Network()

class Network:
    \"\"\"Represents a power grid network.\"\"\"

    def __init__(self):
        self.components = []
    """)

    test_file = tests_dir / "test_core.py"
    test_file.write_text("""
import pytest
from mypackage import load_network

def test_load_network():
    network = load_network()
    assert network is not None
    """)

    # Initialize a proper Git repository

    repo = git.Repo.init(temp_repo_dir)

    repo.git.add(".")

    repo.git.config("user.email", "test@example.com")
    repo.git.config("user.name", "Test User")
    repo.git.commit("-m", "Initial commit")

    return temp_repo_dir


@pytest.mark.integration
def test_git_loader_get_file_paths(sample_repo) -> None:
    """Test GitLoader file path retrieval."""
    loader = GitLoader(local_path=sample_repo)

    file_paths = loader.get_file_paths()

    expected_files = [
        "README.md",
        "pyproject.toml",
        "docs/getting_started.md",
        "src/mypackage/core.py",
        "tests/test_core.py",
    ]

    # Convert all paths to strings for easier comparison
    file_paths_str = [str(path.relative_to(sample_repo)) for path in file_paths]

    for expected in expected_files:
        assert expected in file_paths_str, f"Expected file {expected} not found"

    assert len(file_paths) == len(expected_files)


@pytest.mark.integration
def test_document_processor_end_to_end(sample_repo) -> None:
    """Test DocumentProcessor end-to-end processing."""
    git_loader = GitLoader(local_path=sample_repo)
    processor = DocumentProcessor(git_loader=git_loader)

    documents = processor.process()

    assert len(documents) > 0

    for doc in documents:
        assert "source" in doc.metadata
        assert "file_name" in doc.metadata
        assert "file_type" in doc.metadata
        assert "chunk_index" in doc.metadata
        assert "chunk_count" in doc.metadata

    file_types = {doc.metadata["file_type"] for doc in documents}
    assert ".md" in file_types
    assert ".py" in file_types
    assert ".toml" in file_types


@pytest.mark.integration
@patch("codebase_rag.database.qdrant_store.QdrantClient")
def test_qdrant_store_add_documents(mock_qdrant_client, sample_repo) -> None:
    """Test QdrantStore document addition."""
    mock_client_instance = MagicMock()
    mock_qdrant_client.return_value = mock_client_instance
    mock_client_instance.collection_exists.return_value = False

    git_loader = GitLoader(local_path=sample_repo)
    processor = DocumentProcessor(git_loader=git_loader)
    documents = processor.process()

    store = QdrantStore()
    store.add_documents(documents)

    mock_client_instance.upsert.assert_called()
