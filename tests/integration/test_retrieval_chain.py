"""Integration tests for the retrieval and LLM chain."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever


@pytest.fixture
def test_documents():
    """Create test documents for retrieval."""
    return [
        Document(
            page_content="This is a Python package for code analysis. It provides tools for modeling and simulation.",
            metadata={"source": "intro.md", "file_name": "intro.md", "chunk_index": 0},
        ),
        Document(
            page_content="To install the package, use pip: `pip install mypackage`",
            metadata={"source": "install.md", "file_name": "install.md", "chunk_index": 0},
        ),
        Document(
            page_content="Load a network with `mypackage.load_network(path)` where path is the network data file.",
            metadata={"source": "api.py", "file_name": "api.py", "chunk_index": 0},
        ),
        Document(
            page_content="Calculate power flow with `network.run_power_flow()` after loading a network.",
            metadata={"source": "examples.py", "file_name": "examples.py", "chunk_index": 0},
        ),
        Document(
            page_content="Visualize results with `mypackage.plot(network)` to generate interactive diagrams.",
            metadata={"source": "visualization.py", "file_name": "visualization.py", "chunk_index": 0},
        ),
    ]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.integration
def test_bm25_retrieval(test_documents, temp_cache_dir) -> None:
    """Test BM25 retrieval with actual documents."""
    retriever = BM25Retriever(test_documents)

    results = retriever.search("how to install mypackage", k=1)

    assert len(results) == 1
    assert "install" in results[0][0].page_content.lower()

    # Save and load the retriever
    bm25_path = temp_cache_dir / "bm25_retriever.json"
    retriever.save_json(bm25_path)

    loaded_retriever = BM25Retriever.load_json(bm25_path)

    loaded_results = loaded_retriever.search("how to install mypackage", k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0][0].page_content == results[0][0].page_content


@pytest.mark.integration
@patch.object(QdrantStore, "similarity_search_with_score")
def test_hybrid_retrieval(mock_similarity_search, test_documents) -> None:
    """Test hybrid retrieval with BM25 and mocked vector search."""
    # Set up mock for vector search to return 3 results
    mock_similarity_search.return_value = [
        (test_documents[0], 0.9),  # "codebase is a Python package..."
        (test_documents[1], 0.85),  # Another document
        (test_documents[2], 0.8),  # "Load a network with..."
    ]

    bm25_retriever = BM25Retriever(test_documents)
    qdrant_store = QdrantStore()
    vector_retriever = VectorRetriever(vector_store=qdrant_store)

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    results = hybrid_retriever.search("load mypackage network", k=3)

    assert len(results) == 3

    # Extract document contents for easier assertion
    contents = [doc.page_content for doc, _ in results]

    # The document about loading a network should be ranked highly
    assert any("load_network" in content for content in contents)

    mock_similarity_search.assert_called_once_with("load mypackage network", 6)


@pytest.mark.integration
@patch.object(OllamaClient, "invoke")
def test_rag_chain_integration(mock_llm_invoke, test_documents) -> None:
    """Test the integration of retrieval and LLM in the RAG chain."""
    mock_llm_invoke.return_value = (
        "To install the package, you can use pip as shown in the documentation:\n"
        "```\npip install mypackage\n```\n\n"
        "After installation, you can load a network using the "
        "`mypackage.load_network(path)` function, where `path` is the location "
        "of your network data file.\n\nSources: [1] install.md, [2] api.py"
    )

    # Create BM25 retriever (using actual implementation)
    bm25_retriever = BM25Retriever(test_documents)

    # Create hybrid retriever with mock vector retriever
    vector_retriever = MagicMock()
    vector_retriever.search.return_value = [
        (test_documents[1], 0.9),  # Install document
        (test_documents[2], 0.7),  # API document
    ]

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    llm = OllamaClient()
    rag_chain = RAGChain(retriever=hybrid_retriever, llm=llm)

    result = rag_chain.run("How do I install and use this package?")

    # Verify LLM was called with appropriate context
    prompt = mock_llm_invoke.call_args[0][0]
    assert "install" in prompt.lower()
    assert "load_network" in prompt

    assert "pip install mypackage" in result["answer"]
    assert len(result["sources"]) > 0
    assert "install.md" in [source["file_name"] for source in result["sources"]]
