"""End-to-end smoke tests for core application components.

These tests require running Qdrant (localhost:6333) and optionally Ollama.
"""

import os
import shutil
import tempfile

import pytest
import requests
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from codebase_rag.config import Config
from codebase_rag.database.chat_storage import ChatHistoryManager
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.database.sqlite_storage import SqliteChatStorage
from codebase_rag.retrieval.vector_search import VectorRetriever


@pytest.fixture
def qdrant_store():
    """Create a QdrantStore with a temporary test collection."""

    store = QdrantStore(
        host="localhost",
        port=6333,
        collection_name="test_e2e_smoke",
        recreate_collection=True,
    )
    yield store
    client = QdrantClient(host="localhost", port=6333)
    client.delete_collection("test_e2e_smoke")


@pytest.fixture
def sqlite_db_path():
    """Create a temporary SQLite database path and clean up after."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_chat.db")
    yield db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


SAMPLE_DOCS = [
    Document(
        page_content="Python is a great programming language for data science",
        metadata={"source": "test.md", "file_name": "test.md"},
    ),
    Document(
        page_content="Docker containers simplify deployment workflows",
        metadata={"source": "deploy.md", "file_name": "deploy.md"},
    ),
    Document(
        page_content="Qdrant is a vector database for similarity search",
        metadata={"source": "qdrant.md", "file_name": "qdrant.md"},
    ),
]


@pytest.mark.e2e
class TestQdrantStore:
    def test_add_and_search(self, qdrant_store: QdrantStore) -> None:
        qdrant_store.add_documents(SAMPLE_DOCS)
        assert qdrant_store.collection_exists()

        results = qdrant_store.similarity_search("vector database", k=2)
        assert len(results) == 2

    def test_similarity_search_with_score(self, qdrant_store: QdrantStore) -> None:
        qdrant_store.add_documents(SAMPLE_DOCS)

        results = qdrant_store.similarity_search_with_score("python programming", k=2)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(score, float)
            assert doc.page_content


@pytest.mark.e2e
class TestSqliteChatStorage:
    def test_save_and_retrieve(self, sqlite_db_path: str) -> None:
        storage = SqliteChatStorage(db_path=sqlite_db_path)
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        storage.save_chat("test-chat-1", messages)

        retrieved = storage.get_chat("test-chat-1")
        assert retrieved is not None
        assert len(retrieved) == 2

    def test_list_and_delete(self, sqlite_db_path: str) -> None:
        storage = SqliteChatStorage(db_path=sqlite_db_path)
        storage.save_chat("test-chat-1", [{"role": "user", "content": "Hello!"}])

        assert len(storage.list_chats()) == 1

        storage.delete_chat("test-chat-1")
        assert len(storage.list_chats()) == 0


@pytest.mark.e2e
class TestChatHistoryManager:
    def test_round_trip(self, sqlite_db_path: str) -> None:
        manager = ChatHistoryManager()
        manager.storage = SqliteChatStorage(db_path=sqlite_db_path)

        messages = [
            {"role": "user", "content": "How does Qdrant work?"},
            {"role": "assistant", "content": "Qdrant is a vector database..."},
        ]
        manager.save_chat_history("chat-1", messages)

        retrieved = manager.get_chat_history("chat-1")
        assert retrieved is not None
        assert len(retrieved) == 2

        assert len(manager.list_chat_histories()) >= 1

        manager.delete_chat_history("chat-1")
        assert len(manager.list_chat_histories()) == 0


@pytest.mark.e2e
class TestVectorRetriever:
    def test_search(self) -> None:

        store = QdrantStore(
            host="localhost",
            port=6333,
            collection_name="test_retriever_smoke",
            recreate_collection=True,
        )
        docs = [
            Document(
                page_content="The RAG chain retrieves relevant documents and generates answers",
                metadata={"source": "rag.md", "file_name": "rag.md", "chunk_index": 0},
            ),
            Document(
                page_content="Vector databases store embeddings for fast similarity search",
                metadata={"source": "vectors.md", "file_name": "vectors.md", "chunk_index": 0},
            ),
            Document(
                page_content="BM25 is a keyword-based retrieval algorithm",
                metadata={"source": "bm25.md", "file_name": "bm25.md", "chunk_index": 0},
            ),
        ]
        store.add_documents(docs)

        retriever = VectorRetriever(vector_store=store)
        results = retriever.search("how does RAG work", k=2)
        assert len(results) == 2
        for _doc, score in results:
            assert isinstance(score, float)

        client = QdrantClient(host="localhost", port=6333)
        client.delete_collection("test_retriever_smoke")


@pytest.mark.e2e
class TestOllamaConnection:
    def test_ollama_reachable(self) -> None:

        base_url = Config.get_instance().ollama_base_url
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            assert resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            pytest.skip("Ollama not reachable")


@pytest.mark.e2e
class TestLangfuseConnection:
    def test_langfuse_reachable(self) -> None:

        try:
            resp = requests.get("http://localhost:3000", timeout=5)
            assert resp.status_code == 200
        except requests.ConnectionError:
            pytest.skip("Langfuse not reachable")
