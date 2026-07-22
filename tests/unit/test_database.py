"""Unit tests for database modules: qdrant_store, sqlite_storage, chat_storage."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import codebase_rag.database.chat_storage as mod
from codebase_rag.database.chat_storage import ChatHistoryManager
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.database.sqlite_storage import SqliteChatStorage


class TestSqliteChatStorage:
    """Tests for SqliteChatStorage."""

    def test_save_and_get_chat(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        storage.save_chat("chat-1", messages)

        result = storage.get_chat("chat-1")
        assert result is not None
        assert len(result) == 2
        assert result[0]["content"] == "Hello"

    def test_get_nonexistent_chat(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        result = storage.get_chat("nonexistent")
        assert result is None

    def test_list_chats(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        storage.save_chat("c1", [{"role": "user", "content": "First question"}])
        storage.save_chat("c2", [{"role": "user", "content": "Second question"}])

        chats = storage.list_chats()
        assert len(chats) == 2
        chat_ids = {c["chat_id"] for c in chats}
        assert "c1" in chat_ids
        assert "c2" in chat_ids

    def test_delete_chat(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        storage.save_chat("c1", [{"role": "user", "content": "test"}])
        assert storage.delete_chat("c1") is True
        assert storage.get_chat("c1") is None

    def test_delete_nonexistent_chat(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        assert storage.delete_chat("nonexistent") is False

    def test_save_chat_upsert(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        storage.save_chat("c1", [{"role": "user", "content": "first"}])
        storage.save_chat("c1", [{"role": "user", "content": "updated"}])

        result = storage.get_chat("c1")
        assert result is not None
        assert result[0]["content"] == "updated"

    def test_title_truncation(self, tmp_path: Path) -> None:
        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        long_content = "A" * 100
        storage.save_chat("c1", [{"role": "user", "content": long_content}])

        chats = storage.list_chats()
        assert len(chats) == 1
        assert chats[0]["title"].endswith("...")


class TestChatHistoryManager:
    """Tests for ChatHistoryManager."""

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_save_chat_history_success(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        result = mgr.save_chat_history("c1", [{"role": "user", "content": "hi"}])

        assert result is True
        mock_storage.save_chat.assert_called_once()

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_save_chat_history_failure(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.save_chat.side_effect = RuntimeError("db error")
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        result = mgr.save_chat_history("c1", [{"role": "user", "content": "hi"}])

        assert result is False

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_get_chat_history(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.get_chat.return_value = [{"role": "user", "content": "hello"}]
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        result = mgr.get_chat_history("c1")

        assert result == [{"role": "user", "content": "hello"}]

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_get_chat_history_no_storage(self, mock_storage_cls: MagicMock) -> None:

        mock_storage_cls.side_effect = RuntimeError("init failed")

        mgr = ChatHistoryManager()
        result = mgr.get_chat_history("c1")

        assert result is None

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_list_chat_histories(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.list_chats.return_value = [{"chat_id": "c1"}, {"chat_id": "c2"}]
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        result = mgr.list_chat_histories()

        assert len(result) == 2

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_list_chat_histories_no_storage(self, mock_storage_cls: MagicMock) -> None:

        mock_storage_cls.side_effect = RuntimeError("init failed")

        mgr = ChatHistoryManager()
        assert mgr.list_chat_histories() == []

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_delete_chat_history(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.delete_chat.return_value = True
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        assert mgr.delete_chat_history("c1") is True

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_delete_chat_history_no_storage(self, mock_storage_cls: MagicMock) -> None:

        mock_storage_cls.side_effect = RuntimeError("init failed")

        mgr = ChatHistoryManager()
        assert mgr.delete_chat_history("c1") is False


class TestQdrantStore:
    """Tests for QdrantStore (mocked client)."""

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_embedding_model_passed_through(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:
        QdrantStore(embedding_model="sentence-transformers/other-model")
        mock_emb.assert_called_once_with(model_name="sentence-transformers/other-model")

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_embedding_model_defaults_to_none(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:
        QdrantStore()
        mock_emb.assert_called_once_with(model_name=None)

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_collection_exists(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.collection_exists() is True

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_collection_not_exists(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.collection_exists() is False

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_collection_exists_error(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.side_effect = RuntimeError("connection error")
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.collection_exists() is False

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_add_documents_empty(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        store.add_documents([])

        mock_client.upsert.assert_not_called()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_add_documents(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        docs = [Document(page_content="hello", metadata={"source": "test.py", "chunk_index": 0})]
        store.add_documents(docs)

        mock_client.upsert.assert_called_once()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_add_documents_error(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_embeddings.side_effect = RuntimeError("embedding error")
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        docs = [Document(page_content="hello", metadata={"source": "test.py"})]
        with pytest.raises(RuntimeError):
            store.add_documents(docs)

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_delete_by_source(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        store.delete_by_source("test.py")

        mock_client.delete.assert_called_once()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_delete_by_source_no_collection(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        store.delete_by_source("test.py")

        mock_client.delete.assert_not_called()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        point = MagicMock()
        point.payload = {"page_content": "result text", "source": "a.py"}
        point.score = 0.95
        mock_client.query_points.return_value = MagicMock(points=[point])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        results = store.similarity_search("test query", k=2)

        assert len(results) == 1
        assert results[0].page_content == "result text"

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search_with_score(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        point = MagicMock()
        point.payload = {"page_content": "text", "source": "b.py"}
        point.score = 0.88
        mock_client.query_points.return_value = MagicMock(points=[point])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1, 0.2]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        results = store.similarity_search_with_score("query")

        assert len(results) == 1
        assert results[0][1] == 0.88

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search_no_collection(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        results = store.similarity_search_with_score("query")
        assert results == []

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search_error(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.query_points.side_effect = RuntimeError("search error")
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(RuntimeError, match="Vector search failed"):
            store.similarity_search_with_score("query")

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search_with_filter(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.query_points.return_value = MagicMock(points=[])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        results = store.similarity_search_with_score("query", filter_query={"repo": "test"})
        assert results == []

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_list_repos(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        hit1 = MagicMock()
        hit1.value = "repo-a"
        hit2 = MagicMock()
        hit2.value = "repo-b"
        mock_client.facet.return_value = MagicMock(hits=[hit1, hit2])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        repos = store.list_repos()

        assert repos == ["repo-a", "repo-b"]

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_list_repos_no_collection(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.list_repos() == []

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_list_repos_retry_on_error(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        hit = MagicMock()
        hit.value = "repo-x"
        mock_client.facet.side_effect = [
            RuntimeError("missing index"),
            MagicMock(hits=[hit]),
        ]
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        repos = store.list_repos()
        assert repos == ["repo-x"]

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_delete_by_repo(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.count.return_value = MagicMock(count=5)
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        deleted = store.delete_by_repo("test-repo")

        assert deleted == 5
        mock_client.delete.assert_called_once()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_delete_by_repo_no_collection(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.delete_by_repo("test-repo") == 0

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_delete_by_repo_error(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.count.side_effect = RuntimeError("db error")
        mock_client_cls.return_value = mock_client

        store = QdrantStore()
        assert store.delete_by_repo("test-repo") == 0

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_deterministic_id(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        doc1 = Document(page_content="hello", metadata={"source": "a.py", "chunk_index": 0})
        doc2 = Document(page_content="hello", metadata={"source": "a.py", "chunk_index": 0})
        doc3 = Document(page_content="hello", metadata={"source": "a.py", "chunk_index": 1})

        id1 = QdrantStore._deterministic_id(doc1)
        id2 = QdrantStore._deterministic_id(doc2)
        id3 = QdrantStore._deterministic_id(doc3)

        assert id1 == id2  # Same source+chunk → same ID
        assert id1 != id3  # Different chunk → different ID

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_recreate_collection(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:

        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client_cls.return_value = mock_client

        QdrantStore(recreate_collection=True)

        mock_client.delete_collection.assert_called_once_with("documents")


class TestSqliteChatStorageErrors:
    """Tests for SQLite error handling branches in SqliteChatStorage."""

    def test_save_chat_sqlite_error(self, tmp_path: Path) -> None:

        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        with patch.object(storage, "_get_connection") as mock_conn:
            conn = MagicMock()
            conn.execute.side_effect = sqlite3.Error("write error")
            mock_conn.return_value = conn

            with pytest.raises(sqlite3.Error):
                storage.save_chat("c1", [{"role": "user", "content": "hi"}])
            conn.close.assert_called_once()

    def test_get_chat_sqlite_error(self, tmp_path: Path) -> None:

        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        with patch.object(storage, "_get_connection") as mock_conn:
            conn = MagicMock()
            conn.execute.side_effect = sqlite3.Error("read error")
            mock_conn.return_value = conn

            with pytest.raises(sqlite3.Error):
                storage.get_chat("c1")
            conn.close.assert_called_once()

    def test_list_chats_sqlite_error(self, tmp_path: Path) -> None:

        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        with patch.object(storage, "_get_connection") as mock_conn:
            conn = MagicMock()
            conn.execute.side_effect = sqlite3.Error("list error")
            mock_conn.return_value = conn

            result = storage.list_chats()
            assert result == []
            conn.close.assert_called_once()

    def test_delete_chat_sqlite_error(self, tmp_path: Path) -> None:

        with patch("codebase_rag.database.sqlite_storage.Config") as mock_cfg:
            mock_cfg.get_instance.return_value = MagicMock(chat_storage_path=tmp_path / "test.db")
            storage = SqliteChatStorage(db_path=tmp_path / "test.db")

        with patch.object(storage, "_get_connection") as mock_conn:
            conn = MagicMock()
            conn.execute.side_effect = sqlite3.Error("delete error")
            mock_conn.return_value = conn

            result = storage.delete_chat("c1")
            assert result is False
            conn.close.assert_called_once()


class TestChatHistoryManagerErrors:
    """Tests for error handling branches in ChatHistoryManager."""

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_save_no_storage(self, mock_storage_cls: MagicMock) -> None:

        mock_storage_cls.side_effect = RuntimeError("init failed")
        mgr = ChatHistoryManager()
        assert mgr.save_chat_history("c1", []) is False

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_get_exception(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.get_chat.side_effect = RuntimeError("db error")
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        assert mgr.get_chat_history("c1") is None

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_list_exception(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.list_chats.side_effect = RuntimeError("db error")
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        assert mgr.list_chat_histories() == []

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_delete_exception(self, mock_storage_cls: MagicMock) -> None:

        mock_storage = MagicMock()
        mock_storage.delete_chat.side_effect = RuntimeError("db error")
        mock_storage_cls.return_value = mock_storage

        mgr = ChatHistoryManager()
        assert mgr.delete_chat_history("c1") is False


class TestGetChatHistoryManagerSingleton:
    """Tests for get_chat_history_manager singleton function."""

    @patch("codebase_rag.database.chat_storage.SqliteChatStorage")
    def test_returns_singleton(self, mock_storage_cls: MagicMock) -> None:

        mod._chat_history_manager_instance = None
        try:
            mgr1 = mod.get_chat_history_manager()
            mgr2 = mod.get_chat_history_manager()
            assert mgr1 is mgr2
        finally:
            mod._chat_history_manager_instance = None
