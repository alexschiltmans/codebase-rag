"""Unit tests for database modules: qdrant_store, sqlite_storage, chat_storage."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

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
        mock_emb.assert_called_once_with(
            model_name="sentence-transformers/other-model", max_seq_length=None, dtype=None
        )

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_embedding_model_defaults_to_none(self, mock_client_cls: MagicMock, mock_emb: MagicMock) -> None:
        QdrantStore()
        mock_emb.assert_called_once_with(model_name=None, max_seq_length=None, dtype=None)

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

        # One upsert records the model binding on the new meta collection, one upserts the batch.
        assert mock_client.upsert.call_count == 2

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_add_documents_reuses_surviving_meta_collection(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """Dropping a collection leaves its `__meta` sidecar behind, so a rebuild has to overwrite
        the binding rather than try to create the sidecar a second time and get a 409."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.collection_exists.return_value = True
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        store.add_documents([Document(page_content="hello", metadata={"source": "test.py", "chunk_index": 0})])

        created = [c.kwargs.get("collection_name") for c in mock_client.create_collection.call_args_list]
        assert "test_collection__meta" not in created
        assert mock_client.upsert.call_count == 2

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
        # No `__meta` sidecar, which is the state of any collection created before
        # model binding existed. Verification is a no-op there rather than a failure.
        mock_client.collection_exists.return_value = False
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
        # No `__meta` sidecar, which is the state of any collection created before
        # model binding existed. Verification is a no-op there rather than a failure.
        mock_client.collection_exists.return_value = False
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
    def test_add_documents_raises_on_model_mismatch_before_writing(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """Checking only on the read path reports the corruption after it has happened. A
        same-dimension model re-ingested into an existing collection is accepted by Qdrant and
        leaves two models' vectors under one index."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.collection_exists.return_value = True

        meta_point = MagicMock()
        meta_point.payload = {"embedding_model": "other/model", "dimension": 3}
        mock_client.retrieve.return_value = [meta_point]
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "sentence-transformers/all-mpnet-base-v2"
        mock_emb.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ValueError, match="other/model.*all-mpnet-base-v2"):
            store.add_documents([Document(page_content="hello", metadata={"source": "a.py"})])

        mock_client.upsert.assert_not_called()

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_transport_failure_does_not_latch_the_guard_off(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """A failed read must not be mistaken for 'nothing recorded'. Swallowing it would
        disable the guard for the rest of the process after a single connection blip."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.side_effect = ConnectionError("qdrant restarting")
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "sentence-transformers/all-mpnet-base-v2"
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ConnectionError):
            store.similarity_search_with_score("query")
        assert store._model_binding_verified is False

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_encoding_settings_mismatch_is_caught_not_just_model_name(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """A prompt prefix changes the vectors as much as changing the model does."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.collection_exists.return_value = True

        meta_point = MagicMock()
        meta_point.payload = {
            "embedding_model": "same/model",
            "document_prompt": "passage: ",
            "dimension": 768,
        }
        mock_client.retrieve.return_value = [meta_point]
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "same/model"
        mock_emb.document_prompt = ""
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ValueError, match="document_prompt"):
            store.similarity_search_with_score("query")

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_a_recorded_null_dtype_is_a_value_not_an_absence(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """Every collection built without EMBEDDING_DTYPE records dtype as null, meaning default
        precision. Reading that as 'predates the check' waves through a precision swap, which is
        the exact case this guard exists for and the common one."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        meta_point = MagicMock()
        meta_point.payload = {"embedding_model": "same/model", "dtype": None, "dimension": 768}
        mock_client.retrieve.return_value = [meta_point]
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "same/model"
        mock_emb.dtype = "float16"
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ValueError, match="dtype"):
            store.similarity_search_with_score("query")

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_a_field_absent_from_the_payload_still_predates_the_check(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """The other half of the same distinction: sidecars written before a field existed carry no
        key for it, and that genuinely is no evidence either way."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        meta_point = MagicMock()
        meta_point.payload = {"embedding_model": "same/model", "dimension": 768}
        mock_client.retrieve.return_value = [meta_point]
        mock_client.query_points.return_value = MagicMock(points=[])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "same/model"
        mock_emb.dtype = "float16"
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        assert store.similarity_search_with_score("query") == []

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_a_collection_with_no_sidecar_is_still_checked_for_width(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        """Without a sidecar there is no binding to compare, and `_ensure_collection` never
        back-fills one. Width is the only signal left, and it catches a model swap across
        dimensions rather than letting two models' vectors accumulate under one index."""
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.collection_exists.return_value = False
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=768)))
        )
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "some/1024-dim-model"
        mock_emb.model.get_sentence_embedding_dimension.return_value = 1024
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ValueError, match="768-dimension vectors.*produces 1024"):
            store.similarity_search_with_score("query")

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_similarity_search_raises_on_model_mismatch(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])

        meta_point = MagicMock()
        meta_point.payload = {"embedding_model": "other/model", "dimension": 768}
        mock_client.retrieve.return_value = [meta_point]
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.model_name = "sentence-transformers/all-mpnet-base-v2"
        mock_emb_cls.return_value = mock_emb

        store = QdrantStore()
        with pytest.raises(ValueError, match="other/model.*all-mpnet-base-v2"):
            store.similarity_search_with_score("query")

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
        store, mock_client = self._searching_store(mock_client_cls, mock_emb_cls)
        mock_client.query_points.side_effect = RuntimeError("search error")

        with pytest.raises(RuntimeError, match="Vector search failed"):
            store.similarity_search_with_score("query")

    @staticmethod
    def _searching_store(mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> tuple[QdrantStore, MagicMock]:
        """A store whose collection exists but has no sidecar, wired to return no points."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        coll = MagicMock()
        coll.name = "documents"
        mock_client.get_collections.return_value = MagicMock(collections=[coll])
        mock_client.query_points.return_value = MagicMock(points=[])
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1]
        mock_emb_cls.return_value = mock_emb

        return QdrantStore(), mock_client

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_scalar_filter_value_matches_exactly(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:
        store, mock_client = self._searching_store(mock_client_cls, mock_emb_cls)

        store.similarity_search_with_score("query", filter_query={"repo": "one-repo"})

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert query_filter == Filter(must=[FieldCondition(key="repo", match=MatchValue(value="one-repo"))])

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_list_filter_value_matches_any(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:
        """A list has to become one MatchAny. Split into a MatchValue per entry it would be ANDed
        with itself by the enclosing `must` and match nothing, which reads as an empty index."""
        store, mock_client = self._searching_store(mock_client_cls, mock_emb_cls)

        store.similarity_search_with_score("query", filter_query={"repo": ["repo-a", "repo-b"]})

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert query_filter == Filter(must=[FieldCondition(key="repo", match=MatchAny(any=["repo-a", "repo-b"]))])

    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_scalar_and_list_filter_keys_mix(self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock) -> None:
        store, mock_client = self._searching_store(mock_client_cls, mock_emb_cls)

        store.similarity_search_with_score("query", filter_query={"repo": ["repo-a"], "source": "a.py"})

        query_filter = mock_client.query_points.call_args.kwargs["query_filter"]
        assert query_filter == Filter(
            must=[
                FieldCondition(key="repo", match=MatchAny(any=["repo-a"])),
                FieldCondition(key="source", match=MatchValue(value="a.py")),
            ]
        )

    @pytest.mark.parametrize("filter_query", [None, {}])
    @patch("codebase_rag.database.qdrant_store.EmbeddingManager")
    @patch("codebase_rag.database.qdrant_store.QdrantClient")
    def test_absent_filter_sends_no_filter(
        self, mock_client_cls: MagicMock, mock_emb_cls: MagicMock, filter_query: dict | None
    ) -> None:
        store, mock_client = self._searching_store(mock_client_cls, mock_emb_cls)

        store.similarity_search_with_score("query", filter_query=filter_query)

        assert mock_client.query_points.call_args.kwargs["query_filter"] is None

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
