"""Unit tests for AppRuntime construction and lifecycle, with all backends mocked."""

from unittest.mock import MagicMock, patch

from codebase_rag.app.runtime import AppRuntime, get_repo_list


def _config(**overrides: object) -> MagicMock:
    config = MagicMock()
    config.qdrant_host = "localhost"
    config.qdrant_port = 6333
    config.collection_name = "docs"
    config.llm_model_name = "test-model"
    config.ollama_base_url = "http://localhost:11434"
    config.default_repo_url = ""
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_runtime(config: MagicMock, *, existing_repos: list[str] | None = None) -> AppRuntime:
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = existing_repos is not None
    mock_qdrant.list_repos.return_value = existing_repos or []

    mock_llm = MagicMock()
    mock_llm.check_connection.return_value = {"status": "connected"}
    mock_llm.check_model_availability.return_value = {"status": "available"}
    mock_llm.num_ctx = 8192
    mock_llm.max_tokens = 1024
    mock_llm.prompt_budget_chars = (8192 - 1024 - 256) * 4

    with (
        patch("codebase_rag.app.runtime.QdrantStore", return_value=mock_qdrant),
        patch("codebase_rag.app.runtime.OllamaClient", return_value=mock_llm),
        patch("codebase_rag.app.runtime._load_or_create_bm25_retriever", return_value=MagicMock()),
        patch("codebase_rag.app.runtime.IngestionManager.start") as mock_start,
    ):
        runtime = AppRuntime(config)
    return runtime, mock_start  # type: ignore[return-value]


class TestAppRuntimeConstruction:
    def test_builds_all_components(self) -> None:
        runtime, _ = _build_runtime(_config())
        assert runtime.qdrant_store is not None
        assert runtime.vector_retriever is not None
        assert runtime.hybrid_retriever is not None
        assert runtime.llm is not None
        assert runtime.folder_picker is not None
        assert runtime.ingestion is not None

    def test_no_default_repo_skips_auto_ingest(self) -> None:
        runtime, mock_start = _build_runtime(_config(default_repo_url=""))
        mock_start.assert_not_called()

    def test_default_repo_with_no_existing_data_starts_auto_ingest(self) -> None:
        runtime, mock_start = _build_runtime(
            _config(default_repo_url="https://github.com/owner/default-repo"), existing_repos=None
        )
        mock_start.assert_called_once_with("https://github.com/owner/default-repo", kind="auto")

    def test_default_repo_with_existing_data_skips_auto_ingest(self) -> None:
        runtime, mock_start = _build_runtime(
            _config(default_repo_url="https://github.com/owner/default-repo"), existing_repos=["some-repo"]
        )
        mock_start.assert_not_called()


class TestSwapBm25:
    def test_updates_both_bm25_and_hybrid_retriever(self) -> None:
        runtime, _ = _build_runtime(_config())
        new_index = MagicMock()

        runtime.swap_bm25(new_index)

        assert runtime.bm25_retriever is new_index
        assert runtime.hybrid_retriever.bm25_retriever is new_index


class TestDeleteRepo:
    def test_deletes_from_qdrant_and_rebuilds_bm25(self) -> None:
        runtime, _ = _build_runtime(_config())
        runtime.qdrant_store.delete_by_repo = MagicMock(return_value=5)
        get_repo_list.clear()

        with (
            patch("codebase_rag.retrieval.bm25_search.delete_bm25_corpus") as mock_delete_corpus,
            patch("codebase_rag.retrieval.bm25_search.rebuild_bm25_index", return_value=MagicMock()) as mock_rebuild,
        ):
            deleted = runtime.delete_repo("some-repo")

        assert deleted == 5
        mock_delete_corpus.assert_called_once()
        mock_rebuild.assert_called_once()


class TestNewRagChain:
    def test_creates_a_fresh_chain_sharing_the_retriever(self) -> None:
        runtime, _ = _build_runtime(_config())
        with patch("codebase_rag.app.runtime.RAGChain") as mock_rag_chain_cls:
            runtime.new_rag_chain()
            mock_rag_chain_cls.assert_called_once_with(
                retriever=runtime.hybrid_retriever,
                llm=runtime.llm,
                use_conversation_memory=True,
                max_conversation_history=10,
                prompt_budget_chars=(8192 - 1024 - 256) * 4,
            )
