"""Unit tests for AppRuntime construction and lifecycle, with all backends mocked."""

import logging
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.app.runtime import AppRuntime, get_repo_list


def _config(**overrides: object) -> MagicMock:
    config = MagicMock()
    config.qdrant_host = "localhost"
    config.qdrant_port = 6333
    config.collection_name = "docs"
    config.llm_model_name = "test-model"
    config.ollama_base_url = "http://localhost:11434"
    config.default_repo_url = ""
    config.rerank_enabled = False
    config.rewrite_enabled = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_runtime(config: MagicMock, *, existing_repos: list[str] | None = None) -> tuple[AppRuntime, MagicMock]:
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = existing_repos is not None
    mock_qdrant.list_repos.return_value = existing_repos or []

    mock_llm = MagicMock()
    mock_llm.check_connection.return_value = {"status": "connected", "url": "http://localhost:11434"}
    mock_llm.check_model_availability.return_value = {"status": "available"}
    mock_llm.check_runtime_placement.return_value = {"placement": "gpu", "url": "http://localhost:11434"}
    mock_llm.num_ctx = 8192
    mock_llm.max_tokens = 1024
    mock_llm.prompt_budget_chars = (8192 - 1024 - 256) * 4

    with (
        patch("codebase_rag.app.runtime.QdrantStore", return_value=mock_qdrant),
        patch("codebase_rag.app.runtime.create_llm_client", return_value=mock_llm),
        patch("codebase_rag.app.runtime._load_or_create_bm25_retriever", return_value=MagicMock()),
        patch("codebase_rag.app.runtime.IngestionManager.start") as mock_start,
    ):
        runtime = AppRuntime(config)
    return runtime, mock_start


class TestAppRuntimeConstruction:
    def test_builds_all_components(self) -> None:
        runtime, _ = _build_runtime(_config())
        assert runtime.qdrant_store is not None
        assert runtime.vector_retriever is not None
        assert runtime.bm25_retriever is not None
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
    def test_updates_bm25_retriever(self) -> None:
        runtime, _ = _build_runtime(_config())
        new_index = MagicMock()

        runtime.swap_bm25(new_index)

        assert runtime.bm25_retriever is new_index


class TestDeleteRepo:
    def test_deletes_from_qdrant_and_rebuilds_bm25(self) -> None:
        runtime, _ = _build_runtime(_config())
        cast(Any, runtime.qdrant_store).delete_by_repo = MagicMock(return_value=5)
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
                retriever=runtime.retriever,
                llm=runtime.llm,
                use_conversation_memory=True,
                max_conversation_history=10,
                prompt_budget_chars=(8192 - 1024 - 256) * 4,
            )

    def test_default_config_shares_the_bare_bm25_retriever(self) -> None:
        runtime, _ = _build_runtime(_config())
        assert runtime.retriever is runtime.bm25_retriever

    def test_rerank_stage_wraps_the_base_retriever_when_enabled(self) -> None:
        runtime, _ = _build_runtime(_config(rerank_enabled=True, rerank_model="m", rerank_candidate_depth=50))
        from codebase_rag.retrieval.rerank import RerankingRetriever

        assert isinstance(runtime.retriever, RerankingRetriever)
        assert runtime.retriever.retriever is runtime.bm25_retriever

    def test_rewrite_stage_is_outermost_when_both_enabled(self) -> None:
        runtime, _ = _build_runtime(
            _config(
                rerank_enabled=True,
                rerank_model="m",
                rerank_candidate_depth=50,
                rewrite_enabled=True,
                rewrite_timeout_s=5.0,
            )
        )
        from codebase_rag.retrieval.rerank import RerankingRetriever
        from codebase_rag.retrieval.rewrite import RewritingRetriever

        assert isinstance(runtime.retriever, RewritingRetriever)
        assert isinstance(runtime.retriever.retriever, RerankingRetriever)

    def test_stack_is_built_once_not_per_chain(self) -> None:
        """Two chains must receive the same composed retriever object, so an enabled
        reranker is not reloaded per query. Asserts on what RAGChain was handed,
        not just that the attribute is stable, so a per-query rebuild would fail here."""
        runtime, _ = _build_runtime(_config(rerank_enabled=True, rerank_model="m", rerank_candidate_depth=50))
        with patch("codebase_rag.app.runtime.RAGChain") as mock_rag_chain_cls:
            runtime.new_rag_chain()
            runtime.new_rag_chain()

        first_retriever = mock_rag_chain_cls.call_args_list[0].kwargs["retriever"]
        second_retriever = mock_rag_chain_cls.call_args_list[1].kwargs["retriever"]
        assert first_retriever is second_retriever
        assert first_retriever is runtime.retriever

    def test_swap_bm25_rebuilds_the_stack_over_the_new_index(self) -> None:
        runtime, _ = _build_runtime(_config(rerank_enabled=True, rerank_model="m", rerank_candidate_depth=50))
        new_index = MagicMock()

        runtime.swap_bm25(new_index)

        from codebase_rag.retrieval.rerank import RerankingRetriever

        assert isinstance(runtime.retriever, RerankingRetriever)
        assert runtime.retriever.retriever is new_index


class TestHealthChecks:
    def test_run_health_checks_populates_runtime_health(self) -> None:
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        _run_health_checks(runtime)

        assert "model" in runtime.health
        assert "checked_at" in runtime.health
        assert runtime.health["model"]["status"] == "available"

    def test_run_health_checks_stores_model_status_on_not_found(self) -> None:
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        cast(MagicMock, runtime.llm.check_model_availability).return_value = {
            "status": "not_found",
            "message": "Model not found",
            "suggested_action": "Run 'ollama pull model'",
        }

        _run_health_checks(runtime)

        assert runtime.health["model"]["status"] == "not_found"
        assert runtime.health["model"]["suggested_action"] == "Run 'ollama pull model'"
        assert "checked_at" in runtime.health

    def test_run_health_checks_publishes_connection_and_placement(self) -> None:
        """All three results land together, so a reader never sees a half-filled dict."""
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        _run_health_checks(runtime)

        assert runtime.health["connection"]["url"] == "http://localhost:11434"
        assert runtime.health["model"]["status"] == "available"
        assert runtime.health["placement"]["placement"] == "gpu"
        assert "checked_at" in runtime.health

    def test_run_health_checks_survives_a_raising_placement_check(self) -> None:
        """Placement is the least important result; losing the endpoint over it costs more."""
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        cast(MagicMock, runtime.llm.check_runtime_placement).side_effect = RuntimeError("ps timed out")

        _run_health_checks(runtime)

        assert runtime.health["connection"]["url"] == "http://localhost:11434"
        assert runtime.health["model"]["status"] == "available"
        assert runtime.health["placement"]["placement"] == "unknown"

    def test_run_health_checks_logs_the_resolved_endpoint(self, caplog: pytest.LogCaptureFixture) -> None:
        """A headless run reports the same fact the sidebar shows."""
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        with caplog.at_level(logging.INFO, logger="codebase_rag.app.runtime"):
            _run_health_checks(runtime)

        messages = [record.getMessage() for record in caplog.records]
        assert any("http://localhost:11434" in message and "gpu" in message for message in messages)

    def test_run_health_checks_handles_warm_up_exceptions(self) -> None:
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        with patch("codebase_rag.app.runtime._warm_up_vector_store", side_effect=RuntimeError("Warm-up failed")):
            _run_health_checks(runtime)

        assert "model" in runtime.health
        assert runtime.health["model"]["status"] == "available"

    def test_run_health_checks_still_warms_up_after_llm_check_failure(self) -> None:
        """The LLM check and the vector-store warm-up are independent; one failing must not
        skip the other, which an early `return` in the first try/except used to do.
        """
        from codebase_rag.app.runtime import _run_health_checks

        runtime, _ = _build_runtime(_config())
        cast(MagicMock, runtime.llm.check_connection).side_effect = RuntimeError("LLM unreachable")

        with patch("codebase_rag.app.runtime._warm_up_vector_store") as mock_warm_up:
            _run_health_checks(runtime)

        mock_warm_up.assert_called_once_with(runtime.vector_retriever)
