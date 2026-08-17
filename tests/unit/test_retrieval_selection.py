"""Tests for resolving the configured retriever.

The requirement under test is that one setting decides what every entry point queries
with. Two of the three used to hardcode `BM25Retriever` and never read the setting, so
`RETRIEVER=hybrid` moved the HTTP API and silently did nothing to the app or the CLI.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.config import SUPPORTED_RETRIEVERS, Config
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.retrieval_stack import select_base_retriever


def _config(retriever: str) -> Config:
    return replace(Config(), retriever=retriever)


class TestSelectBaseRetriever:
    def test_bm25_returns_the_keyword_index_itself(self) -> None:
        bm25 = BM25Retriever([])

        selected = select_base_retriever(_config("bm25"), bm25, MagicMock())

        assert selected is bm25

    def test_bm25_never_builds_a_vector_retriever(self) -> None:
        """The CLI passes a callable that stands up a Qdrant client and loads the embedding
        model, which the default path must not pay for."""
        vector = MagicMock()

        select_base_retriever(_config("bm25"), BM25Retriever([]), vector)

        vector.assert_not_called()

    def test_hybrid_fuses_both_rankers(self) -> None:
        bm25 = BM25Retriever([])
        vector = MagicMock()

        selected = select_base_retriever(_config("hybrid"), bm25, lambda: vector)

        assert isinstance(selected, HybridRetriever)
        assert selected.bm25_retriever is bm25
        assert selected.vector_retriever is vector

    def test_unsupported_value_raises_naming_the_accepted_ones(self) -> None:
        """A `Config` built directly bypasses the env-var validation, and quietly serving BM25
        to an operator who asked for something else is worse than failing."""
        with pytest.raises(ValueError, match="elasticsearch") as excinfo:
            select_base_retriever(_config("elasticsearch"), BM25Retriever([]), MagicMock())

        message = str(excinfo.value)
        assert "elasticsearch" in message
        for name in SUPPORTED_RETRIEVERS:
            assert name in message


class TestConfigRejectsUnsupportedValues:
    def test_env_var_validation_names_the_accepted_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Config, "_instance", None)
        monkeypatch.setenv("RETRIEVER", "elasticsearch")

        with pytest.raises(ValueError, match="RETRIEVER must be") as excinfo:
            Config.get_instance()

        assert "'bm25'" in str(excinfo.value)
        assert "'hybrid'" in str(excinfo.value)


class TestCliHonoursTheSetting:
    """The CLI is the entry point that had no vector store at all, so it is the one where
    honouring the setting has to be shown rather than assumed."""

    def test_default_builds_the_keyword_index_without_touching_qdrant(self) -> None:
        bm25 = BM25Retriever([])

        with (
            patch("codebase_rag.cli._load_bm25_retriever", return_value=bm25),
            patch("codebase_rag.database.qdrant_store.QdrantStore") as store_cls,
        ):
            from codebase_rag.cli import _build_retriever

            selected = _build_retriever(_config("bm25"))

        assert selected is bm25
        store_cls.assert_not_called()

    def test_hybrid_builds_the_fused_retriever(self) -> None:
        bm25 = BM25Retriever([])

        with (
            patch("codebase_rag.cli._load_bm25_retriever", return_value=bm25),
            patch("codebase_rag.database.qdrant_store.QdrantStore", return_value=MagicMock()),
            patch("codebase_rag.retrieval.vector_search.VectorRetriever") as vector_cls,
        ):
            from codebase_rag.cli import _build_retriever

            selected = _build_retriever(_config("hybrid"), ["repo-a"])

        assert isinstance(selected, HybridRetriever)
        assert selected.bm25_retriever is bm25
        # The repo restriction has to reach the vector side too, or --repo would narrow one
        # of the two fused rankers and quietly leave the other searching everything.
        assert vector_cls.call_args.kwargs["repos"] == ["repo-a"]

    def test_stage_flags_reach_the_cli_too(self) -> None:
        """`RERANK_ENABLED` is a per-surface setting like `RETRIEVER`, and the CLI is the
        surface that used to ignore both."""
        from codebase_rag.retrieval.rerank import RerankingRetriever

        bm25 = BM25Retriever([])
        config = replace(Config(), rerank_enabled=True)

        with (
            patch("codebase_rag.cli._load_bm25_retriever", return_value=bm25),
            patch("codebase_rag.retrieval.rerank.RerankingRetriever.__init__", return_value=None) as rerank_init,
        ):
            from codebase_rag.cli import _build_retriever

            selected = _build_retriever(config)

        assert isinstance(selected, RerankingRetriever)
        assert rerank_init.call_args.args[0] is bm25

    def test_no_model_client_is_built_when_rewrite_is_off(self) -> None:
        """Only the rewrite stage needs one, and `query` has no other use for it. Building one
        regardless would put a model handshake in front of a git hook."""
        with (
            patch("codebase_rag.cli._load_bm25_retriever", return_value=BM25Retriever([])),
            patch("codebase_rag.cli._create_llm") as create_llm,
        ):
            from codebase_rag.cli import _build_retriever

            _build_retriever(_config("bm25"))

        create_llm.assert_not_called()

    def test_rewrite_gets_a_model_client_when_the_caller_has_none(self) -> None:
        with (
            patch("codebase_rag.cli._load_bm25_retriever", return_value=BM25Retriever([])),
            patch("codebase_rag.cli._create_llm") as create_llm,
            patch("codebase_rag.retrieval.rewrite.RewritingRetriever.__init__", return_value=None),
        ):
            from codebase_rag.cli import _build_retriever

            _build_retriever(replace(Config(), rewrite_enabled=True))

        create_llm.assert_called_once()


class TestAppRuntimeHonoursTheSetting:
    @staticmethod
    def _runtime(retriever: str) -> object:
        from codebase_rag.app.runtime import AppRuntime

        config = MagicMock()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.collection_name = "docs"
        config.llm_model_name = "test-model"
        config.default_repo_url = ""
        config.retriever = retriever
        config.rerank_enabled = False
        config.rewrite_enabled = False

        with (
            patch("codebase_rag.app.runtime.QdrantStore", return_value=MagicMock()),
            patch("codebase_rag.app.runtime.create_llm_client", return_value=MagicMock()),
            patch("codebase_rag.app.runtime._load_or_create_bm25_retriever", return_value=MagicMock()),
            patch("codebase_rag.app.runtime.IngestionManager.start"),
        ):
            return AppRuntime(config)

    def test_default_queries_with_the_keyword_index(self) -> None:
        runtime = self._runtime("bm25")

        assert runtime.retriever is runtime.bm25_retriever  # type: ignore[attr-defined]

    def test_hybrid_reaches_the_app(self) -> None:
        """This is the case that silently did nothing before: the app hardcoded BM25."""
        runtime = self._runtime("hybrid")

        assert isinstance(runtime.retriever, HybridRetriever)  # type: ignore[attr-defined]
        assert runtime.retriever.bm25_retriever is runtime.bm25_retriever  # type: ignore[attr-defined]

    def test_an_ingest_rewires_the_fused_path_onto_the_new_index(self) -> None:
        """`swap_bm25` is written around a BM25 index being the thing replaced, so the fused
        case is the one that could quietly keep serving the pre-ingest corpus."""
        runtime = self._runtime("hybrid")
        rebuilt = BM25Retriever([])

        runtime.swap_bm25(rebuilt)  # type: ignore[attr-defined]

        assert isinstance(runtime.retriever, HybridRetriever)  # type: ignore[attr-defined]
        assert runtime.retriever.bm25_retriever is rebuilt  # type: ignore[attr-defined]
