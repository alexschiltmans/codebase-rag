"""Tests for ApiState's retriever selection and RAGChain wiring.

QdrantStore and the LLM client are mocked; these tests exercise ApiState's
own wiring decisions, not the retrieval or LLM stack.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from codebase_rag.api.state import ApiState
from codebase_rag.config import Config


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    Config._instance = None
    yield
    Config._instance = None


@contextmanager
def _patched_state() -> Iterator[ApiState]:
    """An ApiState whose mocks stay active for the caller.

    Holding the patches past construction matters for anything that reloads the
    BM25 index: unpatched, `_load_bm25_retriever` reads whatever real
    `data/cache/bm25_retriever.json` the developer happens to have. Each load
    returns a distinct mock, so a test asserting two attributes are the same
    object fails when they should differ.
    """
    with (
        patch("codebase_rag.api.state.QdrantStore"),
        patch("codebase_rag.api.state.BM25Retriever") as bm25_cls,
        patch("codebase_rag.api.state.create_llm_client") as create_llm,
    ):
        bm25_cls.side_effect = lambda *_args, **_kwargs: MagicMock()
        bm25_cls.load_json.side_effect = lambda *_args, **_kwargs: MagicMock()
        llm = MagicMock()
        llm.prompt_budget_chars = 200
        create_llm.return_value = llm
        yield ApiState(config=Config.get_instance())


def _make_state() -> ApiState:
    with _patched_state() as state:
        return state


class TestRetrieverSelection:
    def test_defaults_to_bm25(self) -> None:
        state = _make_state()
        assert state.retriever is state.bm25_retriever

    def test_hybrid_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVER", "hybrid")
        state = _make_state()
        assert state.retriever is state.hybrid_retriever

    def test_refresh_bm25_keeps_selection_in_sync(self) -> None:
        with _patched_state() as state:
            original = state.bm25_retriever
            assert state.retriever is original

            state.refresh_bm25()

            assert state.bm25_retriever is not original, "refresh should have loaded a new index"
            assert state.retriever is state.bm25_retriever

    def test_refresh_bm25_keeps_hybrid_selection_in_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RETRIEVER", "hybrid")
        with _patched_state() as state:
            assert state.retriever is state.hybrid_retriever

            state.refresh_bm25()

            assert state.retriever is state.hybrid_retriever
            assert state.hybrid_retriever.bm25_retriever is state.bm25_retriever


class TestPromptBudget:
    def test_new_rag_chain_enforces_llms_prompt_budget(self) -> None:
        state = _make_state()

        chain = state.new_rag_chain()

        assert chain.prompt_budget_chars == state.llm.prompt_budget_chars

    def test_oversized_context_is_trimmed_before_reaching_the_model(self) -> None:
        state = _make_state()
        state.llm.prompt_budget_chars = 250
        long_doc = Document(
            page_content="x" * 1000,
            metadata={"source": "big.py", "start_line": 1, "end_line": 50},
        )

        chain = state.new_rag_chain()
        prompt, docs, docs_dropped, _, _ = chain._build_within_budget("what does this do?", [long_doc])

        assert docs_dropped == 1
        assert docs == []
        assert "x" * 1000 not in prompt
