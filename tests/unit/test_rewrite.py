"""Tests for the query-rewrite stage, covering expansion, fallback, and timeout."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document

from codebase_rag.retrieval.rewrite import RewritingRetriever


class _StubRetriever:
    """Records the query it was asked to retrieve on."""

    def __init__(self) -> None:
        self.received_query: str | None = None

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        self.received_query = query
        return [(Document(page_content="x", metadata={"source": "x.py"}), 1.0)]


class _StubLLM:
    def __init__(self, response: str = "", delay_s: float = 0.0, raises: bool = False) -> None:
        self.response = response
        self.delay_s = delay_s
        self.raises = raises

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raises:
            raise RuntimeError("model down")
        return self.response


class TestRewritingRetriever:
    def test_expansion_is_appended_to_the_original_query(self) -> None:
        retriever = _StubRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(response="retry backoff max_retries"))

        stage.search("retry logic")

        assert retriever.received_query == "retry logic retry backoff max_retries"

    def test_empty_expansion_falls_back_to_original(self) -> None:
        retriever = _StubRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(response="   "))

        stage.search("retry logic")

        assert retriever.received_query == "retry logic"

    def test_model_error_falls_back_to_original_without_raising(self) -> None:
        retriever = _StubRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(raises=True))

        result = stage.search("retry logic")

        assert retriever.received_query == "retry logic"
        assert result  # retrieval still ran

    def test_timeout_falls_back_to_original(self) -> None:
        retriever = _StubRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(response="late", delay_s=0.5), timeout_s=0.05)

        stage.search("retry logic")

        assert retriever.received_query == "retry logic"

    def test_one_slow_expansion_does_not_time_out_the_next_query(self) -> None:
        """The pool has a single worker, so a timed-out expansion left queued would start the
        next query's call late and time it out in turn, forever."""
        retriever = _StubRetriever()
        llm = _StubLLM(response="terms", delay_s=0.3)
        stage = RewritingRetriever(retriever, llm, timeout_s=0.05)

        # First query times out against the slow call and falls back.
        stage.search("first query")
        assert retriever.received_query == "first query"

        # Once the slow call drains, a later query must get its own full timeout budget
        # rather than inheriting the backlog.
        time.sleep(0.35)
        llm.delay_s = 0.0
        stage.search("second query")

        assert retriever.received_query == "second query terms"

    def test_k_is_passed_through(self) -> None:
        class _RecordingRetriever(_StubRetriever):
            asked_k: int | None = None

            def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
                self.asked_k = k
                return super().search(query, k)

        retriever = _RecordingRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(response="terms"))

        stage.search("q", k=7)

        assert retriever.asked_k == 7

    def test_verbose_expansion_is_capped(self) -> None:
        """A chatty model must not append a paragraph to a keyword query."""
        from codebase_rag.retrieval.rewrite import MAX_EXPANSION_TERMS

        retriever = _StubRetriever()
        # 200 terms of noise; only the first MAX_EXPANSION_TERMS should survive.
        verbose = " ".join(f"term{i}" for i in range(200))
        stage = RewritingRetriever(retriever, _StubLLM(response=verbose))

        stage.search("retry logic")

        assert retriever.received_query is not None
        # original 2 words + at most MAX_EXPANSION_TERMS appended terms
        assert len(retriever.received_query.split()) == 2 + MAX_EXPANSION_TERMS
        assert retriever.received_query.startswith("retry logic ")


class TestCloseStages:
    def test_close_shuts_down_the_executor(self) -> None:
        retriever = _StubRetriever()
        stage = RewritingRetriever(retriever, _StubLLM(response="terms"))

        stage.close()

        assert stage._executor._shutdown is True

    def test_close_stages_walks_the_chain(self) -> None:
        from codebase_rag.retrieval.retrieval_stack import close_stages

        inner = _StubRetriever()
        stage = RewritingRetriever(inner, _StubLLM(response="terms"))

        close_stages(stage)

        assert stage._executor._shutdown is True

    def test_close_stages_is_safe_on_none_and_bare_retriever(self) -> None:
        from codebase_rag.retrieval.retrieval_stack import close_stages

        close_stages(None)
        close_stages(_StubRetriever())  # no close(), must not raise
