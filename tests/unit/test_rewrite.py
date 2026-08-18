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

    def test_a_drained_slow_expansion_frees_the_slot_for_the_next_query(self) -> None:
        """A timed-out expansion's slot is freed when its model call drains, so a later query
        gets its own full budget rather than inheriting the backlog. Guards the done-callback
        release: a slot that were never freed would keep the next query falling back. (The
        concurrent case, where several queries arrive at once, is covered separately.)"""
        import threading

        retriever = _StubRetriever()
        hold = threading.Event()

        class _HeldLLM:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    hold.wait(timeout=5.0)  # the first call is held until the test drains it
                return "terms"

        stage = RewritingRetriever(retriever, _HeldLLM(), timeout_s=0.05)

        # First query times out against the held call and falls back.
        stage.search("first query")
        assert retriever.received_query == "first query"

        # Drain the slow call so it frees its slot. The slot is released by the done callback
        # once the call finishes, so wait for the next query to expand rather than guessing a
        # wall-clock sleep; bounded so a regression fails instead of hanging.
        hold.set()
        deadline = time.monotonic() + 2.0
        while retriever.received_query != "second query terms":
            if time.monotonic() > deadline:
                raise AssertionError("second query did not expand after the slow call drained")
            stage.search("second query")
            time.sleep(0.001)

        assert retriever.received_query == "second query terms"

    def test_concurrent_queries_do_not_pay_the_timeout_for_a_slot(self) -> None:
        """Several queries reaching the stage at once must not make the ones that lose the
        expansion wait out the full timeout. The model call is slower than the queries are
        spaced but faster than the timeout, so a query that queues behind another's expansion
        spends its budget waiting for a worker instead of for the model."""
        import threading

        model_delay_s = 0.6
        timeout_s = 1.5
        n_queries = 4

        received: list[str] = []
        received_lock = threading.Lock()

        class _RecordingRetriever:
            def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
                with received_lock:
                    received.append(query)
                return []

        llm = _StubLLM(response="expanded_identifier", delay_s=model_delay_s)
        stage = RewritingRetriever(_RecordingRetriever(), llm, timeout_s=timeout_s)

        latencies: dict[int, float] = {}
        barrier = threading.Barrier(n_queries)

        def run(i: int) -> None:
            barrier.wait()
            start = time.perf_counter()
            stage.search(f"query{i}")
            latencies[i] = time.perf_counter() - start

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n_queries)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stage.close()

        expanded = [i for i in range(n_queries) if f"query{i} expanded_identifier" in received]
        # The stage must still expand under load, not turn itself off entirely.
        assert expanded, "no query was expanded under concurrent load"
        # A query that does not get the expansion must fall back promptly, not pay the timeout.
        promptly = timeout_s / 2
        slow = [i for i in range(n_queries) if i not in expanded and latencies[i] > promptly]
        assert not slow, f"queries {slow} fell back only after waiting out the timeout"

    def test_max_concurrency_below_one_is_rejected(self) -> None:
        """A directly constructed stage (or Config) bypasses get_instance validation, so the
        stage itself must reject a limit below 1 rather than dying in the executor with a
        generic error that does not name the setting."""
        import pytest

        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            RewritingRetriever(_StubRetriever(), _StubLLM(), max_concurrency=0)

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

    def test_close_during_in_flight_expansion_falls_back(self) -> None:
        """close() while an expansion is running must leave the stage falling back, not
        raising, for a request that still holds the old stack after an ingest rebuild."""
        import threading

        retriever = _StubRetriever()
        llm = _StubLLM(response="terms", delay_s=0.5)
        stage = RewritingRetriever(retriever, llm, timeout_s=1.0)

        in_flight = threading.Thread(target=stage.search, args=("in flight",))
        in_flight.start()
        time.sleep(0.1)  # let the expansion submit and start running
        stage.close()
        in_flight.join()

        # A request still holding the closed stage must fall back, not raise.
        result = stage.search("after close")

        assert retriever.received_query == "after close"
        assert result

    def test_close_stages_is_safe_on_none_and_bare_retriever(self) -> None:
        from codebase_rag.retrieval.retrieval_stack import close_stages

        close_stages(None)
        close_stages(_StubRetriever())  # no close(), must not raise
