"""Tests for the cross-encoder rerank stage, with the model stubbed so they run offline."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document

from codebase_rag.retrieval.rerank import RerankingRetriever


class _StubRetriever:
    """First stage that returns a fixed candidate list, recording the k it was asked for."""

    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.asked_k: int | None = None

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        self.asked_k = k
        return self.results


def _doc(source: str) -> Document:
    return Document(page_content=f"content of {source}", metadata={"source": source})


class _StubModel:
    """Cross-encoder stub: returns caller-supplied scores in pair order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> list[float]:
        return self._scores


def _reranker(candidates: list[tuple[Document, float]], scores: list[float], **kwargs: Any) -> RerankingRetriever:
    stage = RerankingRetriever(_StubRetriever(candidates), **kwargs)
    stage._model = _StubModel(scores)
    return stage


class TestRerankingRetriever:
    def test_pulls_candidate_depth_from_first_stage(self) -> None:
        """The first stage is asked for the full candidate depth, not the output k."""
        candidates = [(_doc("a.py"), 1.0), (_doc("b.py"), 0.5)]
        first_stage = _StubRetriever(candidates)
        stage = RerankingRetriever(first_stage, candidate_depth=50)
        stage._model = _StubModel([0.1, 0.2])

        stage.search("q", k=2)

        assert first_stage.asked_k == 50

    def test_reorders_by_cross_encoder_score(self) -> None:
        candidates = [(_doc("a.py"), 1.0), (_doc("b.py"), 0.9), (_doc("c.py"), 0.8)]
        # First-stage order is a, b, c; the reranker scores c highest, then a, then b.
        stage = _reranker(candidates, [0.5, 0.1, 0.9])

        ranked = stage.search("q", k=3)

        assert [doc.metadata["source"] for doc, _ in ranked] == ["c.py", "a.py", "b.py"]

    def test_truncates_to_output_k_after_reranking(self) -> None:
        candidates = [(_doc("a.py"), 1.0), (_doc("b.py"), 0.9), (_doc("c.py"), 0.8)]
        stage = _reranker(candidates, [0.1, 0.9, 0.5])

        ranked = stage.search("q", k=1)

        assert [doc.metadata["source"] for doc, _ in ranked] == ["b.py"]

    def test_empty_first_stage_returns_empty_without_loading_model(self) -> None:
        stage = RerankingRetriever(_StubRetriever([]))
        # _model left unset: reaching the model on an empty list would raise here.
        assert stage.search("q", k=5) == []

    def test_returned_scores_are_the_cross_encoder_scores(self) -> None:
        candidates = [(_doc("a.py"), 1.0), (_doc("b.py"), 0.9)]
        stage = _reranker(candidates, [0.3, 0.7])

        ranked = stage.search("q", k=2)

        assert [score for _, score in ranked] == [0.7, 0.3]

    def test_a_k_deeper_than_candidate_depth_is_honoured(self) -> None:
        """Callers over-fetch on purpose; shrinking their k to candidate_depth silently
        returns fewer results than the same call gets with reranking off."""
        candidates = [(_doc(f"f{i}.py"), 1.0) for i in range(80)]
        first_stage = _StubRetriever(candidates)
        stage = RerankingRetriever(first_stage, candidate_depth=50)
        stage._model = _StubModel([1.0] * 80)

        ranked = stage.search("q", k=80)

        assert first_stage.asked_k == 80
        assert len(ranked) == 80

    def test_the_pinned_revision_reaches_the_cross_encoder(self) -> None:
        """A model name alone resolves to the hub's default branch, so the weights behind a
        published retrieval number can change with nothing in the diff to explain it."""
        loaded: dict[str, Any] = {}

        def record_load(model_name: str, device: str | None = None, revision: str | None = None) -> Any:
            loaded["model_name"] = model_name
            loaded["revision"] = revision
            return _StubModel([0.5])

        stage = RerankingRetriever(_StubRetriever([(_doc("a.py"), 1.0)]), revision="abc123")

        with patch("sentence_transformers.CrossEncoder", record_load):
            stage.search("q", k=1)

        assert loaded["revision"] == "abc123"

    def test_the_model_is_loaded_once_under_concurrent_first_queries(self) -> None:
        """An unguarded check-then-set loads a ~2GB model once per racing thread."""
        import threading

        loads = []
        stage = RerankingRetriever(_StubRetriever([(_doc("a.py"), 1.0)]))

        def slow_load(model_name: str, device: str | None = None, revision: str | None = None) -> Any:
            loads.append(model_name)
            time.sleep(0.05)
            return _StubModel([0.5])

        with patch("sentence_transformers.CrossEncoder", slow_load):
            threads = [threading.Thread(target=lambda: stage.search("q", k=1)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(loads) == 1

    def test_k_none_returns_small_default_not_the_whole_candidate_list(self) -> None:
        """A None caller must not get all `candidate_depth` chunks handed to the prompt."""
        from codebase_rag.retrieval.rerank import DEFAULT_TOP_K

        candidates = [(_doc(f"f{i}.py"), 1.0 - i * 0.01) for i in range(50)]
        stage = _reranker(candidates, [1.0 - i * 0.01 for i in range(50)])

        ranked = stage.search("q", k=None)

        assert len(ranked) == DEFAULT_TOP_K
