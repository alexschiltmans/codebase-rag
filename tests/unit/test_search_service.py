"""Unit tests for services/search_service.py."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from codebase_rag.services.search_service import search


def _doc(source: str, start: int, end: int, content: str, repo: str = "repo-a") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "start_line": start, "end_line": end, "repo": repo},
    )


class TestSearch:
    def test_returns_results_within_budget(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (_doc("a.py", 1, 5, "x" * 40), 0.9),
            (_doc("b.py", 1, 5, "y" * 40), 0.8),
        ]

        results = search(retriever, "query", token_budget=1000)

        assert len(results) == 2
        assert all(r.token_estimate > 0 for r in results)

    def test_stops_once_budget_exhausted(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (_doc("a.py", 1, 5, "x" * 40), 0.9),
            (_doc("b.py", 1, 5, "y" * 40), 0.8),
        ]

        # First chunk alone costs ~10 tokens (40 chars / 4); budget only fits one.
        results = search(retriever, "query", token_budget=10)

        assert len(results) == 1
        assert results[0].path == "a.py"

    def test_always_includes_first_result_even_if_it_exceeds_budget(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [(_doc("a.py", 1, 5, "x" * 4000), 0.9)]

        results = search(retriever, "query", token_budget=1)

        assert len(results) == 1

    def test_repo_filter(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (_doc("a.py", 1, 5, "x" * 40, repo="repo-a"), 0.9),
            (_doc("b.py", 1, 5, "y" * 40, repo="repo-b"), 0.8),
        ]

        results = search(retriever, "query", repo="repo-a")

        assert len(results) == 1
        assert results[0].path == "a.py"

    def test_overlapping_chunks_are_deduplicated(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (_doc("a.py", 1, 10, "x" * 40), 0.9),
            (_doc("a.py", 3, 8, "y" * 40), 0.8),  # fully inside the first chunk's range
        ]

        results = search(retriever, "query")

        assert len(results) == 1
        assert results[0].start_line == 1

    def test_non_overlapping_chunks_from_same_file_both_kept(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (_doc("a.py", 1, 5, "x" * 40), 0.9),
            (_doc("a.py", 20, 25, "y" * 40), 0.8),
        ]

        results = search(retriever, "query")

        assert len(results) == 2

    def test_k_limits_result_count(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [(_doc(f"{i}.py", 1, 5, "x" * 40), 1.0 - i * 0.1) for i in range(5)]

        results = search(retriever, "query", k=2)

        assert len(results) == 2
