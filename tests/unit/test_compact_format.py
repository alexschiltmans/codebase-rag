"""Unit tests for services/compact_format.py."""

from codebase_rag.services.compact_format import format_compact
from codebase_rag.services.search_service import SearchResult


class TestFormatCompact:
    def test_single_result_has_header_and_snippet(self) -> None:
        result = SearchResult(
            path="src/foo.py", start_line=10, end_line=20, score=0.876, snippet="def foo(): ...", token_estimate=5
        )

        output = format_compact([result])

        assert output == "src/foo.py:10-20 (0.876)\ndef foo(): ..."
        assert "{" not in output

    def test_multiple_results_are_blank_line_separated(self) -> None:
        results = [
            SearchResult(path="a.py", start_line=1, end_line=2, score=0.5, snippet="a", token_estimate=1),
            SearchResult(path="b.py", start_line=3, end_line=4, score=0.4, snippet="b", token_estimate=1),
        ]

        output = format_compact(results)

        assert output == "a.py:1-2 (0.500)\na\n\nb.py:3-4 (0.400)\nb"

    def test_empty_results(self) -> None:
        assert format_compact([]) == ""
