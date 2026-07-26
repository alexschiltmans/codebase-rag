"""Tests for the CLI module."""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.cli import (
    _format_compact,
    _format_json,
    _trim_results_by_budget,
    ask_command,
    main,
    query_command,
)


class TestFormatCompact:
    """Tests for compact output formatting."""

    def test_format_compact_single_result(self) -> None:
        """Compact format with one result."""
        results = [("src/app.py", 10, 20, 0.95, "def foo():\n    pass")]
        output = _format_compact(results)
        assert "src/app.py:10-20 (0.950)" in output
        assert "def foo():" in output

    def test_format_compact_multiple_results(self) -> None:
        """Compact format with multiple results."""
        results = [
            ("src/app.py", 10, 20, 0.95, "snippet1"),
            ("src/lib.py", 30, 40, 0.85, "snippet2"),
        ]
        output = _format_compact(results)
        assert "src/app.py:10-20 (0.950)" in output
        assert "snippet1" in output
        assert "src/lib.py:30-40 (0.850)" in output
        assert "snippet2" in output

    def test_format_compact_empty(self) -> None:
        """Compact format with no results."""
        results: list[tuple] = []
        output = _format_compact(results)
        assert output == ""


class TestFormatJson:
    """Tests for JSON output formatting."""

    def test_format_json_single_result(self) -> None:
        """JSON format with one result."""
        results = [("src/app.py", 10, 20, 0.95, "def foo():\n    pass")]
        output = _format_json(results)
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["path"] == "src/app.py"
        assert data[0]["start_line"] == 10
        assert data[0]["end_line"] == 20
        assert data[0]["score"] == 0.95
        assert "def foo():" in data[0]["snippet"]

    def test_format_json_multiple_results(self) -> None:
        """JSON format with multiple results."""
        results = [
            ("src/app.py", 10, 20, 0.95, "snippet1"),
            ("src/lib.py", 30, 40, 0.85, "snippet2"),
        ]
        output = _format_json(results)
        data = json.loads(output)
        assert len(data) == 2
        assert data[0]["path"] == "src/app.py"
        assert data[1]["path"] == "src/lib.py"

    def test_format_json_empty(self) -> None:
        """JSON format with no results."""
        results: list[tuple] = []
        output = _format_json(results)
        data = json.loads(output)
        assert data == []


class TestQueryCommand:
    """Tests for the query subcommand."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_missing_index(self, mock_load_bm25: MagicMock) -> None:
        """Query fails with exit code 1 when index is missing."""
        mock_load_bm25.side_effect = FileNotFoundError("BM25 index not found")

        args = MagicMock(question="test", k=5, format="compact", repo=None, budget=2000)
        result = query_command(args)

        assert result == 1

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_no_results(self, mock_load_bm25: MagicMock) -> None:
        """Query succeeds with exit code 0 when no results are found."""
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = []
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="nonexistent", k=5, format="compact", repo=None, budget=2000)
        result = query_command(args)

        assert result == 0

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_results_compact(self, mock_load_bm25: MagicMock) -> None:
        """Query returns results in compact format."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "src/app.py", "start_line": 10, "end_line": 20}
        mock_doc.page_content = "def foo():\n    pass"

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [(mock_doc, 0.95)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="function", k=5, format="compact", repo=None, budget=2000)
        result = query_command(args)

        assert result == 0
        mock_bm25_instance.search.assert_called_once_with("function", k=5)

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_results_json(self, mock_load_bm25: MagicMock) -> None:
        """Query returns results in JSON format."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "src/app.py", "start_line": 10, "end_line": 20}
        mock_doc.page_content = "def foo():\n    pass"

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [(mock_doc, 0.95)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="function", k=5, format="json", repo=None, budget=2000)
        result = query_command(args)

        assert result == 0

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_k_parameter(self, mock_load_bm25: MagicMock) -> None:
        """Query respects the k parameter."""
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = []
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=10, format="compact", repo=None, budget=2000)
        query_command(args)

        mock_bm25_instance.search.assert_called_once_with("test", k=10)


class TestAskCommand:
    """Tests for the ask subcommand."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_ask_missing_index(self, mock_load_bm25: MagicMock) -> None:
        """Ask fails with exit code 1 when index is missing."""
        mock_load_bm25.side_effect = FileNotFoundError("BM25 index not found")

        args = MagicMock(question="test")
        result = ask_command(args)

        assert result == 1

    @patch("codebase_rag.cli._load_bm25_retriever")
    @patch("codebase_rag.cli.Config.get_instance")
    @patch("codebase_rag.cli.create_llm_client")
    @patch("codebase_rag.cli.RAGChain")
    def test_ask_generates_answer(
        self,
        mock_rag_chain_class: MagicMock,
        mock_create_llm: MagicMock,
        mock_config: MagicMock,
        mock_load_bm25: MagicMock,
    ) -> None:
        """Ask generates an answer via RAG chain."""
        mock_bm25_instance = MagicMock()
        mock_load_bm25.return_value = mock_bm25_instance

        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_rag_chain_instance = MagicMock()
        mock_rag_chain_instance.stream.return_value = iter(["This ", "is ", "a ", "test"])
        mock_rag_chain_instance.last_result = {"sources": []}
        mock_rag_chain_class.return_value = mock_rag_chain_instance

        args = MagicMock(question="explain the architecture")
        result = ask_command(args)

        assert result == 0
        mock_rag_chain_instance.stream.assert_called_once_with("explain the architecture")


class TestMain:
    """Tests for the main CLI entry point."""

    def test_main_no_args(self) -> None:
        """Main shows help when no subcommand is provided."""
        with patch("sys.argv", ["codebase-rag"]):
            result = main()
            assert result == 1

    def test_main_help_flag(self) -> None:
        """Main shows help with --help flag."""
        with patch("sys.argv", ["codebase-rag", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("codebase_rag.cli.query_command")
    def test_main_query_subcommand(self, mock_query: MagicMock) -> None:
        """Main routes to query_command for query subcommand."""
        mock_query.return_value = 0
        with patch("sys.argv", ["codebase-rag", "query", "test question"]):
            result = main()
            assert result == 0

    @patch("codebase_rag.cli.ask_command")
    def test_main_ask_subcommand(self, mock_ask: MagicMock) -> None:
        """Main routes to ask_command for ask subcommand."""
        mock_ask.return_value = 0
        with patch("sys.argv", ["codebase-rag", "ask", "test question"]):
            result = main()
            assert result == 0


class TestBudgetTrimming:
    """Tests for budget-based result trimming."""

    def test_trim_results_within_budget(self) -> None:
        """Results that fit within budget are kept."""
        results = [("src/app.py", 10, 20, 0.95, "short")]
        trimmed = _trim_results_by_budget(results, 100)
        assert len(trimmed) == 1

    def test_trim_results_exceed_budget(self) -> None:
        """Results exceeding budget are trimmed."""
        results = [
            ("src/app.py", 10, 20, 0.95, "a" * 30),
            ("src/lib.py", 30, 40, 0.85, "b" * 100),
        ]
        trimmed = _trim_results_by_budget(results, 60)
        assert len(trimmed) == 1
        assert trimmed[0][0] == "src/app.py"

    def test_trim_results_empty(self) -> None:
        """Empty results remain empty."""
        results: list[tuple] = []
        trimmed = _trim_results_by_budget(results, 1000)
        assert trimmed == []


class TestRepoFiltering:
    """Tests for repository filtering."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_repo_filter(self, mock_load_bm25: MagicMock) -> None:
        """Query filters results by repository."""
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"source": "data/repos/foo/src/app.py", "start_line": 10, "end_line": 20}
        mock_doc1.page_content = "snippet1"

        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"source": "data/repos/bar/src/lib.py", "start_line": 30, "end_line": 40}
        mock_doc2.page_content = "snippet2"

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [(mock_doc1, 0.95), (mock_doc2, 0.85)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=5, format="compact", repo="foo", budget=2000)
        result = query_command(args)

        assert result == 0
        mock_bm25_instance.search.assert_called_once_with("test", k=5)


class TestKValidation:
    """Tests for k parameter validation."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_negative_k(self, mock_load_bm25: MagicMock) -> None:
        """Query rejects negative k values."""
        args = MagicMock(question="test", k=-1, format="compact", repo=None, budget=2000)
        result = query_command(args)
        assert result == 1

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_zero_k(self, mock_load_bm25: MagicMock) -> None:
        """Query rejects zero k values."""
        args = MagicMock(question="test", k=0, format="compact", repo=None, budget=2000)
        result = query_command(args)
        assert result == 1


class TestCLIIntegration:
    """Integration tests using subprocess to test the actual CLI."""

    def test_cli_query_help(self) -> None:
        """Query subcommand shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "codebase_rag.cli", "query", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Search query" in result.stdout
        assert "--format" in result.stdout

    def test_cli_ask_help(self) -> None:
        """Ask subcommand shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "codebase_rag.cli", "ask", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Question to ask about the codebase" in result.stdout

    def test_cli_no_subcommand(self) -> None:
        """CLI shows help when no subcommand is given."""
        result = subprocess.run(
            [sys.executable, "-m", "codebase_rag.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "positional arguments" in result.stdout or "available commands" in result.stdout.lower()
