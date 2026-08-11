"""Tests for the CLI module."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from codebase_rag.cli import (
    _format_compact,
    _format_json,
    _load_bm25_retriever,
    _trim_results_by_budget,
    ask_command,
    main,
    query_command,
)
from codebase_rag.retrieval.bm25_search import BM25Retriever


class TestFormatCompact:
    """Tests for compact output formatting."""

    def test_format_compact_single_result(self) -> None:
        """Compact format with one result."""
        results = [("src/app.py", 0.95, "def foo():\n    pass")]
        output = _format_compact(results)
        assert "src/app.py (0.950)" in output
        assert "def foo():" in output

    def test_format_compact_multiple_results(self) -> None:
        """Compact format with multiple results."""
        results = [
            ("src/app.py", 0.95, "snippet1"),
            ("src/lib.py", 0.85, "snippet2"),
        ]
        output = _format_compact(results)
        assert "src/app.py (0.950)" in output
        assert "snippet1" in output
        assert "src/lib.py (0.850)" in output
        assert "snippet2" in output

    def test_format_compact_empty(self) -> None:
        """Compact format with no results."""
        results: list[tuple[Any, ...]] = []
        output = _format_compact(results)
        assert output == ""


class TestFormatJson:
    """Tests for JSON output formatting."""

    def test_format_json_single_result(self) -> None:
        """JSON format with one result."""
        results = [("src/app.py", 0.95, "def foo():\n    pass")]
        output = _format_json(results)
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["path"] == "src/app.py"
        assert data[0]["score"] == 0.95
        assert "def foo():" in data[0]["snippet"]

    def test_format_json_multiple_results(self) -> None:
        """JSON format with multiple results."""
        results = [
            ("src/app.py", 0.95, "snippet1"),
            ("src/lib.py", 0.85, "snippet2"),
        ]
        output = _format_json(results)
        data = json.loads(output)
        assert len(data) == 2
        assert data[0]["path"] == "src/app.py"
        assert data[1]["path"] == "src/lib.py"

    def test_format_json_empty(self) -> None:
        """JSON format with no results."""
        results: list[tuple[Any, ...]] = []
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
        """Query exits 2 (not 1) and prints no stdout when no results are found."""
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = []
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="nonexistent", k=5, format="compact", repo=None, budget=2000)
        result = query_command(args)

        assert result == 2

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_results_compact(self, mock_load_bm25: MagicMock) -> None:
        """Query returns results in compact format."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "src/app.py"}
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
        mock_doc.metadata = {"source": "src/app.py"}
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

    @patch("codebase_rag.cli._load_bm25_retriever")
    @patch("codebase_rag.cli.Config.get_instance")
    @patch("codebase_rag.cli.create_llm_client")
    @patch("codebase_rag.cli.RAGChain")
    @patch("sys.stdout")
    def test_ask_streams_when_interactive(
        self,
        mock_stdout: MagicMock,
        mock_rag_chain_class: MagicMock,
        mock_create_llm: MagicMock,
        mock_config: MagicMock,
        mock_load_bm25: MagicMock,
    ) -> None:
        """Ask streams chunks live when stdout is a terminal."""
        mock_stdout.isatty.return_value = True

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
        printed = "".join(call.args[0] for call in mock_stdout.write.call_args_list if call.args)
        assert "This is a test" in printed

    @patch("codebase_rag.cli._load_bm25_retriever")
    @patch("codebase_rag.cli.Config.get_instance")
    @patch("codebase_rag.cli.create_llm_client")
    @patch("codebase_rag.cli.RAGChain")
    def test_ask_prints_sources_and_still_exits_zero(
        self,
        mock_rag_chain_class: MagicMock,
        mock_create_llm: MagicMock,
        mock_config: MagicMock,
        mock_load_bm25: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-empty sources list must not fail the command.

        Every other ask test sets ``last_result = {"sources": []}``, so the branch that renders
        sources never ran and shipped a ``.metadata`` lookup against the plain dicts that
        ``RAGChain._format_sources`` actually returns. That exited 1 after a correct answer.
        The literal below is that real shape, not a Document.
        """
        mock_load_bm25.return_value = MagicMock()
        mock_create_llm.return_value = MagicMock()

        mock_rag_chain_instance = MagicMock()
        mock_rag_chain_instance.stream.return_value = iter(["the ", "answer"])
        mock_rag_chain_instance.last_result = {
            "sources": [
                {"id": "1", "file_path": "data/repos/demo/src/mod.py", "file_name": "[DEMO] mod.py"},
                {"id": "2", "file_path": "data/repos/demo/README.md", "file_name": "[DEMO] README.md"},
            ]
        }
        mock_rag_chain_class.return_value = mock_rag_chain_instance

        result = ask_command(MagicMock(question="explain the architecture"))

        captured = capsys.readouterr()
        assert result == 0
        assert "the answer" in captured.out
        assert "data/repos/demo/src/mod.py" in captured.err
        assert "data/repos/demo/README.md" in captured.err
        assert "Answer generation failed" not in captured.err


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
        results = [("src/app.py", 0.95, "short")]
        trimmed = _trim_results_by_budget(results, 100, "compact")
        assert len(trimmed) == 1

    def test_trim_results_exceed_budget(self) -> None:
        """Results exceeding budget are trimmed."""
        results = [
            ("src/app.py", 0.95, "a" * 30),
            ("src/lib.py", 0.85, "b" * 100),
        ]
        trimmed = _trim_results_by_budget(results, 60, "compact")
        assert len(trimmed) == 1
        assert trimmed[0][0] == "src/app.py"

    def test_trim_results_empty(self) -> None:
        """Empty results remain empty."""
        results: list[tuple[Any, ...]] = []
        trimmed = _trim_results_by_budget(results, 1000, "compact")
        assert trimmed == []

    def test_trim_results_json_budget_matches_rendered_output(self) -> None:
        """JSON budget is measured against actual JSON output, not compact-shaped text."""
        results = [("src/app.py", 0.95, "x" * 50, "repo")]
        trimmed = _trim_results_by_budget(results, 2000, "json")
        rendered = _format_json(trimmed)
        assert len(rendered) <= 2000


class TestRepoFiltering:
    """Tests for repository filtering."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_repo_filter(self, mock_load_bm25: MagicMock) -> None:
        """Query filters results by the repo metadata tag, not the source path."""
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {
            "source": "data/repos/foo/src/app.py",
            "repo": "foo",
        }
        mock_doc1.page_content = "snippet1"

        mock_doc2 = MagicMock()
        mock_doc2.metadata = {
            "source": "/Users/dev/local-folder/src/lib.py",
            "repo": "bar",
        }
        mock_doc2.page_content = "snippet2"

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.documents = [mock_doc1, mock_doc2]
        mock_bm25_instance.search.return_value = [(mock_doc1, 0.95), (mock_doc2, 0.85)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=5, format="compact", repo="foo", budget=2000)
        result = query_command(args)

        assert result == 0
        mock_bm25_instance.search.assert_called_once_with("test", k=2)

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_repo_filter_no_match(self, mock_load_bm25: MagicMock) -> None:
        """Query exits 2 (not 1) when --repo matches nothing, instead of a bare newline."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "data/repos/foo/src/app.py", "repo": "foo"}
        mock_doc.page_content = "snippet1"

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [(mock_doc, 0.95)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=5, format="compact", repo="nonexistent", budget=2000)
        result = query_command(args)

        assert result == 2

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_repo_filter_over_fetches_before_applying_k(self, mock_load_bm25: MagicMock) -> None:
        """--k limits results returned, not search depth: --repo must not discard matches before --k caps them."""
        foo_docs = []
        for i in range(6):
            doc = MagicMock()
            doc.metadata = {"source": f"data/repos/foo/src/f{i}.py", "repo": "foo"}
            doc.page_content = f"snippet{i}"
            foo_docs.append((doc, 0.9 - i * 0.01))

        bar_doc = MagicMock()
        bar_doc.metadata = {"source": "data/repos/bar/src/g.py", "repo": "bar"}
        bar_doc.page_content = "bar snippet"

        all_results = [(bar_doc, 0.95), *foo_docs]

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.documents = [d for d, _ in all_results]
        mock_bm25_instance.search.return_value = all_results
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=5, format="json", repo="foo", budget=100_000)
        result = query_command(args)

        assert result == 0
        mock_bm25_instance.search.assert_called_once_with("test", k=len(all_results))


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


class TestBudgetValidation:
    """Tests for budget parameter validation."""

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_zero_budget(self, mock_load_bm25: MagicMock) -> None:
        """Query rejects a zero budget instead of trimming away all results silently."""
        args = MagicMock(question="test", k=5, format="compact", repo=None, budget=0)
        result = query_command(args)
        assert result == 1

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_negative_budget(self, mock_load_bm25: MagicMock) -> None:
        """Query rejects a negative budget."""
        args = MagicMock(question="test", k=5, format="compact", repo=None, budget=-1)
        result = query_command(args)
        assert result == 1

    @patch("codebase_rag.cli._load_bm25_retriever")
    def test_query_with_budget_too_small_for_any_result(self, mock_load_bm25: MagicMock) -> None:
        """A positive but too-small budget is 'no results', exit 2, not a usage error."""
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "src/app.py"}
        mock_doc.page_content = "x" * 500

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [(mock_doc, 0.95)]
        mock_load_bm25.return_value = mock_bm25_instance

        args = MagicMock(question="test", k=5, format="compact", repo=None, budget=1)
        result = query_command(args)
        assert result == 2


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

    def test_cli_query_works_from_outside_repo_root(self, tmp_path: Path) -> None:
        """A git hook or CI job invokes the console script from an arbitrary cwd, not the repo root.

        Points CODEBASE_RAG_DATA_DIR at a known fixture index rather than relying on whatever the
        developer's machine happens to have ingested (or hasn't, on a fresh clone or in CI, since
        data/cache is gitignored) — the prior version of this test asserted returncode in (0, 1)
        and "no BM25 index not found", both true whether or not "test" matched anything, and both
        vacuously true on a machine with no index at all.
        """
        # BM25's IDF for a term appearing in exactly 1 of 2 documents comes out to exactly 0
        # (log(N-n+0.5) - log(n+0.5) with N=2, n=1 cancels), so a two-document corpus still
        # scores "test" at 0 and search() excludes it as a non-match. A third, unrelated
        # document is needed for "test" to carry positive discriminative signal.
        data_dir = tmp_path / "data_dir"
        cache_dir = data_dir / "data" / "cache"
        cache_dir.mkdir(parents=True)
        BM25Retriever(
            [
                Document(page_content="function test example", metadata={"source": "f.py"}),
                Document(page_content="unrelated database migration logic", metadata={"source": "g.py"}),
                Document(page_content="another chunk about caching layers", metadata={"source": "h.py"}),
            ]
        ).save_json(cache_dir / "bm25_retriever.json")
        cwd = tmp_path / "somewhere_else"
        cwd.mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "codebase_rag.cli", "query", "test", "--k", "1"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            env={**os.environ, "CODEBASE_RAG_DATA_DIR": str(data_dir)},
        )

        assert result.returncode == 0
        assert "f.py" in result.stdout


class TestLoadBm25RetrieverCwdIndependence:
    """The BM25 index lives under the project's own data/cache, not wherever the CLI runs."""

    def test_finds_index_regardless_of_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data_dir = tmp_path / "data_dir"
        cache_dir = data_dir / "data" / "cache"
        cache_dir.mkdir(parents=True)
        BM25Retriever([Document(page_content="hello", metadata={"source": "f.py"})]).save_json(
            cache_dir / "bm25_retriever.json"
        )
        monkeypatch.setenv("CODEBASE_RAG_DATA_DIR", str(data_dir))
        monkeypatch.chdir(tmp_path)

        retriever = _load_bm25_retriever()

        assert [d.metadata["source"] for d in retriever.documents] == ["f.py"]
