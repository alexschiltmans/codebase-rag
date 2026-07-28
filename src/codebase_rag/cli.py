"""Command-line interface for codebase-rag search and Q&A."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from codebase_rag.config import Config
from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever

logger = logging.getLogger(__name__)

# The BM25 index lives under this project's own data/cache, not wherever the
# CLI happens to be invoked from. Anchoring to cwd broke the console script
# for its stated use case (git hooks, CI, scripts) the moment it ran from
# anywhere but the repo root: src/codebase_rag/cli.py -> repo root.
#
# This is right for an editable install and for the container's
# PYTHONPATH=/app/src, but wrong for a plain non-editable `pip install .`:
# __file__ then resolves under site-packages/, and three parents up lands in
# lib/python3.12, not the project. CODEBASE_RAG_DATA_DIR overrides this for
# that case (and gives tests a fixture data dir to point at).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_logging() -> None:
    """Route all logging to stderr, keeping stdout clean for results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def _data_dir() -> Path:
    """Directory holding data/cache, overridable via CODEBASE_RAG_DATA_DIR."""
    override = os.environ.get("CODEBASE_RAG_DATA_DIR")
    return Path(override) if override else _PROJECT_ROOT


def _load_bm25_retriever() -> BM25Retriever:
    """Load BM25 retriever from cache or return empty if missing."""
    cache_dir = _data_dir() / "data" / "cache"
    bm25_file = cache_dir / "bm25_retriever.json"

    if bm25_file.exists():
        return BM25Retriever.load_json(bm25_file)

    logger.warning("No BM25 index found at %s. Run `make ingest-default` or use the UI to ingest a repo.", bm25_file)
    raise FileNotFoundError(f"BM25 index not found at {bm25_file}")


def _format_compact(results: list[tuple]) -> str:
    """Format search results in compact text form: path (score)\\nsnippet."""
    lines = []
    for path, score, snippet, *_ in results:
        header = f"{path} ({score:.3f})"
        lines.append(header)
        lines.append(snippet)
    return "\n".join(lines)


def _format_json(results: list[tuple]) -> str:
    """Format search results as JSON array."""
    json_results = []
    for path, score, snippet, *_ in results:
        json_results.append(
            {
                "path": path,
                "score": float(score),
                "snippet": snippet,
            }
        )
    return json.dumps(json_results, indent=2)


def _trim_results_by_budget(results: list[tuple], budget: int, output_format: str) -> list[tuple]:
    """Trim results to fit within the character budget of the requested output format.

    Rendered length grows monotonically with the number of results, so the largest
    fitting prefix is found by binary search: O(log n) renders instead of re-rendering
    every candidate prefix from scratch.
    """
    render = _format_json if output_format == "json" else _format_compact

    lo, hi = 0, len(results)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render(results[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1

    return results[:lo]


def query_command(args: argparse.Namespace) -> int:
    """Execute the query subcommand.

    Exit codes: 0 success, 1 error (bad flags, missing index, unreachable backend),
    2 no results (query, --repo filter, or --budget legitimately produced nothing
    to print). Callers that treat "no results" as acceptable, such as a commit-msg
    git hook, should check for 2 specifically rather than treating any non-zero
    exit as a hard failure.
    """
    try:
        if args.k <= 0:
            logger.error("--k must be greater than 0")
            return 1
        if args.budget <= 0:
            logger.error("--budget must be greater than 0")
            return 1

        bm25_retriever = _load_bm25_retriever()

        # When filtering by repo, over-fetch across the whole index so --repo
        # narrows the result set rather than truncating it before the filter runs.
        search_k = len(bm25_retriever.documents) if args.repo else args.k
        search_results = bm25_retriever.search(args.question, k=search_k)
        if not search_results:
            logger.info("No results found for query")
            return 2

        # Convert LangChain Document tuples to our format for formatting
        formatted_results = []
        for doc, score in search_results:
            path = doc.metadata.get("source", "unknown")
            snippet = doc.page_content
            repo = doc.metadata.get("repo")
            formatted_results.append((path, score, snippet, repo))

        # Filter by repo if specified, then cap back to k (search results are already
        # sorted by score, so the top k of the filtered set is the correct top k).
        if args.repo:
            formatted_results = [r for r in formatted_results if r[3] == args.repo][: args.k]
            if not formatted_results:
                logger.info("No results found for repo '%s'", args.repo)
                return 2

        # Apply budget trimming
        formatted_results = _trim_results_by_budget(formatted_results, args.budget, args.format)
        if not formatted_results:
            logger.error("--budget %d is too small to hold any result", args.budget)
            return 2

        # Format and output results
        output = _format_json(formatted_results) if args.format == "json" else _format_compact(formatted_results)
        print(output)  # noqa: T201
        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:  # noqa: BLE001
        logger.error("Query failed: %s", e)
        return 1


def ask_command(args: argparse.Namespace) -> int:
    """Execute the ask subcommand."""
    try:
        config = Config.get_instance()
        bm25_retriever = _load_bm25_retriever()

        # Initialize LLM and RAG chain
        llm = create_llm_client(
            model_name=config.llm_model_name,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            timeout=120,
            num_ctx=config.ollama_num_ctx,
        )

        rag_chain = RAGChain(
            retriever=bm25_retriever,
            llm=llm,
            use_conversation_memory=False,
            prompt_budget_chars=llm.prompt_budget_chars,
        )

        if sys.stdout.isatty():
            # Interactive: stream live so the user isn't staring at nothing.
            try:
                for chunk in rag_chain.stream(args.question):
                    print(chunk, end="", flush=True)  # noqa: T201
            except Exception as e:  # noqa: BLE001
                logger.error("Answer generation failed: %s", e)
                return 1
            print()  # noqa: T201
        else:
            # Piped: buffer so a failure mid-generation leaves stdout empty.
            answer_text = ""
            try:
                for chunk in rag_chain.stream(args.question):
                    answer_text += chunk
            except Exception as e:  # noqa: BLE001
                logger.error("Answer generation failed: %s", e)
                return 1
            print(answer_text)  # noqa: T201

        # Print sources from last result to stderr. RAGChain._format_sources yields plain
        # dicts, not Documents, so these are subscripts rather than .metadata lookups.
        if rag_chain.last_result:
            sources = rag_chain.last_result.get("sources", [])
            if sources:
                print("\nSources:", file=sys.stderr)  # noqa: T201
                for source in sources:
                    path = source.get("file_path", "unknown")
                    print(f"  {path}", file=sys.stderr)  # noqa: T201

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:  # noqa: BLE001
        logger.error("Answer generation failed: %s", e)
        return 1


def main() -> int:
    """Main CLI entry point."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Query and explore codebases using retrieval-augmented generation",
        prog="codebase-rag",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Search for code snippets")
    query_parser.add_argument("question", help="Search query")
    query_parser.add_argument("--repo", default=None, help="Filter results to a specific repository")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results to return (default: 5)")
    query_parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Character budget for results (default: 2000)",
    )
    query_parser.add_argument(
        "--format",
        choices=["compact", "json"],
        default="compact",
        help="Output format (default: compact)",
    )
    query_parser.set_defaults(func=query_command)

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a question and get a generated answer")
    ask_parser.add_argument("question", help="Question to ask about the codebase")
    ask_parser.set_defaults(func=ask_command)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    sys.exit(main())
