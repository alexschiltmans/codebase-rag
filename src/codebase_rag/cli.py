"""Command-line interface for codebase-rag search and Q&A."""

import argparse
import json
import logging
import sys
from pathlib import Path

from codebase_rag.config import Config
from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Route all logging to stderr, keeping stdout clean for results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_bm25_retriever() -> BM25Retriever:
    """Load BM25 retriever from cache or return empty if missing."""
    cache_dir = Path("data/cache")
    bm25_file = cache_dir / "bm25_retriever.json"

    if bm25_file.exists():
        return BM25Retriever.load_json(bm25_file)

    logger.warning("No BM25 index found at %s. Run `make ingest-default` or use the UI to ingest a repo.", bm25_file)
    raise FileNotFoundError(f"BM25 index not found at {bm25_file}")


def _format_compact(results: list[tuple]) -> str:
    """Format search results in compact text form: path:start-end (score)\\nsnippet."""
    lines = []
    for path, start_line, end_line, score, snippet in results:
        header = f"{path}:{start_line}-{end_line} ({score:.3f})"
        lines.append(header)
        lines.append(snippet)
    return "\n".join(lines)


def _format_json(results: list[tuple]) -> str:
    """Format search results as JSON array."""
    json_results = []
    for path, start_line, end_line, score, snippet in results:
        json_results.append(
            {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "score": float(score),
                "snippet": snippet,
            }
        )
    return json.dumps(json_results, indent=2)


def _count_chars(text: str) -> int:
    """Approximate character count for token budgeting."""
    return len(text)


def _trim_results_by_budget(results: list[tuple], budget: int) -> list[tuple]:
    """Trim results to fit within character budget."""
    trimmed = []
    total_chars = 0

    for result in results:
        result_text = f"{result[0]}:{result[1]}-{result[2]}\n{result[4]}"
        chars = _count_chars(result_text)

        if total_chars + chars <= budget:
            trimmed.append(result)
            total_chars += chars
        else:
            break

    return trimmed


def query_command(args: argparse.Namespace) -> int:
    """Execute the query subcommand."""
    try:
        if args.k <= 0:
            logger.error("--k must be greater than 0")
            return 1

        bm25_retriever = _load_bm25_retriever()

        # Execute search
        search_results = bm25_retriever.search(args.question, k=args.k)
        if not search_results:
            logger.info("No results found for query")
            return 0

        # Convert LangChain Document tuples to our format for formatting
        formatted_results = []
        for doc, score in search_results:
            path = doc.metadata.get("source", "unknown")
            start_line = doc.metadata.get("start_line", 0)
            end_line = doc.metadata.get("end_line", 0)
            snippet = doc.page_content
            formatted_results.append((path, start_line, end_line, score, snippet))

        # Filter by repo if specified
        if args.repo:
            formatted_results = [r for r in formatted_results if r[0].startswith(f"data/repos/{args.repo}/")]

        # Apply budget trimming
        formatted_results = _trim_results_by_budget(formatted_results, args.budget)

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

        # Buffer answer generation to ensure stdout is clean on failure
        answer_text = ""
        try:
            for chunk in rag_chain.stream(args.question):
                answer_text += chunk
        except Exception as e:  # noqa: BLE001
            logger.error("Answer generation failed: %s", e)
            return 1

        # Only write to stdout if generation succeeded
        print(answer_text)  # noqa: T201
        print()  # noqa: T201 newline after answer

        # Print sources from last result to stderr
        if rag_chain.last_result:
            sources = rag_chain.last_result.get("sources", [])
            if sources:
                print("\nSources:", file=sys.stderr)  # noqa: T201
                for source in sources:
                    path = source.metadata.get("source", "unknown")
                    start_line = source.metadata.get("start_line", 0)
                    end_line = source.metadata.get("end_line", 0)
                    print(f"  {path}:{start_line}-{end_line}", file=sys.stderr)  # noqa: T201

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
