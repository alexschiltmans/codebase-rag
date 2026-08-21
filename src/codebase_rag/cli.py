"""Command-line interface for codebase-rag search and Q&A."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from codebase_rag.config import Config
from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.retriever_protocol import RetrieverProtocol

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

# Rule under each result header. Capped so a long path does not draw a rule across a wide terminal.
_RULE_MAX_WIDTH = 80


# Quiet by default: piped output and git hooks do not want per-stage INFO lines around the result.
_VERBOSITY_LEVELS = {0: logging.WARNING, 1: logging.INFO}


def _setup_logging(verbosity: int = 0) -> None:
    """Route all logging to stderr, keeping stdout clean for results."""
    logging.basicConfig(
        level=_VERBOSITY_LEVELS.get(verbosity, logging.DEBUG),
        format="%(name)s: %(message)s",
        stream=sys.stderr,
    )


def _data_dir() -> Path:
    """Directory holding data/cache, overridable via CODEBASE_RAG_DATA_DIR."""
    override = os.environ.get("CODEBASE_RAG_DATA_DIR")
    return Path(override) if override else _PROJECT_ROOT


def _load_bm25_retriever(repos: list[str] | None = None) -> BM25Retriever:
    """Load BM25 retriever from cache or return empty if missing."""
    cache_dir = _data_dir() / "data" / "cache"
    bm25_file = cache_dir / "bm25_retriever.json"

    if bm25_file.exists():
        return BM25Retriever.load_json(bm25_file, repos=repos)

    logger.warning("No BM25 index found at %s. Run `make ingest-default` or use the UI to ingest a repo.", bm25_file)
    raise FileNotFoundError(f"BM25 index not found at {bm25_file}")


def _create_llm(config: Config) -> Any:
    """Build the generation client. Shared so `ask` and the rewrite stage cannot configure it apart."""
    return create_llm_client(
        model_name=config.llm_model_name,
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_tokens=1024,
        timeout=120,
        num_ctx=config.ollama_num_ctx,
    )


def _build_retriever(config: Config, repos: list[str] | None = None, llm: Any = None) -> RetrieverProtocol:
    """Build the configured retrieval stack, scoped to `repos` if given.

    The CLI reads the same settings the app and the API do, so `RETRIEVER`, `RERANK_ENABLED`
    and `REWRITE_ENABLED` each mean one thing across all three rather than moving some of them.

    Two things are built lazily, because this command is meant for git hooks and CI where the
    keyword path answers in milliseconds. The vector store is constructed inside the callable,
    so the default never pays for a Qdrant client and the embedding model load behind it. The
    generation client is built only when the rewrite stage is enabled and the caller has not
    already got one; reranking needs no model client, and `query` has no other use for one.

    Args:
        config: Supplies the retriever choice and the stage flags.
        repos: Optional repository restriction, pushed into both rankers.
        llm: An existing generation client to hand the rewrite stage, if the caller has one.
    """
    from codebase_rag.retrieval.retrieval_stack import apply_stages, select_base_retriever

    def vector_retriever() -> RetrieverProtocol:
        from codebase_rag.database.qdrant_store import QdrantStore
        from codebase_rag.retrieval.vector_search import VectorRetriever, resolve_score_threshold

        store = QdrantStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
        )
        return VectorRetriever(store, score_threshold=resolve_score_threshold(config.embedding_model), repos=repos)

    base = select_base_retriever(config, _load_bm25_retriever(repos), vector_retriever)
    if llm is None and config.rewrite_enabled:
        llm = _create_llm(config)
    return apply_stages(base, config, llm)


def _repo_relative(path: str, repo: Any) -> str:
    """Path as it reads inside its repo checkout, or unchanged when the checkout root is not in it."""
    if not repo:
        return path
    _, separator, tail = path.partition(f"/{repo}/")
    return f"{repo}/{tail}" if separator and tail else path


def _format_compact(results: list[tuple[Any, ...]]) -> str:
    """Format search results as a numbered header, the repo-relative path, a rule, then the snippet."""
    headers = []
    for index, (path, score, _snippet, *rest) in enumerate(results, start=1):
        location = _repo_relative(str(path), rest[0] if rest else None)
        headers.append((f"[{index}] {location.rsplit('/', 1)[-1]}  ({score:.3f})", f"    {location}"))

    # One rule width for the whole output rather than per result, so the blocks line up as a list.
    width = min(max((len(line) for pair in headers for line in pair), default=0), _RULE_MAX_WIDTH)

    blocks = []
    for (title, location_line), (_path, _score, snippet, *_rest) in zip(headers, results, strict=True):
        # Trim blank lines off each end, newline-only at the front so the first line keeps its indent.
        body = str(snippet).lstrip("\n").rstrip()
        blocks.append("\n".join([title, location_line, "\u2500" * width, body]))
    return "\n\n".join(blocks)


def _format_json(results: list[tuple[Any, ...]]) -> str:
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


def _trim_results_by_budget(results: list[tuple[Any, ...]], budget: int, output_format: str) -> list[tuple[Any, ...]]:
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

        # --repo is a restriction on the retriever, not a filter over its output. The
        # retriever therefore returns k in-scope results directly; this used to rank the
        # entire corpus and filter afterwards, which is affordable against an in-memory
        # keyword index and is not against a vector store.
        repos = [args.repo] if args.repo else None
        retriever = _build_retriever(Config.get_instance(), repos)

        search_results = retriever.search(args.question, k=args.k)
        if not search_results:
            # Warning, not info: with the default level quiet, info leaves exit 2 as the only signal.
            if args.repo:
                logger.warning("No results found for repo '%s'", args.repo)
            else:
                logger.warning("No results found for query")
            return 2

        # Convert LangChain Document tuples to our format for formatting
        formatted_results = []
        for doc, score in search_results:
            path = doc.metadata.get("source", "unknown")
            snippet = doc.page_content
            repo = doc.metadata.get("repo")
            formatted_results.append((path, score, snippet, repo))

        # Apply budget trimming
        formatted_results = _trim_results_by_budget(formatted_results, args.budget, args.format)
        if not formatted_results:
            logger.error("--budget %d is too small to hold any result", args.budget)
            return 2

        # Format and output results
        output = _format_json(formatted_results) if args.format == "json" else _format_compact(formatted_results)
        print(output)
        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error("Query failed: %s", e)
        return 1


def ask_command(args: argparse.Namespace) -> int:
    """Execute the ask subcommand."""
    try:
        config = Config.get_instance()
        # Built before the retriever so the rewrite stage shares this one client rather than
        # standing up a second identical one.
        llm = _create_llm(config)
        retriever = _build_retriever(config, llm=llm)

        rag_chain = RAGChain(
            retriever=retriever,
            llm=llm,
            use_conversation_memory=False,
            prompt_budget_chars=llm.prompt_budget_chars,
        )

        if sys.stdout.isatty():
            # Interactive: stream live so the user isn't staring at nothing.
            try:
                for chunk in rag_chain.stream(args.question):
                    print(chunk, end="", flush=True)
            except Exception as e:
                logger.error("Answer generation failed: %s", e)
                return 1
            print()
        else:
            # Piped: buffer so a failure mid-generation leaves stdout empty.
            answer_text = ""
            try:
                for chunk in rag_chain.stream(args.question):
                    answer_text += chunk
            except Exception as e:
                logger.error("Answer generation failed: %s", e)
                return 1
            print(answer_text)

        # Sources go to stderr. RAGChain._format_sources yields plain dicts, so these are subscripts.
        if rag_chain.last_result:
            sources = rag_chain.last_result.get("sources", [])
            # One line per file: several retrieved chunks usually come from the same file, so the list repeats paths.
            seen: set[str] = set()
            paths = []
            for source in sources:
                path = source.get("file_path", "unknown")
                if path not in seen:
                    seen.add(path)
                    paths.append(path)

            if paths:
                print("\nSources:", file=sys.stderr)
                for path in paths:
                    print(f"  {path}", file=sys.stderr)

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return 1


def main() -> int:
    """Main CLI entry point."""
    # SUPPRESS, not 0: the flag sits on every subparser too, and a real default there would clobber it.
    verbosity_parser = argparse.ArgumentParser(add_help=False)
    verbosity_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="Show progress logging on stderr (-vv for debug)",
    )

    parser = argparse.ArgumentParser(
        description="Query and explore codebases using retrieval-augmented generation",
        prog="codebase-rag",
        parents=[verbosity_parser],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Search for code snippets", parents=[verbosity_parser])
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
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question and get a generated answer", parents=[verbosity_parser]
    )
    ask_parser.add_argument("question", help="Question to ask about the codebase")
    ask_parser.set_defaults(func=ask_command)

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", 0))

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    sys.exit(main())
