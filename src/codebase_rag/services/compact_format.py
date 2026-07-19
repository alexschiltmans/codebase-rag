"""Shared compact serializer for search results.

Used by `/search?format=compact` and (per the design) the future MCP tool
results and CLI, so terseness fixes propagate everywhere.
"""

from codebase_rag.services.search_service import SearchResult


def format_compact(results: list[SearchResult]) -> str:
    """Render results as newline-delimited `path:start-end (score)` + snippet blocks."""
    blocks = []
    for result in results:
        header = f"{result.path}:{result.start_line}-{result.end_line} ({result.score:.3f})"
        blocks.append(f"{header}\n{result.snippet}")
    return "\n\n".join(blocks)
