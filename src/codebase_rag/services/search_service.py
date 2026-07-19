"""Token-budgeted search over the hybrid retriever.

Shared by the HTTP API, and (per the design) the future MCP server and CLI,
so budgeting/dedupe behavior only needs to be correct in one place.
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from codebase_rag.services.token_estimator import estimate_tokens

DEFAULT_TOKEN_BUDGET = 2000


@dataclass
class SearchResult:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    token_estimate: int


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Whether [a_start, a_end] overlaps [b_start, b_end] by more than half of a's lines."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    overlap = max(0, overlap_end - overlap_start + 1)
    a_len = max(1, a_end - a_start + 1)
    return overlap > a_len / 2


def _dedupe(results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """Drop a candidate chunk if an already-selected chunk from the same file
    overlaps it by more than half its lines. Candidates are assumed to already
    be sorted best-first, so the earlier (higher-scoring) chunk wins.
    """
    kept: list[tuple[Document, float]] = []
    kept_ranges: dict[str, list[tuple[int, int]]] = {}

    for doc, score in results:
        path = str(doc.metadata.get("source", ""))
        start = int(doc.metadata.get("start_line", 0))
        end = int(doc.metadata.get("end_line", start))

        ranges = kept_ranges.setdefault(path, [])
        if any(_overlaps(start, end, other_start, other_end) for other_start, other_end in ranges):
            continue

        ranges.append((start, end))
        kept.append((doc, score))

    return kept


def search(
    hybrid_retriever: Any,
    query: str,
    k: int | None = None,
    repo: str | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    tokenizer: Any = None,
) -> list[SearchResult]:
    """Search, dedupe overlapping chunks, and stop once the token budget is spent.

    Args:
        hybrid_retriever: A `HybridRetriever`-like object exposing `.search(query, k)`.
        query: The search query.
        k: Number of candidates to fetch before dedupe/budgeting (fetches extra
            when a repo filter is set, since filtering happens after ranking).
        repo: Optional repo name filter.
        token_budget: Stop adding results once their combined token estimate
            would exceed this.
        tokenizer: Optional tokenizer passed through to the token estimator.
    """
    fetch_k = (k or 10) * (4 if repo else 1)
    raw_results = hybrid_retriever.search(query, k=fetch_k)

    if repo:
        raw_results = [(doc, score) for doc, score in raw_results if doc.metadata.get("repo") == repo]

    deduped = _dedupe(raw_results)
    if k is not None:
        deduped = deduped[:k]

    results: list[SearchResult] = []
    spent = 0
    for doc, score in deduped:
        token_estimate = estimate_tokens(doc.page_content, tokenizer)
        if results and spent + token_estimate > token_budget:
            break
        spent += token_estimate
        results.append(
            SearchResult(
                path=str(doc.metadata.get("source", "")),
                start_line=int(doc.metadata.get("start_line", 0)),
                end_line=int(doc.metadata.get("end_line", 0)),
                score=score,
                snippet=doc.page_content,
                token_estimate=token_estimate,
            )
        )

    return results
