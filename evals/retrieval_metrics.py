"""Retrieval-only metrics, shared by `run_eval.py` and `bench_retrieval.py`.

Kept in one module so the two entry points cannot silently drift into
disagreeing about what counts as a hit.
"""


def compute_retrieval_hit_and_reciprocal_rank(expected: list[str], actual: list[str]) -> tuple[int, float]:
    """Score one question's retrieval against its expected sources, independent of the generated answer.

    Args:
        expected: Expected source patterns (e.g. `"enum.py"`), matched as
            case-insensitive substrings, same convention as source precision.
        actual: Retrieved document paths, in rank order.

    Returns:
        `(hit, reciprocal_rank)`: hit is 1 if any expected source matches any
        retrieved document, else 0; reciprocal_rank is `1 / (1-based rank of
        the first match)`, or 0 if there is no match.
    """
    expected_lower = [s.lower() for s in expected]
    for rank, src in enumerate(actual, start=1):
        src_lower = src.lower()
        if any(exp in src_lower for exp in expected_lower):
            return 1, 1 / rank
    return 0, 0.0


def compute_recall_at_depth(expected: list[str], actual: list[str]) -> int:
    """Score whether any expected source appears anywhere in the candidate list.

    Same case-insensitive substring convention as `compute_retrieval_hit_and_reciprocal_rank`.

    This is only a distinct number when `actual` is a different list from the one the hit was
    scored over. Given the same list, it returns exactly the hit component, because a hit
    already scans the whole list rather than only its head. It earns its keep when a deep
    candidate list is scored against a shallower output: the recall of the input list is then
    the ceiling on anything a reranker can achieve from it, which the hit rate at output depth
    is not.

    Args:
        expected: Expected source patterns, matched as case-insensitive substrings.
        actual: Retrieved document paths, in any order.

    Returns:
        1 if any expected source matches any retrieved document, else 0.
    """
    expected_lower = [s.lower() for s in expected]
    return int(any(exp in src.lower() for src in actual for exp in expected_lower))
