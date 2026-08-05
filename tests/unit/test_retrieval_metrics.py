"""Unit tests for evals/retrieval_metrics.py, shared by run_eval.py and bench_retrieval.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.retrieval_metrics import compute_recall_at_depth, compute_retrieval_hit_and_reciprocal_rank


class TestComputeRetrievalHitAndReciprocalRank:
    def test_match_at_rank_two_scores_half_mrr(self) -> None:
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["enum.py"], ["data-model.md", "enum.py", "node.hpp"])
        assert hit == 1
        assert rr == pytest.approx(0.5)

    def test_no_match_scores_zero(self) -> None:
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["enum.py"], ["data-model.md", "node.hpp"])
        assert hit == 0
        assert rr == 0.0


class TestComputeRecallAtDepth:
    def test_match_anywhere_in_list_counts(self) -> None:
        assert compute_recall_at_depth(["enum.py"], ["node.hpp", "data-model.md", "enum.py"]) == 1

    def test_no_match_scores_zero(self) -> None:
        assert compute_recall_at_depth(["enum.py"], ["node.hpp", "data-model.md"]) == 0

    def test_case_insensitive_substring_match(self) -> None:
        assert compute_recall_at_depth(["Enum.py"], ["src/ENUM.PY"]) == 1

    def test_empty_actual_list_scores_zero(self) -> None:
        assert compute_recall_at_depth(["enum.py"], []) == 0
