"""Unit tests for evals/rescore_testset.py: rescoring, coverage gating, and provenance.

This script writes back into the published arm files under `evals/`, so a silent regression here
does not fail loudly, it just republishes wrong numbers. The coverage and staleness gates are what
stand between an edit to `testset.json` and that outcome, and they are what these tests pin.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.rescore_testset import (
    expected_sources_by_question,
    rescore_bench_result,
    rescore_rerank_result,
)
from evals.testset_provenance import testset_provenance


def _bench_record(rows: list[tuple[str, list[str]]]) -> dict[str, Any]:
    return {
        "hit_rate": 0.0,
        "mrr": 0.0,
        "recall_at_depth": 0.0,
        "per_question": [{"question": q, "category": "factual_lookup", "actual_sources": actual} for q, actual in rows],
    }


class TestExpectedSourcesByQuestion:
    def test_expected_failure_questions_are_dropped(self) -> None:
        testset: list[dict[str, Any]] = [
            {"question": "q1", "sources": ["a.py"]},
            {"question": "q2", "sources": ["b.py"], "expected_failure": True},
        ]
        assert expected_sources_by_question(testset) == {"q1": ["a.py"]}


class TestRescoreBenchResult:
    def test_hit_and_mrr_come_from_the_saved_retrieval_order(self) -> None:
        record = _bench_record([("q1", ["miss.py", "hit.py"]), ("q2", ["nowhere.py"])])

        after = rescore_bench_result(record, {"q1": ["hit.py"], "q2": ["gone.py"]})

        assert after["hit_rate"] == 0.5
        assert after["mrr"] == 0.25
        assert after["covers_testset"] is True

    def test_scoping_a_source_pattern_can_turn_a_hit_into_a_miss(self) -> None:
        """The whole reason this script writes back: a narrower pattern must lower the number."""
        record = _bench_record([("q1", ["tests/cpp/CMakeLists.txt"])])

        loose = rescore_bench_result(record, {"q1": ["CMakeLists.txt"]})
        scoped = rescore_bench_result(record, {"q1": ["power-grid-model/CMakeLists.txt"]})

        assert loose["hit_rate"] == 1.0
        assert scoped["hit_rate"] == 0.0

    def test_questions_the_arm_never_retrieved_for_block_write_back(self) -> None:
        """An arm scored on 29 of 42 questions averages over its own smaller denominator, which
        reads as comparable to a full arm unless the gap is named."""
        record = _bench_record([("q1", ["hit.py"])])

        after = rescore_bench_result(record, {"q1": ["hit.py"], "q2": ["b.py"]})

        assert after["n_scored"] == 1
        assert after["unscored_questions"] == ["q2"]
        assert after["covers_testset"] is False

    def test_questions_dropped_from_the_testset_are_reported_not_scored(self) -> None:
        record = _bench_record([("q1", ["hit.py"]), ("retired", ["x.py"])])

        after = rescore_bench_result(record, {"q1": ["hit.py"]})

        assert after["skipped_questions"] == ["retired"]
        assert after["n_scored"] == 1
        assert after["covers_testset"] is True

    def test_an_arm_with_no_scorable_rows_never_claims_coverage(self) -> None:
        after = rescore_bench_result(_bench_record([]), {"q1": ["hit.py"]})

        assert after["covers_testset"] is False

    def test_category_breakdown_is_recomputed_per_category(self) -> None:
        record = _bench_record([("q1", ["hit.py"]), ("q2", ["nowhere.py"])])
        record["per_question"][1]["category"] = "conceptual"

        after = rescore_bench_result(record, {"q1": ["hit.py"], "q2": ["gone.py"]})

        factual = after["category_breakdown"]["factual_lookup"]
        assert factual == {"hit_rate": 1.0, "mrr": 1.0, "recall_at_depth": 1.0, "n": 1}
        assert after["category_breakdown"]["conceptual"]["hit_rate"] == 0.0


class TestRescoreRerankResult:
    def _record(self) -> dict[str, Any]:
        return {
            "input_depth": 2,
            "output_depth": 1,
            "per_question": [{"question": "q1", "top_sources": ["hit.py"]}],
        }

    def _candidates(self) -> dict[str, list[dict[str, Any]]]:
        return {"q1": [{"source": "miss.py"}, {"source": "hit.py"}]}

    def test_baseline_comes_from_candidates_and_score_from_the_saved_order(self) -> None:
        after = rescore_rerank_result(self._record(), {"q1": ["hit.py"]}, self._candidates())

        assert after["input_recall"] == 1.0
        assert after["baseline_hit_rate"] == 0.0
        assert after["hit_rate"] == 1.0

    def test_a_candidates_file_missing_the_question_is_refused(self) -> None:
        """bench_candidates/ is untracked and regenerable. Pairing a regenerated list with an older
        record silently mixes two runs: the baseline from the new data, the score from the old."""
        after = rescore_rerank_result(self._record(), {"q1": ["hit.py"]}, {"other": [{"source": "x.py"}]})

        assert "error" in after
        assert after["questions_without_usable_candidates"] == ["q1"]

    def test_a_candidates_file_shallower_than_the_arm_is_refused(self) -> None:
        after = rescore_rerank_result(self._record(), {"q1": ["hit.py"]}, {"q1": [{"source": "miss.py"}]})

        assert "error" in after

    def test_partial_coverage_blocks_write_back(self) -> None:
        after = rescore_rerank_result(self._record(), {"q1": ["hit.py"], "q2": ["b.py"]}, self._candidates())

        assert after["unscored_questions"] == ["q2"]
        assert after["covers_testset"] is False


class TestTestsetProvenance:
    def test_size_counts_only_scored_questions(self) -> None:
        testset: list[dict[str, Any]] = [
            {"question": "q1", "sources": ["a.py"]},
            {"question": "q2", "sources": ["b.py"], "expected_failure": True},
        ]
        assert testset_provenance(testset)["testset_size"] == 1

    def test_scoping_a_source_changes_the_hash_at_unchanged_size(self) -> None:
        """The case a size alone cannot catch, and the reason the hash exists."""
        loose = testset_provenance([{"question": "q1", "sources": ["CMakeLists.txt"]}])
        scoped = testset_provenance([{"question": "q1", "sources": ["power-grid-model/CMakeLists.txt"]}])

        assert loose["testset_size"] == scoped["testset_size"]
        assert loose["testset_hash"] != scoped["testset_hash"]

    def test_rewording_an_expected_answer_leaves_arms_comparable(self) -> None:
        base: list[dict[str, Any]] = [{"question": "q1", "sources": ["a.py"], "expected_answer": "one"}]
        reworded: list[dict[str, Any]] = [{"question": "q1", "sources": ["a.py"], "expected_answer": "two"}]

        assert testset_provenance(base) == testset_provenance(reworded)

    def test_question_order_does_not_change_the_hash(self) -> None:
        a: list[dict[str, Any]] = [{"question": "q1", "sources": ["a.py"]}, {"question": "q2", "sources": ["b.py"]}]
        b: list[dict[str, Any]] = list(reversed(a))

        assert testset_provenance(a) == testset_provenance(b)
