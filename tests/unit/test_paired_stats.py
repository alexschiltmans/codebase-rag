"""Unit tests for evals/paired_stats.py: the paired tests, the gates, and the effective-size count.

The number this module produces is meant to replace an uncalibrated "three questions is a ranking"
rule, so the arithmetic has to be right and has to be pinned: the exact binomial against the values
the review recorded, the Wilson interval against its published figure, the bootstrap against
reproducibility, and the derived hit rate and MRR against the aggregates an end-to-end file already
publishes. The comparability gates are what stand between a paired test and a comparison of two
different test sets, so they are pinned too.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.paired_stats import (
    IncomparableArmsError,
    MalformedRecordError,
    PairedStatsError,
    compare,
    discriminating_questions,
    exact_binomial_test,
    load_record,
    main,
    normalise_record,
    resolve_record_path,
    wilson_interval,
)

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"
BENCH_RESULTS_DIR = EVALS_DIR / "bench_results"

# A retrieval-only row: (question, category, hit, reciprocal_rank, expected_sources, actual_sources).
Row = tuple[str, str, int, float, list[str], list[str]]


def _e2e_row(row: Row) -> tuple[str, str, list[str], list[str]]:
    """The (question, category, expected_sources, actual_sources) the end-to-end builder takes."""
    return (row[0], row[1], row[4], row[5])


def _retrieval_record(
    name: str, rows: list[Row], *, depth: int = 10, testset_hash: str | None = None
) -> dict[str, Any]:
    """Build a retrieval-only arm record from rows, all on the same test set unless a hash is given."""
    per_question = [
        {
            "question": question,
            "category": category,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "expected_sources": sorted(expected),
            "actual_sources": list(actual),
        }
        for question, category, hit, reciprocal_rank, expected, actual in rows
    ]
    record: dict[str, Any] = {"arm_name": name, "candidate_depth": depth, "per_question": per_question}
    if testset_hash is not None:
        record["testset_hash"] = testset_hash
        record["testset_size"] = len(rows)
    return record


def _end_to_end_record(
    name: str, rows: list[tuple[str, str, list[str], list[str]]], *, fail: set[int] | None = None
) -> dict[str, Any]:
    """Build an end-to-end result record; `fail` is the set of row indices marked expected_failure."""
    fail = fail or set()
    results = [
        {
            "question": question,
            "category": category,
            "sources_expected": expected,
            "sources_actual": actual,
            **({"expected_failure": True} if i in fail else {}),
        }
        for i, (question, category, expected, actual) in enumerate(rows)
    ]
    return {"arm_name": name, "results": results}


def _label(record: dict[str, Any], name: str) -> dict[str, Any]:
    record["label"] = name
    return record


Q1: Row = ("q1", "factual_lookup", 1, 1.0, ["a.py"], ["a.py", "noise.py"])
Q2: Row = ("q2", "conceptual", 0, 0.0, ["b.py"], ["noise.py"])
Q3: Row = ("q3", "how_does_it_work", 1, 0.5, ["c.py"], ["x.py", "c.py"])


class TestExactBinomial:
    @pytest.mark.parametrize(
        ("a_only", "b_only", "expected"),
        [
            (8, 3, 0.2266),
            (3, 1, 0.6250),
            (8, 2, 0.1094),
            (4, 4, 1.0000),
            (2, 0, 0.5000),
            (11, 6, 0.3323),
        ],
    )
    def test_known_values(self, a_only: int, b_only: int, expected: float) -> None:
        result = exact_binomial_test(a_only, b_only)
        assert round(result, 4) == expected

    def test_no_discordant_pairs_is_one_not_a_crash(self) -> None:
        assert exact_binomial_test(0, 0) == 1.0


class TestWilsonInterval:
    def test_published_hit_rate_gives_the_recorded_interval(self) -> None:
        # 38 of 42, the shipped hybrid hit rate the review intervals.
        low, high = wilson_interval(38, 42)
        assert 0.779 <= low <= 0.780
        assert 0.962 <= high <= 0.963

    def test_the_interval_stays_within_zero_and_one(self) -> None:
        # A hit rate of 1.0 on a small sample must not interval above 1.
        low, high = wilson_interval(42, 42)
        assert high <= 1.0
        assert low >= 0.0


class TestNormalisation:
    def test_retrieval_only_carries_the_saved_hit_and_rank(self) -> None:
        record = _retrieval_record("a", [Q1, Q2, Q3])
        normalised = normalise_record(record)
        assert normalised["shape"] == "retrieval-only"
        assert [row["hit"] for row in normalised["rows"]] == [1, 0, 1]
        assert [row["reciprocal_rank"] for row in normalised["rows"]] == [1.0, 0.0, 0.5]

    def test_end_to_end_derives_hit_and_rank_from_sources(self) -> None:
        record = _end_to_end_record("a", [_e2e_row(Q1), _e2e_row(Q2), _e2e_row(Q3)])
        normalised = normalise_record(record)
        assert normalised["shape"] == "end-to-end"
        # q1 retrieves a.py at rank 1, q3 at rank 2, q2 never.
        assert [row["hit"] for row in normalised["rows"]] == [1, 0, 1]
        assert [row["reciprocal_rank"] for row in normalised["rows"]] == [1.0, 0.0, 0.5]

    def test_expected_failure_row_is_dropped(self) -> None:
        # The published aggregates drop it, so the paired test must not score it as a miss for both.
        record = _end_to_end_record("a", [_e2e_row(Q1), _e2e_row(Q2), _e2e_row(Q3)], fail={1})
        normalised = normalise_record(record)
        assert [row["question"] for row in normalised["rows"]] == ["q1", "q3"]

    def test_a_record_that_is_neither_shape_is_refused(self) -> None:
        with pytest.raises(MalformedRecordError):
            normalise_record({"arm_name": "a", "unexpected": []})


class TestReconciliation:
    def test_derived_aggregates_equal_the_published_ones(self) -> None:
        # q4 is an expected failure and is dropped, leaving 3 scored questions: 2 hit, mean rank 0.5.
        rows = [
            ("q1", "factual_lookup", ["a.py"], ["a.py", "b.py"]),
            ("q2", "conceptual", ["b.py"], ["a.py", "b.py"]),
            ("q3", "how_does_it_work", ["c.py"], ["d.py"]),
            ("q4", "conceptual", ["e.py"], ["e.py"]),
        ]
        record = _end_to_end_record("a", rows, fail={3})
        record["custom_metrics"] = {"avg_hit_rate": 2 / 3, "avg_mrr": 0.5}
        normalised = normalise_record(record)
        n = len(normalised["rows"])
        assert n == 3
        assert sum(row["hit"] for row in normalised["rows"]) / n == pytest.approx(2 / 3)
        assert sum(row["reciprocal_rank"] for row in normalised["rows"]) / n == pytest.approx(0.5)
        assert record["custom_metrics"]["avg_hit_rate"] == pytest.approx(2 / 3)
        assert record["custom_metrics"]["avg_mrr"] == pytest.approx(0.5)

    def test_derived_figures_reconcile_with_the_real_bm25_file(self) -> None:
        # The published avg_hit_rate and avg_mrr, reproduced from the stored per-question sources.
        record = load_record(EVALS_DIR / "results_bm25.json")
        normalised = normalise_record(record)
        n = len(normalised["rows"])
        assert sum(row["hit"] for row in normalised["rows"]) / n == pytest.approx(
            record["custom_metrics"]["avg_hit_rate"]
        )
        assert sum(row["reciprocal_rank"] for row in normalised["rows"]) / n == pytest.approx(
            record["custom_metrics"]["avg_mrr"]
        )


class TestComparabilityGates:
    def test_different_question_lists_are_refused(self) -> None:
        rows_b = [Q1, ("qX", "conceptual", 1, 1.0, ["z.py"], ["z.py"]), Q3]
        with pytest.raises(IncomparableArmsError, match="question list"):
            compare(_label(_retrieval_record("a", [Q1, Q2, Q3]), "a"), _label(_retrieval_record("b", rows_b), "b"))

    def test_reordered_question_lists_are_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="question list"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3]), "a"),
                _label(_retrieval_record("b", [Q3, Q2, Q1]), "b"),
            )

    def test_different_expected_sources_are_refused(self) -> None:
        rows_b = [
            ("q1", "factual_lookup", 1, 1.0, ["other.py"], ["other.py"]),
            Q2,
            Q3,
        ]
        with pytest.raises(IncomparableArmsError, match="expected sources"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3]), "a"),
                _label(_retrieval_record("b", rows_b), "b"),
            )

    def test_mixed_record_shapes_are_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="shape"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3]), "a"),
                _label(_end_to_end_record("b", [_e2e_row(Q1), _e2e_row(Q2), _e2e_row(Q3)]), "b"),
            )

    def test_mismatched_testset_hash_is_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="testset_hash"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3], testset_hash="abc"), "a"),
                _label(_retrieval_record("b", [Q1, Q2, Q3], testset_hash="xyz"), "b"),
            )

    def test_missing_testset_hash_does_not_match_a_present_one(self) -> None:
        with pytest.raises(IncomparableArmsError, match="testset_hash"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3]), "a"),
                _label(_retrieval_record("b", [Q1, Q2, Q3], testset_hash="abc"), "b"),
            )

    def test_mismatched_candidate_depth_is_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="candidate_depth"):
            compare(
                _label(_retrieval_record("a", [Q1, Q2, Q3], depth=10), "a"),
                _label(_retrieval_record("b", [Q1, Q2, Q3], depth=50), "b"),
            )

    def test_mismatched_chunk_size_is_refused(self) -> None:
        a = _retrieval_record("a", [Q1, Q2, Q3])
        b = _retrieval_record("b", [Q1, Q2, Q3])
        a["chunk_size"] = 614
        a["chunk_overlap"] = 122
        with pytest.raises(IncomparableArmsError, match="chunk_size"):
            compare(_label(a, "a"), _label(b, "b"))

    def test_both_chunks_absent_passes_the_gate(self) -> None:
        # Arms written before chunking was recorded carry no value; absence equals absence.
        a = _retrieval_record("a", [Q1, Q2, Q3])
        b = _retrieval_record("b", [Q1, Q2, Q3])
        result = compare(_label(a, "a"), _label(b, "b"))
        assert result["n"] == 3


class TestCompare:
    def _pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        a = _retrieval_record(
            "a",
            [
                ("q1", "factual_lookup", 1, 1.0, ["a.py"], ["a.py"]),
                ("q2", "conceptual", 1, 1.0, ["b.py"], ["b.py"]),
                ("q3", "how_does_it_work", 1, 1.0, ["c.py"], ["c.py"]),
                ("q4", "conceptual", 0, 0.0, ["d.py"], ["noise.py"]),
            ],
        )
        b = _retrieval_record(
            "b",
            [
                ("q1", "factual_lookup", 0, 0.0, ["a.py"], ["noise.py"]),
                ("q2", "conceptual", 0, 0.0, ["b.py"], ["noise.py"]),
                ("q3", "how_does_it_work", 0, 0.0, ["c.py"], ["noise.py"]),
                ("q4", "conceptual", 1, 1.0, ["d.py"], ["d.py"]),
            ],
        )
        return _label(a, "a"), _label(b, "b")

    def test_discordant_counts_and_p_value(self) -> None:
        record_a, record_b = self._pair()
        result = compare(record_a, record_b)
        # a wins q1, q2, q3; b wins q4: three discordant one way, one the other, over four questions.
        assert result["discordant_a_only"] == 3
        assert result["discordant_b_only"] == 1
        assert round(result["mcnemar_exact_p"], 4) == 0.625

    def test_identical_pair_yields_an_interval_containing_zero(self) -> None:
        record_a, _ = self._pair()
        identical_a = _label(dict(record_a), "a")
        identical_b = _label(dict(record_a), "b")
        result = compare(identical_a, identical_b, seed=0)
        low, high = result["metric_ci"]
        assert low <= 0.0 <= high
        assert result["metric_ci_excludes_zero"] is False

    def test_bootstrap_is_reproducible_under_a_fixed_seed(self) -> None:
        record_a, record_b = self._pair()
        first = compare(record_a, record_b, seed=1234)
        second = compare(_label(dict(record_a), "a"), _label(dict(record_b), "b"), seed=1234)
        assert first["metric_ci"] == second["metric_ci"]

    def test_a_metric_neither_record_carries_is_refused(self) -> None:
        record_a, record_b = self._pair()
        # Neither record carries recall_at_depth, so requesting it must refuse rather than assume zero.
        with pytest.raises(PairedStatsError, match="recall_at_depth"):
            compare(record_a, record_b, metric="recall_at_depth")


class TestDiscriminatingCount:
    def test_always_hit_always_miss_and_discriminating_are_known_by_construction(self) -> None:
        # q_always is hit or missed by every record; q_move changes outcome across records.
        a = _retrieval_record(
            "a",
            [
                ("q_always", "factual_lookup", 1, 1.0, ["a.py"], ["a.py"]),
                ("q_miss", "conceptual", 0, 0.0, ["b.py"], ["noise.py"]),
                ("q_move", "how_does_it_work", 1, 1.0, ["c.py"], ["c.py"]),
            ],
        )
        b = _retrieval_record(
            "b",
            [
                ("q_always", "factual_lookup", 1, 1.0, ["a.py"], ["a.py"]),
                ("q_miss", "conceptual", 0, 0.0, ["b.py"], ["noise.py"]),
                ("q_move", "how_does_it_work", 0, 0.0, ["c.py"], ["noise.py"]),
            ],
        )
        c = _retrieval_record(
            "c",
            [
                ("q_always", "factual_lookup", 1, 1.0, ["a.py"], ["a.py"]),
                ("q_miss", "conceptual", 0, 0.0, ["b.py"], ["noise.py"]),
                ("q_move", "how_does_it_work", 1, 1.0, ["c.py"], ["c.py"]),
            ],
        )
        result = discriminating_questions([_label(a, "a"), _label(b, "b"), _label(c, "c")])
        assert result["n"] == 3
        assert result["always_hit"] == 1
        assert result["always_miss"] == 1
        assert result["discriminating_count"] == 1
        assert result["discriminating"][0]["category"] == "how_does_it_work"

    def test_discriminating_applies_the_comparability_gate(self) -> None:
        a = _retrieval_record("a", [Q1, Q2, Q3])
        b = _retrieval_record("b", [Q1, Q2, Q3])
        a["chunk_size"] = 614
        with pytest.raises(IncomparableArmsError, match="chunk_size"):
            discriminating_questions([_label(a, "a"), _label(b, "b")])


class TestLoader:
    def test_bare_name_resolves_under_bench_results(self) -> None:
        # The bm25 d10 arm is a stable, tracked record on the 42-question test set.
        path = resolve_record_path("sentence-transformers-all-mpnet-base-v2_bm25_d10")
        assert path == BENCH_RESULTS_DIR / "sentence-transformers-all-mpnet-base-v2_bm25_d10.json"

    def test_a_missing_name_is_not_found_by_stem(self) -> None:
        with pytest.raises(MalformedRecordError, match="no record"):
            resolve_record_path("no-such-arm-xyz")

    def test_a_result_file_is_found_under_evals_when_not_in_bench_results(self) -> None:
        # End-to-end result files live in evals/, not bench_results/.
        path = resolve_record_path("results_bm25")
        assert path == EVALS_DIR / "results_bm25.json"

    def test_a_missing_name_reports_the_directories_it_searched(self) -> None:
        with pytest.raises(MalformedRecordError, match=str(BENCH_RESULTS_DIR)):
            resolve_record_path("no-such-arm-xyz")

    def test_a_path_is_honoured_as_a_path(self) -> None:
        record = load_record(resolve_record_path("sentence-transformers-all-mpnet-base-v2_bm25_d10"))
        assert record["arm_name"] == "sentence-transformers-all-mpnet-base-v2_bm25_d10"


class TestNormaliseValidation:
    def test_a_retrieval_row_missing_hit_is_refused(self) -> None:
        per_question = [
            {
                "question": "q1",
                "category": "c",
                "reciprocal_rank": 1.0,
                "expected_sources": ["a.py"],
                "actual_sources": ["a.py"],
            },
        ]
        record = {"arm_name": "a", "candidate_depth": 10, "per_question": per_question}
        with pytest.raises(MalformedRecordError, match="missing hit"):
            normalise_record(record)

    def test_a_rerank_row_without_hit_is_refused(self) -> None:
        # A rerank_results row carries baseline_hit/reranked_hit and no hit; it is not a scored first-stage arm.
        per_question = [
            {
                "question": "q1",
                "category": "c",
                "baseline_hit": 1,
                "reranked_hit": 1,
                "reciprocal_rank": 1.0,
                "expected_sources": ["a.py"],
                "actual_sources": ["a.py"],
            },
        ]
        record = {"arm_name": "a", "candidate_depth": 10, "per_question": per_question}
        with pytest.raises(MalformedRecordError, match="missing hit"):
            normalise_record(record)

    def test_an_end_to_end_row_with_an_error_is_refused_by_name(self) -> None:
        record = _end_to_end_record(
            "sentence-transformers-all-mpnet-base-v2_hybrid_d10_chunk614",
            [("q1", "c", ["a.py"], ["a.py"])],
        )
        record["results"][0]["error"] = "context window overflow"
        with pytest.raises(PairedStatsError, match="q1"):
            normalise_record(record)

    def test_an_end_to_end_row_with_no_expected_sources_is_skipped(self) -> None:
        record = _end_to_end_record(
            "results_hybrid",
            [
                ("q1", "c", ["a.py"], ["a.py"]),
                ("q2", "c", [], ["a.py"]),
                ("q3", "c", ["c.py"], ["c.py"]),
            ],
        )
        rows = normalise_record(record)["rows"]
        assert [row["question"] for row in rows] == ["q1", "q3"]


class TestCompareEdgeCases:
    def test_zero_resamples_is_refused(self) -> None:
        a = _retrieval_record("a", [Q1, Q2, Q3])
        b = _retrieval_record("b", [Q1, Q2, Q3])
        with pytest.raises(PairedStatsError, match="resamples"):
            compare(_label(a, "a"), _label(b, "b"), "hit", resamples=0, seed=0)

    def test_no_scored_questions_is_refused_before_division(self) -> None:
        a = _end_to_end_record("a", [("q1", "c", ["a.py"], ["a.py"])], fail={0})
        b = _end_to_end_record("b", [("q1", "c", ["a.py"], ["a.py"])], fail={0})
        with pytest.raises(PairedStatsError, match="nothing to compare"):
            compare(_label(a, "a"), _label(b, "b"), "hit", resamples=100, seed=0)


class TestMainCLI:
    def _install_fake_loader(self, monkeypatch: pytest.MonkeyPatch, records: dict[str, Any]) -> None:
        monkeypatch.setattr("evals.paired_stats.resolve_record_path", lambda name: name)
        monkeypatch.setattr("evals.paired_stats.load_record", lambda name: records[name])

    def test_a_single_arm_is_refused_before_indexing(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_loader(
            monkeypatch,
            {"only": _end_to_end_record("only", [(_e2e_row(Q1))])},
        )
        monkeypatch.setattr(sys, "argv", ["paired_stats.py", "only"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "at least two arms" in capsys.readouterr().err

    def test_duplicate_arm_names_compare_distinct_columns(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a_rows = [
            ("q1", "factual_lookup", 1, 1.0, ["a.py"], ["a.py"]),
            ("q2", "conceptual", 1, 1.0, ["b.py"], ["b.py"]),
            ("q3", "how_does_it_work", 1, 1.0, ["c.py"], ["c.py"]),
        ]
        b_rows = [
            ("q1", "factual_lookup", 0, 0.0, ["a.py"], ["noise"]),
            ("q2", "conceptual", 0, 0.0, ["b.py"], ["noise"]),
            ("q3", "how_does_it_work", 0, 0.0, ["c.py"], ["noise"]),
        ]
        a_record = _end_to_end_record("dup", [_e2e_row(r) for r in a_rows])
        b_record = _end_to_end_record("dup", [_e2e_row(r) for r in b_rows])
        monkeypatch.setattr("evals.paired_stats.resolve_record_path", lambda name: name)
        loaded = iter([a_record, b_record])
        monkeypatch.setattr("evals.paired_stats.load_record", lambda name: next(loaded))
        monkeypatch.setattr(sys, "argv", ["paired_stats.py", "dup", "dup", "--metric", "hit"])
        main()
        out = capsys.readouterr().out
        assert "hit:  1.0000 (3/3)" in out
        assert "0.0000 (0/3)" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
