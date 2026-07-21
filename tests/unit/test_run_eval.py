"""Unit tests for the eval harness's judge-coverage accounting and publish gate.

`evals/run_eval.py` is a script, not a package under `src/`; add the repo
root to `sys.path` so it's importable as `evals.run_eval` (implicit
namespace package), matching the pattern in `tests/e2e/test_ingest_script.py`.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.run_eval import (
    RAGAS_METRIC_NAMES,
    build_ragas_metrics,
    check_coverage_gate,
    compute_custom_metrics,
    compute_ragas_scores_and_coverage,
    compute_retrieval_hit_and_reciprocal_rank,
    publish_retriever_results,
    resolve_judge_timeout_s,
    resolve_max_workers,
    resolve_min_coverage,
    resolve_skip_metrics,
)


class TestComputeRagasScoresAndCoverage:
    """5.1: coverage accounting against a synthetic DataFrame with a known NaN pattern."""

    def test_clean_metric_reports_full_coverage(self):
        df = pd.DataFrame({"faithfulness": [1.0, 0.5, 0.75]})
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert scores["faithfulness"] == pytest.approx(0.75)
        assert coverage["faithfulness"] == {"attempted": 3, "completed": 3, "failed": 0}

    def test_partially_failed_metric_publishes_with_nonzero_failed(self):
        df = pd.DataFrame({"context_recall": [1.0, float("nan"), 0.5, float("nan")]})
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert scores["context_recall"] == pytest.approx(0.75)
        assert coverage["context_recall"] == {"attempted": 4, "completed": 2, "failed": 2}

    def test_all_nan_metric_publishes_none(self):
        df = pd.DataFrame({"answer_relevancy": [float("nan"), float("nan")]})
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert scores["answer_relevancy"] is None
        assert coverage["answer_relevancy"] == {"attempted": 2, "completed": 0, "failed": 2}

    def test_non_metric_columns_are_excluded(self):
        df = pd.DataFrame(
            {
                "user_input": ["q1", "q2"],
                "response": ["a1", "a2"],
                "retrieved_contexts": [["c1"], ["c2"]],
                "reference": ["r1", "r2"],
                "faithfulness": [1.0, 1.0],
            }
        )
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert set(scores) == {"faithfulness"}
        assert set(coverage) == {"faithfulness"}


class TestComputeRetrievalHitAndReciprocalRank:
    """1.3: hit rate / MRR math against hand-built expected/actual source lists."""

    def test_match_at_rank_two_scores_half_mrr(self):
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["enum.py"], ["data-model.md", "enum.py", "node.hpp"])
        assert hit == 1
        assert rr == pytest.approx(0.5)

    def test_match_at_rank_one_scores_full_mrr(self):
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["enum.py"], ["enum.py", "node.hpp"])
        assert hit == 1
        assert rr == pytest.approx(1.0)

    def test_no_match_scores_zero(self):
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["enum.py"], ["data-model.md", "node.hpp"])
        assert hit == 0
        assert rr == 0.0

    def test_match_is_case_insensitive_substring(self):
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(["Enum.py"], ["src/ENUM.PY"])
        assert hit == 1
        assert rr == pytest.approx(1.0)


class TestComputeCustomMetricsRetrieval:
    """1.3: avg_hit_rate/avg_mrr aggregation, including exclusion rules."""

    def _result(self, sources_expected=None, sources_actual=None, expected_failure=False, error=None):
        return {
            "question": "q",
            "answer": "a",
            "keywords": [],
            "sources_expected": sources_expected if sources_expected is not None else [],
            "sources_actual": sources_actual if sources_actual is not None else [],
            "expected_failure": expected_failure,
            "elapsed": 1.0,
            "error": error,
        }

    def test_rank_two_match_averages_to_half_mrr(self):
        results = [self._result(["enum.py"], ["data-model.md", "enum.py"])]
        metrics = compute_custom_metrics(results)
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(0.5)

    def test_no_match_averages_to_zero(self):
        results = [self._result(["enum.py"], ["node.hpp"])]
        metrics = compute_custom_metrics(results)
        assert metrics["avg_hit_rate"] == pytest.approx(0.0)
        assert metrics["avg_mrr"] == pytest.approx(0.0)

    def test_empty_sources_expected_is_excluded_from_average(self):
        results = [
            self._result([], ["node.hpp"]),
            self._result(["enum.py"], ["enum.py"]),
        ]
        metrics = compute_custom_metrics(results)
        # Only the second question (with sources_expected) contributes.
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(1.0)

    def test_expected_failure_question_is_excluded_from_average(self):
        results = [
            self._result(["enum.py"], ["node.hpp"], expected_failure=True),
            self._result(["enum.py"], ["enum.py"]),
        ]
        metrics = compute_custom_metrics(results)
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(1.0)

    def test_errored_result_is_excluded_from_average(self):
        results = [
            self._result(["enum.py"], [], error="boom"),
            self._result(["enum.py"], ["enum.py"]),
        ]
        metrics = compute_custom_metrics(results)
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(1.0)

    def test_no_eligible_questions_defaults_to_zero(self):
        results = [self._result([], [])]
        metrics = compute_custom_metrics(results)
        assert metrics["avg_hit_rate"] == 0
        assert metrics["avg_mrr"] == 0


class TestCheckCoverageGate:
    def test_coverage_below_threshold_fails(self):
        coverage = {"faithfulness": {"attempted": 16, "completed": 6, "failed": 10}}
        assert check_coverage_gate(coverage, min_coverage=0.9) == "faithfulness"

    def test_coverage_at_threshold_passes(self):
        coverage = {"faithfulness": {"attempted": 10, "completed": 9, "failed": 1}}
        assert check_coverage_gate(coverage, min_coverage=0.9) is None

    def test_skipped_metric_absent_from_coverage_cannot_fail_gate(self):
        # A skipped metric never enters `ragas_coverage` in the first place —
        # simulate that by simply not including it here.
        coverage = {"faithfulness": {"attempted": 10, "completed": 10, "failed": 0}}
        assert check_coverage_gate(coverage, min_coverage=0.9) is None

    def test_first_failing_metric_is_reported(self):
        coverage = {
            "faithfulness": {"attempted": 10, "completed": 10, "failed": 0},
            "answer_relevancy": {"attempted": 10, "completed": 0, "failed": 10},
        }
        assert check_coverage_gate(coverage, min_coverage=0.9) == "answer_relevancy"

    def test_missing_requested_metric_is_reported_deterministically(self):
        # 9.2: several requested metrics all missing (wholesale failure) — the
        # reported name must be stable, not whichever the set happens to yield.
        requested = {"faithfulness", "answer_relevancy", "context_recall"}
        first = check_coverage_gate({}, min_coverage=0.9, requested_metrics=requested)
        assert first == min(requested)
        # Same answer regardless of set construction order.
        assert first == check_coverage_gate(
            {}, min_coverage=0.9, requested_metrics={"context_recall", "faithfulness", "answer_relevancy"}
        )

    def test_requested_metric_missing_from_coverage_fails_gate(self):
        # Wholesale judge-phase failure: nothing was measured at all, but
        # something was requested — an empty `coverage` must not read as "pass".
        assert check_coverage_gate({}, min_coverage=0.9, requested_metrics={"faithfulness"}) == "faithfulness"

    def test_empty_coverage_with_no_requested_metrics_passes(self):
        # Nothing was requested (e.g. all metrics explicitly skipped) — an
        # empty `coverage` here really does mean "nothing to check".
        assert check_coverage_gate({}, min_coverage=0.9, requested_metrics=set()) is None


def _publish_args(ragas_coverage, ragas_scores, requested_metrics=None):
    return {
        "results": [{"question": "q", "answer": "a", "error": None}],
        "custom_metrics": {
            "avg_keyword_recall": 1.0,
            "avg_source_precision": 1.0,
            "questions_answered": 1,
            "questions_failed": 0,
            "avg_latency_s": 1.0,
        },
        "ragas_scores": ragas_scores,
        "ragas_coverage": ragas_coverage,
        "requested_metrics": requested_metrics if requested_metrics is not None else set(ragas_coverage),
        "latency_probe_s": 1.23,
        "judge_model_name": "judge-model",
        "is_self_judged": False,
        "min_coverage": 0.9,
    }


class TestPublishRetrieverResults:
    """5.2: gate below threshold writes nothing and exits non-zero; at threshold, writes and exits zero."""

    def test_below_threshold_exits_nonzero_and_writes_nothing(self, tmp_path):
        results_path = tmp_path / "results_vector.json"
        md_path = tmp_path / "results_vector.md"
        results_path.write_text("preexisting json")
        md_path.write_text("preexisting md")
        json_mtime = results_path.stat().st_mtime_ns
        md_mtime = md_path.stat().st_mtime_ns

        ragas_coverage = {"faithfulness": {"attempted": 16, "completed": 1, "failed": 15}}
        ragas_scores = {"faithfulness": 1.0}

        with pytest.raises(SystemExit) as exc_info:
            publish_retriever_results(tmp_path, "vector", **_publish_args(ragas_coverage, ragas_scores))

        assert exc_info.value.code != 0
        assert results_path.read_text() == "preexisting json"
        assert md_path.read_text() == "preexisting md"
        assert results_path.stat().st_mtime_ns == json_mtime
        assert md_path.stat().st_mtime_ns == md_mtime

    def test_at_threshold_writes_and_returns(self, tmp_path):
        ragas_coverage = {"faithfulness": {"attempted": 10, "completed": 9, "failed": 1}}
        ragas_scores = {"faithfulness": 0.95}

        publish_retriever_results(tmp_path, "vector", **_publish_args(ragas_coverage, ragas_scores))

        results_path = tmp_path / "results_vector.json"
        md_path = tmp_path / "results_vector.md"
        assert results_path.exists()
        assert md_path.exists()
        assert '"ragas_coverage"' in results_path.read_text()


class TestWholesaleJudgeFailureGate:
    """8.2: the ragas_error/empty-coverage shape (judge phase failed wholesale) exits non-zero and writes nothing."""

    def test_wholesale_failure_exits_nonzero_and_writes_nothing(self, tmp_path):
        results_path = tmp_path / "results_vector.json"
        md_path = tmp_path / "results_vector.md"
        results_path.write_text("preexisting json")
        md_path.write_text("preexisting md")
        json_mtime = results_path.stat().st_mtime_ns
        md_mtime = md_path.stat().st_mtime_ns

        # Shape returned by run_ragas_evaluation's `except` branch: no counts
        # to threshold at all, even though metrics were requested.
        args = _publish_args(
            ragas_coverage={},
            ragas_scores={"ragas_error": "judge model unreachable"},
            requested_metrics={"faithfulness", "answer_relevancy", "context_recall"},
        )

        with pytest.raises(SystemExit) as exc_info:
            publish_retriever_results(tmp_path, "vector", **args)

        assert exc_info.value.code != 0
        assert results_path.read_text() == "preexisting json"
        assert md_path.read_text() == "preexisting md"
        assert results_path.stat().st_mtime_ns == json_mtime
        assert md_path.stat().st_mtime_ns == md_mtime

    def test_no_metrics_requested_is_not_a_wholesale_failure(self, tmp_path):
        # All metrics explicitly skipped: empty coverage here is legitimate
        # ("nothing asked for"), not a failure to measure — should publish.
        args = _publish_args(ragas_coverage={}, ragas_scores={}, requested_metrics=set())

        publish_retriever_results(tmp_path, "vector", **args)

        assert (tmp_path / "results_vector.json").exists()
        assert (tmp_path / "results_vector.md").exists()


class TestResolveSkipMetrics:
    """5.3: --skip-metric removes a metric from the output and from the gate."""

    def test_no_flag_returns_empty_set(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py"])
        assert resolve_skip_metrics() == set()

    def test_single_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--skip-metric", "context_recall"])
        assert resolve_skip_metrics() == {"context_recall"}

    def test_repeatable_flag(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_eval.py", "--skip-metric", "context_recall", "--skip-metric", "answer_relevancy"],
        )
        assert resolve_skip_metrics() == {"context_recall", "answer_relevancy"}

    def test_skipped_metric_would_be_excluded_from_scores_and_coverage(self):
        # A skipped metric is filtered out of `metrics` before `evaluate()` runs
        # (see `run_ragas_evaluation`), so it never appears as a DataFrame column
        # in the first place — simulate that resulting frame directly.
        df = pd.DataFrame({"faithfulness": [1.0, 1.0]})
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert "context_recall" not in scores
        assert "context_recall" not in coverage
        assert check_coverage_gate(coverage, min_coverage=0.9) is None

    def test_enabled_but_empty_metric_still_fails_gate(self):
        df = pd.DataFrame({"faithfulness": [1.0, 1.0], "context_recall": [float("nan"), float("nan")]})
        scores, coverage = compute_ragas_scores_and_coverage(df)
        assert scores["context_recall"] is None
        assert check_coverage_gate(coverage, min_coverage=0.9) == "context_recall"


class TestRagasMetricNames:
    """9.1: the all-skip guard's name set is derived from the metric list, so it can't drift."""

    def test_names_match_built_metrics(self):
        assert {m.name for m in build_ragas_metrics(None, None)} == RAGAS_METRIC_NAMES

    def test_names_are_nonempty(self):
        assert RAGAS_METRIC_NAMES


class TestResolveConfigValueFlagForms:
    """9.4: `--flag=value` is honored, not silently ignored in favor of the default."""

    def test_equals_form_is_parsed(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--max-workers=4"])
        assert resolve_max_workers() == 4

    def test_space_form_still_works(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--max-workers", "3"])
        assert resolve_max_workers() == 3

    def test_equals_form_for_float(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--min-coverage=0.5"])
        assert resolve_min_coverage() == pytest.approx(0.5)

    def test_default_when_absent(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py"])
        monkeypatch.delenv("RAGAS_MAX_WORKERS", raising=False)
        assert resolve_max_workers() == 1


class TestResolveJudgeTimeout:
    def test_default_is_1200(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py"])
        monkeypatch.delenv("RAGAS_JUDGE_TIMEOUT", raising=False)
        assert resolve_judge_timeout_s() == 1200

    def test_flag_overrides_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--judge-timeout", "1800"])
        assert resolve_judge_timeout_s() == 1800

    def test_equals_form(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--judge-timeout=900"])
        assert resolve_judge_timeout_s() == 900

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_eval.py"])
        monkeypatch.setenv("RAGAS_JUDGE_TIMEOUT", "600")
        assert resolve_judge_timeout_s() == 600
