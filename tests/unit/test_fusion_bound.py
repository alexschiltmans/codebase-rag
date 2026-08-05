"""Unit tests for evals/fusion_bound.py: the union bound and the gates that keep it meaningful.

The number this script produces is published as a ceiling on a retriever combination nobody built,
which means no later measurement will contradict it. The comparability gates are what stand between a
correct bound and a union taken across two different test sets, so they are pinned here alongside the
arithmetic.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals import fusion_bound as fusion_bound_module
from evals.fusion_bound import (
    IncomparableArmsError,
    MalformedArmRecordError,
    fusion_bound,
    main,
)


def _arm(
    name: str,
    rows: list[tuple[str, list[str], list[str]]],
    *,
    depth: int = 10,
    testset_hash: str | None = "abc123",
    testset_size: int | None = 3,
    repositories: list[str] | None = None,
) -> dict[str, Any]:
    """Build an arm record from (question, expected_sources, actual_sources) rows."""
    record: dict[str, Any] = {
        "arm_name": name,
        "candidate_depth": depth,
        "per_question": [
            {"question": q, "expected_sources": expected, "actual_sources": actual} for q, expected, actual in rows
        ],
    }
    if testset_hash is not None:
        record["testset_hash"] = testset_hash
    if testset_size is not None:
        record["testset_size"] = testset_size
    if repositories is not None:
        record["ingested_repositories"] = repositories
    return record


ROWS_A = [
    ("q1", ["alpha.py"], ["alpha.py", "noise.py"]),
    ("q2", ["beta.py"], ["noise.py"]),
    ("q3", ["gamma.py"], ["noise.py"]),
]
ROWS_B = [
    ("q1", ["alpha.py"], ["noise.py"]),
    ("q2", ["beta.py"], ["beta.py"]),
    ("q3", ["gamma.py"], ["noise.py"]),
]


class TestUnionArithmetic:
    def test_union_of_an_arm_with_itself_is_that_arms_recall(self) -> None:
        result = fusion_bound([_arm("a", ROWS_A), _arm("a-copy", ROWS_A)])
        assert result["components"]["a"]["hits"] == 1
        assert result["union_hits"] == 1
        assert result["headroom_questions"] == 0
        assert result["questions_gained_by_union"] == {"a": [], "a-copy": []}

    def test_question_hit_by_exactly_one_component_is_counted_once_in_the_union(self) -> None:
        result = fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B)])
        assert result["components"]["a"]["hits"] == 1
        assert result["components"]["b"]["hits"] == 1
        assert result["union_hits"] == 2
        assert result["headroom_questions"] == 1

    def test_question_hit_by_neither_component_stays_a_miss(self) -> None:
        result = fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B)])
        assert result["union_hits"] < result["n_questions"]
        assert "q3" not in result["questions_gained_by_union"]

    def test_headroom_is_measured_against_the_best_component_not_the_worst(self) -> None:
        # b hits q1 and q2; a hits only q1, so the union gains nothing over b.
        rows_b_strong = [
            ("q1", ["alpha.py"], ["alpha.py"]),
            ("q2", ["beta.py"], ["beta.py"]),
            ("q3", ["gamma.py"], ["noise.py"]),
        ]
        result = fusion_bound([_arm("a", ROWS_A), _arm("b", rows_b_strong)])
        assert result["best_components"] == ["b"]
        assert result["union_hits"] == 2
        assert result["headroom_questions"] == 0

    def test_scoring_ignores_the_saved_hit_field(self) -> None:
        # A stale `hit` of 1 on a question the arm plainly missed must not reach the output.
        arm = _arm("a", ROWS_A)
        for row in arm["per_question"]:
            row["hit"] = 1
        result = fusion_bound([arm, _arm("b", ROWS_B)])
        assert result["components"]["a"]["hits"] == 1


class TestTiedComponents:
    def test_every_arm_tied_for_best_is_reported_as_best(self) -> None:
        result = fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B)])
        assert result["best_components"] == ["a", "b"]
        assert result["best_component_hits"] == 1

    def test_each_tied_best_gets_the_questions_the_union_gains_over_it(self) -> None:
        result = fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B)])
        assert result["questions_gained_by_union"] == {"a": ["q2"], "b": ["q1"]}

    def test_attribution_does_not_depend_on_argument_order(self) -> None:
        # A tie broken by argument order would name a different question as the source of the
        # headroom depending on which arm was typed first, and that question gets published.
        forward = fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B)])
        reversed_ = fusion_bound([_arm("b", ROWS_B), _arm("a", ROWS_A)])
        assert forward["questions_gained_by_union"] == reversed_["questions_gained_by_union"]
        assert sorted(forward["best_components"]) == sorted(reversed_["best_components"])


class TestComparabilityGates:
    def test_mismatched_testset_hash_is_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="testset_hash"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B, testset_hash="different")])

    def test_missing_testset_hash_does_not_match_a_present_one(self) -> None:
        with pytest.raises(IncomparableArmsError, match="testset_hash"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B, testset_hash=None, testset_size=None)])

    def test_two_arms_without_provenance_are_allowed(self) -> None:
        # The 29-question historical arms carry no provenance fields at all, and the published table
        # needs their union; the question-list gate is what keeps them from being fused across sets.
        result = fusion_bound(
            [
                _arm("a", ROWS_A, testset_hash=None, testset_size=None),
                _arm("b", ROWS_B, testset_hash=None, testset_size=None),
            ]
        )
        assert result["union_hits"] == 2

    def test_differing_question_lists_are_refused(self) -> None:
        # Same declared provenance, different questions: the provenance gates pass and the question
        # list is the only thing left to catch it.
        rows_extra = [*ROWS_B, ("q4", ["delta.py"], ["delta.py"])]
        with pytest.raises(IncomparableArmsError, match="question list"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", rows_extra)])

    def test_reordered_question_lists_are_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="question list"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", list(reversed(ROWS_B)))])

    def test_mismatched_candidate_depth_is_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="candidate_depth"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", ROWS_B, depth=50)])

    def test_disagreeing_expected_sources_are_refused(self) -> None:
        rows_reground = [("q1", ["alpha/alpha.py"], ["noise.py"]), *ROWS_B[1:]]
        with pytest.raises(IncomparableArmsError, match="expected sources"):
            fusion_bound([_arm("a", ROWS_A), _arm("b", rows_reground)])

    def test_reordered_expected_sources_are_the_same_ground_truth(self) -> None:
        # The scorer matches any pattern against any document and the provenance hash sorts before
        # hashing, so source order carries no meaning and must not split two comparable arms.
        rows_multi_a = [("q1", ["alpha.py", "beta.py"], ["alpha.py", "beta.py"]), *ROWS_A[1:]]
        rows_multi_b = [("q1", ["beta.py", "alpha.py"], ["noise.py"]), *ROWS_B[1:]]
        result = fusion_bound([_arm("a", rows_multi_a), _arm("b", rows_multi_b)])
        assert result["union_hits"] == 2

    def test_mismatched_ingested_repositories_are_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="ingested_repositories"):
            fusion_bound(
                [
                    _arm("a", ROWS_A, repositories=["power-grid-model"]),
                    _arm("b", ROWS_B, repositories=["power-grid-model", "other-repo"]),
                ]
            )

    def test_reordered_ingested_repositories_are_the_same_corpus(self) -> None:
        result = fusion_bound(
            [
                _arm("a", ROWS_A, repositories=["one", "two"]),
                _arm("b", ROWS_B, repositories=["two", "one"]),
            ]
        )
        assert result["union_hits"] == 2

    def test_a_single_arm_is_refused(self) -> None:
        with pytest.raises(IncomparableArmsError, match="at least two arms"):
            fusion_bound([_arm("a", ROWS_A)])

    def test_the_same_arm_twice_is_refused(self) -> None:
        # Keying components by name would collapse these into one column and report the arm's own
        # recall as a union with zero headroom, which reads as "no ensemble gain".
        with pytest.raises(IncomparableArmsError, match="more than once"):
            fusion_bound([_arm("a", ROWS_A), _arm("a", ROWS_A)])


class TestMalformedRecords:
    def test_a_record_without_candidate_depth_is_refused(self) -> None:
        arm = _arm("a", ROWS_A)
        del arm["candidate_depth"]
        with pytest.raises(MalformedArmRecordError, match="candidate_depth"):
            fusion_bound([arm, _arm("b", ROWS_B)])

    def test_rows_without_ground_truth_are_refused(self) -> None:
        # Reranked and candidate records live one directory from the arms and lack these fields.
        arm = _arm("a", ROWS_A)
        for row in arm["per_question"]:
            del row["expected_sources"]
        with pytest.raises(MalformedArmRecordError, match="expected_sources"):
            fusion_bound([arm, _arm("b", ROWS_B)])

    def test_a_record_without_per_question_rows_is_refused(self) -> None:
        arm = _arm("a", ROWS_A)
        arm["per_question"] = []
        with pytest.raises(MalformedArmRecordError, match="per_question"):
            fusion_bound([arm, _arm("b", ROWS_B)])


class TestCommandLine:
    def _write(self, tmp_path: Path, name: str, record: dict[str, Any]) -> str:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({k: v for k, v in record.items() if k != "arm_name"}))
        return str(path)

    def test_incomparable_arms_exit_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = self._write(tmp_path, "a", _arm("a", ROWS_A))
        b = self._write(tmp_path, "b", _arm("b", ROWS_B, testset_hash="different"))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", a, b])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "testset_hash" in capsys.readouterr().err

    def test_missing_arm_record_exits_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", "no-such-arm", "also-missing"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_a_path_that_does_not_exist_is_not_resolved_by_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Stems repeat across bench_results/, bench_candidates/ and rerank_results/, so a mistyped
        # directory must not quietly load a different arm that happens to share the name.
        monkeypatch.setattr(fusion_bound_module, "BENCH_RESULTS_DIR", tmp_path)
        self._write(tmp_path, "a", _arm("a", ROWS_A))
        wrong_directory = str(tmp_path / "elsewhere" / "a.json")
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", wrong_directory, str(tmp_path / "a.json")])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "no arm record at" in capsys.readouterr().err

    def test_the_same_file_twice_exits_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = self._write(tmp_path, "a", _arm("a", ROWS_A))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", a, a])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "more than once" in capsys.readouterr().err

    def test_two_directories_sharing_a_stem_are_labelled_apart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        a = self._write(left, "arm", _arm("arm", ROWS_A))
        b = self._write(right, "arm", _arm("arm", ROWS_B))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", a, b, "--json"])
        main()
        result = json.loads(capsys.readouterr().out)
        assert sorted(result["components"]) == ["left/arm", "right/arm"]
        assert result["union_hits"] == 2

    def test_a_list_shaped_file_exits_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `bench_candidates/` dumps are top-level JSON lists and share stems with the arms.
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps([{"question": "q1"}]))
        a = self._write(tmp_path, "a", _arm("a", ROWS_A))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", str(dump), a])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "not a benchmark arm record" in capsys.readouterr().err

    def test_a_record_missing_candidate_depth_exits_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        record = _arm("a", ROWS_A)
        del record["candidate_depth"]
        a = self._write(tmp_path, "a", record)
        b = self._write(tmp_path, "b", _arm("b", ROWS_B))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", a, b])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "candidate_depth" in capsys.readouterr().err

    def test_comparable_arms_report_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = self._write(tmp_path, "a", _arm("a", ROWS_A))
        b = self._write(tmp_path, "b", _arm("b", ROWS_B))
        monkeypatch.setattr(sys, "argv", ["fusion_bound.py", a, b, "--json"])
        main()
        result = json.loads(capsys.readouterr().out)
        assert result["union_hits"] == 2
        assert result["headroom_questions"] == 1
