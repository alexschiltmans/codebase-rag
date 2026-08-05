"""Unit tests for evals/rerank_bench.py: order parsing, depth handling, and aggregation.

Runnable offline with a stubbed reranker; the reranker sweeps themselves are not tests.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.rerank_bench import (
    ListwiseLLMReranker,
    expected_sources_by_question,
    run_rerank,
    score_ranking,
    slugify,
    truncate,
)


def _candidate(source: str, content: str = "text") -> dict[str, Any]:
    return {"source": source, "page_content": content}


class TestSlugify:
    def test_slashes_and_dots_become_dashes(self) -> None:
        assert slugify("cross-encoder/ms-marco-MiniLM-L6-v2") == "cross-encoder-ms-marco-minilm-l6-v2"


class TestTruncate:
    def test_leaves_short_text_alone(self) -> None:
        assert truncate("abc", 10) == "abc"

    def test_cuts_at_limit(self) -> None:
        assert truncate("abcdef", 3) == "abc"


class TestExpectedSourcesByQuestion:
    def test_expected_failure_questions_are_dropped(self) -> None:
        """The published eval excludes these, so scoring them would make every
        reranker number incomparable to the first-stage numbers it is judged against."""
        testset: list[dict[str, Any]] = [
            {"question": "q1", "sources": ["a.py"]},
            {"question": "q2", "sources": ["b.py"], "expected_failure": True},
        ]
        assert expected_sources_by_question(testset) == {"q1": ["a.py"]}


class TestApplyOrder:
    def _reranker(self) -> ListwiseLLMReranker:
        return ListwiseLLMReranker.__new__(ListwiseLLMReranker)

    def test_well_formed_reply_reorders(self) -> None:
        window = [_candidate("a.py"), _candidate("b.py"), _candidate("c.py")]
        ordered = self._reranker()._apply_order(window, "[3] > [1] > [2]")
        assert [c["source"] for c in ordered] == ["c.py", "a.py", "b.py"]

    def test_unmentioned_candidates_keep_their_place_at_the_back(self) -> None:
        window = [_candidate("a.py"), _candidate("b.py"), _candidate("c.py")]
        ordered = self._reranker()._apply_order(window, "[2]")
        assert [c["source"] for c in ordered] == ["b.py", "a.py", "c.py"]

    def test_duplicate_and_out_of_range_identifiers_are_ignored(self) -> None:
        window = [_candidate("a.py"), _candidate("b.py")]
        ordered = self._reranker()._apply_order(window, "[2] > [2] > [9] > [1]")
        assert [c["source"] for c in ordered] == ["b.py", "a.py"]

    def test_unparseable_reply_degrades_to_input_order(self) -> None:
        """A model that cannot follow the format must score as a no-op, not drop candidates."""
        window = [_candidate("a.py"), _candidate("b.py")]
        ordered = self._reranker()._apply_order(window, "I cannot rank these.")
        assert [c["source"] for c in ordered] == ["a.py", "b.py"]
        assert len(ordered) == len(window)


class TestScoreRanking:
    def test_only_the_output_depth_is_scored(self) -> None:
        ranked = [_candidate("miss.py"), _candidate("hit.py")]
        assert score_ranking(["hit.py"], ranked, output_depth=1) == (0, 0.0)
        assert score_ranking(["hit.py"], ranked, output_depth=2) == (1, 0.5)


class TestRunRerank:
    def _lists(self) -> list[dict[str, Any]]:
        return [
            {"question": "q1", "candidates": [_candidate("miss.py"), _candidate("hit.py")]},
            {"question": "q2", "candidates": [_candidate("nowhere.py")]},
        ]

    def test_no_reranker_control_matches_its_own_baseline(self) -> None:
        result = run_rerank(None, self._lists(), {"q1": ["hit.py"], "q2": ["gone.py"]}, 10, 10)

        assert result["questions_scored"] == 2
        assert result["hit_rate"] == result["baseline_hit_rate"] == 0.5
        assert result["mrr"] == result["baseline_mrr"] == 0.25
        assert result["latency_s"]["total"] == 0.0

    def test_reranker_gain_is_reported_against_the_unranked_list(self) -> None:
        reranker = MagicMock()
        reranker.rank.side_effect = lambda _q, cands: sorted(cands, key=lambda c: c["source"] != "hit.py")

        result = run_rerank(reranker, self._lists(), {"q1": ["hit.py"], "q2": ["gone.py"]}, 10, 10)

        assert result["baseline_mrr"] == 0.25
        assert result["mrr"] == 0.5

    def test_input_recall_is_the_ceiling_not_the_score(self) -> None:
        """A hit that sits below the output depth still counts toward recall, which is
        what bounds every reranker scored against the list."""
        result = run_rerank(None, self._lists(), {"q1": ["hit.py"], "q2": ["gone.py"]}, 10, 1)

        assert result["input_recall"] == 0.5
        assert result["hit_rate"] == 0.0

    def test_input_depth_truncates_before_reranking(self) -> None:
        reranker = MagicMock()
        reranker.rank.side_effect = lambda _q, cands: cands

        run_rerank(reranker, self._lists(), {"q1": ["hit.py"], "q2": ["gone.py"]}, 1, 10)

        assert len(reranker.rank.call_args_list[0].args[1]) == 1

    def test_questions_absent_from_the_testset_are_skipped(self) -> None:
        result = run_rerank(None, self._lists(), {"q1": ["hit.py"]}, 10, 10)

        assert result["questions_scored"] == 1
        assert result["per_question"][0]["question"] == "q1"
