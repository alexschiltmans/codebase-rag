"""Rescore saved benchmark and reranker results against the current test set.

Every published arm's retrieved document order is already on disk: `per_question.actual_sources`
in `evals/bench_results/*.json`, and the frozen candidate lists in `evals/bench_candidates/*.json`
plus the post-rerank `top_sources` in `evals/rerank_results/*.json`. Scoring only compares
`expected_sources` against that retrieved order, so when only `testset.json`'s `sources` change,
every arm can be rescored without touching Qdrant or re-running retrieval.

This is not a substitute for `bench_retrieval.py` when the corpus, embeddings, or candidate depth
change. It only isolates the effect of a test-set edit against retrieval that is otherwise
byte-identical to the original run.

By default this only reports: it writes a before/after summary and leaves the arm files alone.
`--write` additionally folds the recomputed numbers back into the arm files, but only for arms that
cover every scored question in the test set. Arms recorded against a smaller, older test set are
left untouched, so a historical run stays readable as the run it actually was. Rewritten arms are
restamped with the test set they now reflect, so the untouched ones stay distinguishable by more
than their filename.

Usage:
    uv run python evals/rescore_testset.py
    uv run python evals/rescore_testset.py --testset evals/testset.json
    uv run python evals/rescore_testset.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from retrieval_metrics import compute_recall_at_depth, compute_retrieval_hit_and_reciprocal_rank
from testset_provenance import testset_provenance

EVALS_DIR = Path(__file__).parent
BENCH_RESULTS_DIR = EVALS_DIR / "bench_results"
RERANK_RESULTS_DIR = EVALS_DIR / "rerank_results"
BENCH_CANDIDATES_DIR = EVALS_DIR / "bench_candidates"


def load_testset(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def expected_sources_by_question(testset: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each scored question to its expected sources, dropping expected_failure questions."""
    return {q["question"]: q.get("sources", []) for q in testset if not q.get("expected_failure")}


def rescore_bench_result(record: dict[str, Any], expected_by_question: dict[str, list[str]]) -> dict[str, Any]:
    """Recompute hit/mrr/recall and category breakdown for one first-stage arm."""
    per_question = []
    skipped = []
    for row in record["per_question"]:
        expected = expected_by_question.get(row["question"])
        if expected is None:
            skipped.append(row["question"])
            continue
        actual = row["actual_sources"]
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(expected, actual)
        recall = compute_recall_at_depth(expected, actual)
        per_question.append(
            {
                "question": row["question"],
                "category": row.get("category", ""),
                "expected_sources": expected,
                "actual_sources": actual,
                "hit": hit,
                "reciprocal_rank": rr,
                "recall_at_depth": recall,
            }
        )

    scored = {r["question"] for r in per_question}
    # Questions the arm never retrieved for cannot be rescored from disk, and their absence is what
    # makes an arm's numbers incomparable to an arm that did cover them. Name them rather than average
    # over a smaller denominator in silence.
    unscored = [q for q in expected_by_question if q not in scored]

    n = len(per_question) or 1
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in per_question:
        by_category.setdefault(r["category"], []).append(r)
    category_breakdown = {
        category: {
            "hit_rate": sum(r["hit"] for r in rows) / len(rows),
            "mrr": sum(r["reciprocal_rank"] for r in rows) / len(rows),
            "recall_at_depth": sum(r["recall_at_depth"] for r in rows) / len(rows),
            "n": len(rows),
        }
        for category, rows in by_category.items()
    }

    return {
        "hit_rate": sum(r["hit"] for r in per_question) / n,
        "mrr": sum(r["reciprocal_rank"] for r in per_question) / n,
        "recall_at_depth": sum(r["recall_at_depth"] for r in per_question) / n,
        "category_breakdown": category_breakdown,
        "per_question": per_question,
        "n_scored": len(per_question),
        "skipped_questions": skipped,
        "unscored_questions": unscored,
        "covers_testset": not unscored and bool(per_question),
    }


def rescore_rerank_result(
    record: dict[str, Any],
    expected_by_question: dict[str, list[str]],
    candidates_by_question: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Recompute input recall, no-rerank baseline, and post-rerank hit/mrr for one reranker arm."""
    input_depth = record["input_depth"]
    output_depth = record["output_depth"]
    per_question = []
    stale_candidates = []

    # `hit`/`mrr` come from the record's saved post-rerank order while `input_recall` and the baseline
    # come from the candidates file on disk. That only stays coherent while the candidates file is the
    # one the arm actually ran against; bench_candidates/ is regenerable and gets overwritten by any
    # later sweep, so a question the record scored but the file no longer carries means the two halves
    # have drifted apart and the arm needs a real rerun rather than a rescore.
    #
    # Depth is checked against the file as a whole, never per question. `input_depth` is the deepest
    # list the arm saw, not a length every list has: a thresholded arm returns fewer candidates for
    # some questions by construction, and reading one short list as drift condemns an untouched file.
    deepest_on_disk = max((len(c) for c in candidates_by_question.values()), default=0)
    if deepest_on_disk > input_depth:
        stale_candidates.append(f"file holds {deepest_on_disk} candidates where the arm ran at {input_depth}")
    elif candidates_by_question and deepest_on_disk < input_depth:
        stale_candidates.append(f"file is only {deepest_on_disk} deep where the arm ran at {input_depth}")

    skipped = []
    for row in record["per_question"]:
        question = row["question"]
        expected = expected_by_question.get(question)
        if expected is None:
            skipped.append(question)
            continue
        candidates = candidates_by_question.get(question)
        if candidates is None:
            stale_candidates.append(question)
            continue

        input_sources = [c["source"] for c in candidates[:input_depth]]
        ceiling = compute_recall_at_depth(expected, input_sources)
        base_hit, base_rr = compute_retrieval_hit_and_reciprocal_rank(expected, input_sources[:output_depth])
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(expected, row["top_sources"])

        per_question.append(
            {
                "question": question,
                "input_recall": ceiling,
                "baseline_hit": base_hit,
                "baseline_reciprocal_rank": base_rr,
                "hit": hit,
                "reciprocal_rank": rr,
            }
        )

    # Staleness is checked first: when every question is stale nothing gets scored, and reporting that
    # as an empty overlap sends the reader to testset.json instead of to the candidates file.
    if stale_candidates:
        return {
            "error": "candidates file no longer matches this arm's saved run; rerun rerank_bench.py",
            "candidate_file_mismatches": stale_candidates,
        }
    n = len(per_question)
    if n == 0:
        return {"error": "no overlapping questions between candidates file and test set"}

    scored = {r["question"] for r in per_question}
    unscored = [q for q in expected_by_question if q not in scored]

    return {
        "questions_scored": n,
        "input_recall": sum(q["input_recall"] for q in per_question) / n,
        "baseline_hit_rate": sum(q["baseline_hit"] for q in per_question) / n,
        "baseline_mrr": sum(q["baseline_reciprocal_rank"] for q in per_question) / n,
        "hit_rate": sum(q["hit"] for q in per_question) / n,
        "mrr": sum(q["reciprocal_rank"] for q in per_question) / n,
        "per_question": per_question,
        "unscored_questions": unscored,
        "skipped_questions": skipped,
        "covers_testset": not unscored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=EVALS_DIR / "testset.json")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the rescore summary JSON.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Fold the recomputed numbers back into the arm files, for arms covering the whole test set.",
    )
    args = parser.parse_args()

    testset = load_testset(args.testset)
    expected_by_question = expected_sources_by_question(testset)
    n_scored_target = len(expected_by_question)
    provenance = testset_provenance(testset)

    summary: dict[str, Any] = {
        "testset_path": str(args.testset),
        "n_scored": n_scored_target,
        **provenance,
        "written": [],
        "bench_results": {},
        "rerank_results": {},
    }

    for path in sorted(BENCH_RESULTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        before = {"hit_rate": record["hit_rate"], "mrr": record["mrr"], "recall_at_depth": record["recall_at_depth"]}
        after = rescore_bench_result(record, expected_by_question)
        if args.write and after["covers_testset"]:
            record |= {
                **provenance,
                "hit_rate": after["hit_rate"],
                "mrr": after["mrr"],
                "recall_at_depth": after["recall_at_depth"],
                "category_breakdown": after["category_breakdown"],
                "per_question": after["per_question"],
            }
            path.write_text(json.dumps(record, indent=2))
            summary["written"].append(path.stem)
        summary["bench_results"][path.stem] = {
            "before": before,
            "after": {k: v for k, v in after.items() if k != "per_question"},
        }

    for path in sorted(RERANK_RESULTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        candidates_path = Path(record["candidates_file"])
        if not candidates_path.exists():
            summary["rerank_results"][path.stem] = {"error": f"missing candidates file {candidates_path}"}
            continue
        candidates_by_question = {
            entry["question"]: entry["candidates"] for entry in json.loads(candidates_path.read_text())
        }
        before = {
            "hit_rate": record["hit_rate"],
            "mrr": record["mrr"],
            "input_recall": record["input_recall"],
            "baseline_hit_rate": record["baseline_hit_rate"],
            "baseline_mrr": record["baseline_mrr"],
        }
        after = rescore_rerank_result(record, expected_by_question, candidates_by_question)
        if args.write and after.get("covers_testset"):
            rescored_rows = {r["question"]: r for r in after["per_question"]}
            # Merge rather than replace: the saved rows also carry `latency_s` and `top_sources`, which
            # are measurements of the original run and are not ours to regenerate. Rows the rescore did
            # not reach are dropped rather than merged, because their scores were computed against
            # ground truth the test set no longer holds and the aggregates no longer include them.
            record["per_question"] = [
                row | rescored_rows[row["question"]]
                for row in record["per_question"]
                if row["question"] in rescored_rows
            ]
            record |= provenance | {
                k: after[k]
                for k in ("questions_scored", "input_recall", "baseline_hit_rate", "baseline_mrr", "hit_rate", "mrr")
            }
            path.write_text(json.dumps(record, indent=2))
            summary["written"].append(path.stem)
        summary["rerank_results"][path.stem] = {
            "before": before,
            "after": {k: v for k, v in after.items() if k != "per_question"},
        }

    output_path = args.output or (EVALS_DIR / "rescore_summary.json")
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
