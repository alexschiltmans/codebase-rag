"""Paired significance tests over saved benchmark arms, from records already on disk.

The repository's ranking rule calls a difference of three questions or more a ranking, on a set
where one question is worth a little under a quarter of a point. That is a consistent screen and it
is applied the same way to every comparison, but it is uncalibrated: it says nothing about how often
such a gap arises from noise on a fixed set of paired questions. This module turns the screen into a
measured statement without re-retrieving anything.

Every arm record in `bench_results/` carries per-question hit and reciprocal rank for the same
question set, which is exactly the input a paired test needs. It computes McNemar's exact test on the
discordant hit pairs between two records, a Wilson interval on each record's hit rate, and a paired
bootstrap over questions for the difference in a continuous per-question metric (reciprocal rank by
default). It also counts, across any number of records, how many questions ever change outcome: the
test set's effective size for ranking configurations, which is smaller than the number of questions
scored.

The three-question rule stays useful as a screen. It is not a significance threshold, and on 42
paired questions an eight-versus-three discordant split is p=0.227, which is what decides that a
difference passing the screen is not yet a ranking.

The exact binomial test is a few lines with `math.comb` and needs no scipy. numpy is taken for the
resampling, which is one indexed draw over a (resamples, questions) matrix, and is declared in the
dev extra rather than relied on from somebody else's dependency tree.

Two records are only paired when they measured the same thing: the same record shape, the same
questions in the same order, and the same expected sources per question. A retrieval-only arm and an
end-to-end run are different instruments, and arms at different chunk sizes are too; every mismatch
exits non-zero rather than producing a statistic that reads as a retriever comparison and is a harness
comparison. Refusal names what disagreed.

Usage:
    uv run python evals/paired_stats.py \
        sentence-transformers-all-mpnet-base-v2_hybrid_d10_chunk614_on-documents \
        sentence-transformers-all-mpnet-base-v2_bm25_d10_chunk614
    uv run python evals/paired_stats.py --pairs \
        <arm> <arm> <arm> ...
    uv run python evals/paired_stats.py --discriminating <arm> <arm> ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from fusion_bound import GATED_FIELDS, _hashable
from retrieval_metrics import compute_retrieval_hit_and_reciprocal_rank

EVALS_DIR = Path(__file__).parent
BENCH_RESULTS_DIR = EVALS_DIR / "bench_results"

# GATED_FIELDS and _hashable live in fusion_bound.py so the two modules refuse the same pairs. A
# second copy here would let them drift apart and silently stop refusing a mismatch.

# Per-question keys already given a meaning above, so anything else numeric on a row rides along as a
# candidate metric without being re-interpreted.
_KNOWN_ROW_KEYS = {
    "question",
    "category",
    "expected_failure",
    "sources_expected",
    "sources_actual",
    "hit",
    "reciprocal_rank",
    "metrics",
    "answer",
    "expected_answer",
    "keywords",
    "contexts",
    "difficulty",
}


class PairedStatsError(Exception):
    """Base for every condition that stops a paired statistic from being computed."""


class IncomparableArmsError(PairedStatsError):
    """Raised when the given records were not measured on the same questions or instrument."""


class MalformedRecordError(PairedStatsError):
    """Raised when a file is not a benchmark arm or result record this can score."""


def resolve_record_path(name: str) -> Path:
    """Accept a bare record name or a path, and return the record path.

    A name carrying a directory is honored as a path, including its failure. Arms live in
    `bench_results/` and end-to-end result files in `evals/`, so a bare name is looked up in each,
    the directory first. Stems repeat across `bench_results/`, `bench_candidates/` and
    `rerank_results/`, so falling back to the stem across those would answer a mistyped directory
    with a different arm's numbers instead of an error; the search is limited to the two
    directories that hold records this can score, and reports both when neither has the name.
    """
    path = Path(name)
    if path.is_absolute() or len(path.parts) > 1:
        if not path.exists():
            raise MalformedRecordError(f"no record at {path}")
        return path

    stem = path.name.removesuffix(".json")
    candidates = [BENCH_RESULTS_DIR / f"{stem}.json", EVALS_DIR / f"{stem}.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    looked = ", ".join(str(candidate) for candidate in candidates)
    raise MalformedRecordError(f"no record for {name!r} in {looked}")


def load_record(path: Path) -> dict[str, Any]:
    """Read a record file into a dict, naming it and locating it for the report and the gates."""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MalformedRecordError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedRecordError(
            f"{path} holds a JSON {type(payload).__name__}, not a record; "
            "candidate dumps and other list-shaped files cannot be scored here"
        )
    payload["arm_name"] = path.stem
    payload["arm_path"] = str(path)
    return payload


def _is_retrieval_only(record: dict[str, Any]) -> bool:
    rows = record.get("per_question")
    return isinstance(rows, list) and bool(rows)


def _is_end_to_end(record: dict[str, Any]) -> bool:
    results = record.get("results")
    return isinstance(results, list) and bool(results)


def _extra_numeric(row: dict[str, Any]) -> dict[str, float]:
    """Any numeric field on a row that is not one of the known per-question fields above."""
    extras: dict[str, float] = {}
    for key, value in row.items():
        if key in _KNOWN_ROW_KEYS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            extras[key] = float(value)
    return extras


RETRIEVAL_REQUIRED_FIELDS = ("question", "hit", "reciprocal_rank")


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce one record to per-question rows carrying question, category, hit, reciprocal rank,
    the sorted expected sources, and any further numeric per-question field the record holds.

    Hit and reciprocal rank are read straight off a retrieval-only arm, which must therefore carry
    them. An end-to-end result file carries neither, so they are derived through the same shared
    metric `run_eval.py` and `bench_retrieval.py` use, which keeps a hit to one definition. The
    end-to-end branch drops rows flagged `expected_failure` and rows with no expected sources,
    because the published aggregates drop them and scoring an unanswerable question as a miss for
    both arms would push every p-value toward one.
    """
    if _is_retrieval_only(record):
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(record["per_question"]):
            missing = [field for field in RETRIEVAL_REQUIRED_FIELDS if field not in row]
            if missing:
                raise MalformedRecordError(
                    f"per_question[{index}] is missing {', '.join(missing)}; this is not a "
                    "scored first-stage arm record"
                )
            normalised: dict[str, Any] = {
                "question": row["question"],
                "category": row.get("category"),
                "hit": int(row["hit"]),
                "reciprocal_rank": float(row["reciprocal_rank"]),
                "expected_sources": sorted(row.get("expected_sources", [])),
            }
            normalised.update(_extra_numeric(row))
            rows.append(normalised)
        shape = "retrieval-only"
        header = {field: record.get(field) for field in GATED_FIELDS}
    elif _is_end_to_end(record):
        rows = []
        for row in record["results"]:
            if row.get("error"):
                name = record.get("label") or record.get("arm_name") or record.get("arm_path", "record")
                raise PairedStatsError(
                    f"{name} has an errored run on {row.get('question')!r}; the published aggregate "
                    "excludes failed questions, so this record cannot be scored here"
                )
            if row.get("expected_failure"):
                continue
            if not row.get("sources_expected"):
                continue
            hit, reciprocal_rank = compute_retrieval_hit_and_reciprocal_rank(
                row.get("sources_expected", []),
                row.get("sources_actual", []),
            )
            normalised = {
                "question": row["question"],
                "category": row.get("category"),
                "hit": int(hit),
                "reciprocal_rank": float(reciprocal_rank),
                "expected_sources": sorted(row.get("sources_expected", [])),
            }
            normalised.update(_extra_numeric(row))
            rows.append(normalised)
        shape = "end-to-end"
        header = {}
    else:
        raise MalformedRecordError(
            "record is neither a retrieval-only arm (per_question) nor an end-to-end run (results)"
        )
    return {"shape": shape, "rows": rows, "header": header}


def _record_label(record: dict[str, Any]) -> str:
    return record.get("label") or record.get("arm_name") or "a record"


def check_comparable(records: list[dict[str, Any]]) -> None:
    """Reject records that were not measured on the same questions with the same instrument.

    The shape gate keeps a retrieval-only arm from being paired with an end-to-end run: they are
    different instruments and any difference between them describes the harnesses, not the retrievers.
    The question list must match in content and order, and each question's expected sources must
    match, because the scorer matches patterns against documents and only the contents carry meaning.
    For retrieval-only arms the fields `fusion_bound.py` gates are compared too, including their
    absence, which is why an arm recording a chunk size is not comparable against one that does not.
    """
    if len({record["shape"] for record in records}) > 1:
        shapes = ", ".join(sorted(record["shape"] for record in records))
        raise IncomparableArmsError(
            f"{_record_label(records[1])} is not the same shape as the first arm ({shapes}); a "
            "retrieval-only arm and an end-to-end run are different instruments and cannot be paired"
        )

    reference = records[0]["rows"]
    reference_questions = [row["question"] for row in reference]
    for record in records[1:]:
        questions = [row["question"] for row in record["rows"]]
        if questions != reference_questions:
            raise IncomparableArmsError(
                f"{_record_label(record)} disagrees on the question list: the first arm has "
                f"{len(reference_questions)} questions and this one does not match in content and order"
            )

    reference_expected = {row["question"]: row["expected_sources"] for row in reference}
    for record in records[1:]:
        for row in record["rows"]:
            if row["expected_sources"] != reference_expected[row["question"]]:
                raise IncomparableArmsError(
                    f"{_record_label(record)} disagrees on the expected sources for {row['question']!r}: "
                    f"{reference_expected[row['question']]!r} vs {row['expected_sources']!r}"
                )

    if records[0]["shape"] == "retrieval-only":
        for field in GATED_FIELDS:
            values = [record["header"].get(field) for record in records]
            if len({_hashable(value) for value in values}) > 1:
                detail = ", ".join(f"{i}={value!r}" for i, value in enumerate(values))
                raise IncomparableArmsError(f"{_record_label(records[1])} disagrees on {field}: {detail}")


def _metric_value(row: dict[str, Any], metric: str, label: str) -> float:
    value = row.get(metric)
    if value is None:
        raise PairedStatsError(
            f"metric {metric!r} is not available per question in {label}; "
            "the interval runs only on a field both records carry"
        )
    return float(value)


def exact_binomial_test(a_only: int, b_only: int) -> float:
    """Two-sided exact binomial test on the discordant pairs, at the null of no difference.

    The discordant pairs are the questions one record won and the other missed. Under the null each
    discordant pair is equally likely to fall either way, so the number going one way is
    Binomial(a_only + b_only, 1/2). The two-sided p-value is the smaller tail, doubled and capped at
    1.0. It is 1.0 when there are no discordant pairs, which is the case this test exists to report
    rather than crash on. No scipy: the tail is a sum of `math.comb` terms.
    """
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    majority = max(a_only, b_only)
    tail = sum(math.comb(discordant, k) for k in range(majority, discordant + 1))
    total = 2**discordant
    return float(min(2 * tail / total, 1.0))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion at the given confidence.

    Preferred over the normal approximation here because hit rates sit near 0.9 on a sample of 42,
    where the normal interval overshoots 1.0 and undershoots 0. The Wilson interval stays in [0, 1].
    """
    if n == 0:
        raise PairedStatsError("hit rate is undefined over zero questions")
    proportion = successes / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    # A proportion interval is bounded by [0, 1]; the algebra stays inside it and the clamp only
    # absorbs the floating-point overshoot that shows up at the extremes.
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap_difference(
    a_values: list[float],
    b_values: list[float],
    resamples: int,
    seed: int,
) -> tuple[float, float, float, bool]:
    """Paired bootstrap over questions for the difference of a per-question metric.

    The per-question difference is drawn once per record pair, and the bootstrap resamples those
    differences with replacement, so the pairing is preserved and the interval is on the difference,
    not on either record alone. Returns the mean difference, the percentile interval, and whether that
    interval excludes zero.
    """
    import numpy as np

    if resamples < 1:
        raise PairedStatsError(f"resamples must be at least 1, got {resamples}")
    n = len(a_values)
    differences = np.asarray(a_values, dtype=float) - np.asarray(b_values, dtype=float)
    rng = np.random.default_rng(seed)
    mean = float(differences.mean())
    draws = rng.integers(0, n, size=(resamples, n))
    bootstrapped = differences[draws].mean(axis=1)
    percentiles = np.percentile(bootstrapped, [2.5, 97.5])
    low, high = float(percentiles[0]), float(percentiles[1])
    excludes_zero = bool(low > 0 or high < 0)
    return mean, low, high, excludes_zero


def compare(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    metric: str = "reciprocal_rank",
    resamples: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two records on the questions they share, and refuse if they were not measured together.

    Raises:
        IncomparableArmsError: If the records were not scored on the same questions with the same
            instrument, which makes every paired quantity below meaningless.
        PairedStatsError: If the requested metric is not present per question in one of the records,
            if resamples is not positive, or if neither arm has a scored question.
    """
    normalised_a = normalise_record(record_a)
    normalised_b = normalise_record(record_b)
    normalised_a["label"] = record_a.get("label", record_a.get("arm_name", "record a"))
    normalised_b["label"] = record_b.get("label", record_b.get("arm_name", "record b"))
    check_comparable([normalised_a, normalised_b])
    rows_a = normalised_a["rows"]
    rows_b = normalised_b["rows"]
    n = len(rows_a)
    if n == 0:
        raise PairedStatsError("no scored questions on either arm; there is nothing to compare")

    hits_a = [row["hit"] for row in rows_a]
    hits_b = [row["hit"] for row in rows_b]
    a_only = sum(1 for x, y in zip(hits_a, hits_b, strict=True) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(hits_a, hits_b, strict=True) if x == 0 and y == 1)

    label_a = normalised_a["label"]
    label_b = normalised_b["label"]
    a_values = [_metric_value(row, metric, label_a) for row in rows_a]
    b_values = [_metric_value(row, metric, label_b) for row in rows_b]
    metric_mean_a = sum(a_values) / n
    metric_mean_b = sum(b_values) / n
    metric_difference, metric_low, metric_high, metric_excludes_zero = paired_bootstrap_difference(
        a_values,
        b_values,
        resamples,
        seed,
    )

    return {
        "n": n,
        "metric": metric,
        "hit_a": sum(hits_a) / n,
        "hit_b": sum(hits_b) / n,
        "hits_a": sum(hits_a),
        "hits_b": sum(hits_b),
        "hit_ci_a": wilson_interval(sum(hits_a), n),
        "hit_ci_b": wilson_interval(sum(hits_b), n),
        "discordant_a_only": a_only,
        "discordant_b_only": b_only,
        "mcnemar_exact_p": exact_binomial_test(a_only, b_only),
        "metric_mean_a": metric_mean_a,
        "metric_mean_b": metric_mean_b,
        "metric_difference": metric_difference,
        "metric_ci": (metric_low, metric_high),
        "metric_ci_excludes_zero": metric_excludes_zero,
    }


def discriminating_questions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count how many questions ever change outcome across a set of records scored on the same questions.

    A question hit by every record or missed by every record contributes nothing to any comparison
    between them. The count of the rest is the test set's effective size for ranking configurations,
    which is smaller than the number of questions scored and is not otherwise tracked. The records
    are held to the same comparability gate as a pairwise comparison before counting, so the result
    is over one question set measured one way.
    """
    normalised = [normalise_record(record) for record in records]
    for entry, record in zip(normalised, records, strict=True):
        entry["label"] = record.get("label", record.get("arm_name", "record"))
    check_comparable(normalised)
    rows = normalised[0]["rows"]
    n = len(rows)
    always_hit = 0
    always_miss = 0
    discriminating: list[dict[str, Any]] = []
    for index in range(n):
        outcomes = [entry["rows"][index]["hit"] for entry in normalised]
        category = rows[index]["category"]
        if all(outcomes):
            always_hit += 1
        elif not any(outcomes):
            always_miss += 1
        else:
            discriminating.append({"question": rows[index]["question"], "category": category})
    return {
        "n": n,
        "always_hit": always_hit,
        "always_miss": always_miss,
        "discriminating": discriminating,
        "discriminating_count": len(discriminating),
    }


def _format_comparison(label_a: str, label_b: str, result: dict[str, Any]) -> str:
    metric = result["metric"]
    metric_ci = result["metric_ci"]
    excludes = "excludes 0" if result["metric_ci_excludes_zero"] else "includes 0"
    p_value = result["mcnemar_exact_p"]
    discordant = f"{result['discordant_a_only']} / {result['discordant_b_only']}"
    return "\n".join(
        [
            f"{label_a}",
            f"  vs {label_b}",
            (
                f"    hit:  {result['hit_a']:.4f} ({result['hits_a']}/{result['n']}) "
                f"[{result['hit_ci_a'][0]:.4f}, {result['hit_ci_a'][1]:.4f}]  vs  "
                f"{result['hit_b']:.4f} ({result['hits_b']}/{result['n']}) "
                f"[{result['hit_ci_b'][0]:.4f}, {result['hit_ci_b'][1]:.4f}]"
            ),
            f"    discordant: {discordant}   exact p = {p_value:.4f}",
            (
                f"    {metric} difference: {result['metric_difference']:+.4f}  "
                f"95% CI [{metric_ci[0]:+.4f}, {metric_ci[1]:+.4f}]   {excludes}"
            ),
        ]
    )


def _format_discriminating(result: dict[str, Any]) -> str:
    categories = sorted({entry["category"] for entry in result["discriminating"]})
    return "\n".join(
        [
            f"questions scored: {result['n']}",
            f"always hit: {result['always_hit']}",
            f"always missed: {result['always_miss']}",
            f"discriminating: {result['discriminating_count']}",
            f"discriminating by category: {', '.join(categories) if categories else 'none'}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("arms", nargs="+", help="Record names or paths under evals/bench_results/.")
    parser.add_argument("--pairs", action="store_true", help="Compare every unordered pair rather than the first two.")
    parser.add_argument("--discriminating", action="store_true", help="Report the discriminating-question count.")
    parser.add_argument(
        "--metric", default="reciprocal_rank", help="Per-question metric to interval (default: reciprocal_rank)."
    )
    parser.add_argument("--resamples", type=int, default=20000, help="Bootstrap resamples (default: 20000).")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed, reported with the interval (default: 0).")
    args = parser.parse_args()

    try:
        records = [load_record(resolve_record_path(name)) for name in args.arms]
        for record, name in zip(records, args.arms, strict=True):
            record["label"] = name

        if len(args.arms) < 2:
            raise PairedStatsError("at least two arms are required to compare")

        if args.discriminating:
            result = discriminating_questions(records)
            print(_format_discriminating(result))
            return

        indices = list(range(len(records)))
        pair_indices = [(i, j) for i in indices for j in indices[i + 1 :]] if args.pairs else [(0, 1)]
        printed = []
        for i, j in pair_indices:
            result = compare(records[i], records[j], args.metric, args.resamples, args.seed)
            printed.append(_format_comparison(args.arms[i], args.arms[j], result))
        print("\n\n".join(printed))
    except PairedStatsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
