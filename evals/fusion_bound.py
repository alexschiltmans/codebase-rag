"""Bound what combining several retrievers could reach, from arms already on disk.

A combined first stage built over several retrievers draws its output from the union of their
candidate lists, and truncating that union to an output depth can only lose documents. So the recall
of the union at matched input depth is a ceiling on any fusion of those inputs: rank fusion, weighted
score fusion, or anything else. When that ceiling sits at or near the best single component, the
combination cannot pay for itself and does not need to be built to find out.

This computes the ceiling, not a measurement. It says what a combination could reach, never what one
did, and a real fusion may fall short of it.

Every arm's retrieved order is already saved as `per_question.actual_sources` in
`evals/bench_results/*.json`, so nothing here touches Qdrant, an embedding model, or the corpus.
Components and their union are both scored through `compute_recall_at_depth`, so one definition of a
hit produces every number in the output; the `hit` fields saved in the arm records are not read.

Arms are only comparable when they were scored against the same questions at the same depth over the
same corpus, and `bench_results/` holds arms from more than one test set under filenames that encode
the model, retriever and depth but nothing about the questions. Every mismatch below exits non-zero
rather than producing a union that averages over two different instruments. Chunking is gated the
same way, including its absence: arms written before the chunk configuration was recorded carry no
value for it, and an arm that records one is not comparable against an arm that does not.

Two arms that both record nothing pass that gate and cannot be checked further, so the report says
so rather than implying the chunking was verified. It is only reachable for arms predating the field;
every corpus written since, the application's included, carries a sidecar.

Usage:
    uv run python evals/fusion_bound.py baai-bge-m3_vector_d10_float16_seq768 \\
        qwen-qwen3-embedding-0-6b_vector_d10_float16_seq768
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from retrieval_metrics import compute_recall_at_depth

EVALS_DIR = Path(__file__).parent
BENCH_RESULTS_DIR = EVALS_DIR / "bench_results"

GATED_FIELDS = (
    "testset_hash",
    "testset_size",
    "candidate_depth",
    "ingested_repositories",
    "chunk_size",
    "chunk_overlap",
)
REQUIRED_ROW_FIELDS = ("question", "expected_sources", "actual_sources")


class FusionBoundError(Exception):
    """Base for every condition that stops a bound from being computed."""


class IncomparableArmsError(FusionBoundError):
    """Raised when the given arms cannot be combined into a meaningful union."""


class MalformedArmRecordError(FusionBoundError):
    """Raised when a file is not a benchmark arm record this can score."""


def resolve_arm_path(name: str) -> Path:
    """Accept a bare arm name or a path, and return the arm record path.

    A name carrying a directory is honored as a path, including its failure. Stems repeat across
    `bench_results/`, `bench_candidates/` and `rerank_results/`, so falling back to the stem would
    answer a mistyped directory with a different arm's numbers instead of an error.
    """
    path = Path(name)
    if path.is_absolute() or len(path.parts) > 1:
        if not path.exists():
            raise MalformedArmRecordError(f"no arm record at {path}")
        return path

    stem = path.name[: -len(".json")] if path.name.endswith(".json") else path.name
    candidate = BENCH_RESULTS_DIR / f"{stem}.json"
    if not candidate.exists():
        raise MalformedArmRecordError(f"no arm record at {candidate}")
    return candidate


def load_arm(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MalformedArmRecordError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedArmRecordError(
            f"{path} holds a JSON {type(payload).__name__}, not a benchmark arm record; "
            "candidate dumps and other list-shaped files cannot be scored here"
        )
    record: dict[str, Any] = payload
    record["arm_name"] = path.stem
    record["arm_path"] = str(path)
    return record


def label_arms(arms: list[dict[str, Any]]) -> list[str]:
    """Give every arm a distinct label, and refuse the ones that cannot have it.

    Labels key the gates, the components and the per-question attribution, so two arms sharing one
    label would be collapsed into a single column and a single gate entry. Same stem in different
    directories is a real pairing and gets the directory in its label; the same record twice is not
    a union and stops here.
    """
    counts = Counter(arm["arm_name"] for arm in arms)
    labels = []
    for arm in arms:
        name = arm["arm_name"]
        path = arm.get("arm_path")
        labels.append(f"{Path(path).parent.name}/{name}" if counts[name] > 1 and path else name)

    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise IncomparableArmsError(
            f"the same arm was given more than once: {', '.join(repeated)}; "
            "the union of an arm with itself is that arm, not a bound on a combination"
        )
    return labels


def validate_arm(arm: dict[str, Any], label: str) -> None:
    """Reject records that are not shaped like a scored benchmark arm.

    `rerank_results/` records and candidate dumps live one directory away and carry neither the
    ground truth nor the retrieved order under these names.
    """
    if "candidate_depth" not in arm:
        raise MalformedArmRecordError(f"{label} has no candidate_depth, so its depth cannot be matched to the others")

    rows = arm.get("per_question")
    if not isinstance(rows, list) or not rows:
        raise MalformedArmRecordError(f"{label} has no per_question rows to score")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MalformedArmRecordError(f"{label} per_question[{index}] is not an object")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise MalformedArmRecordError(
                f"{label} per_question[{index}] is missing {', '.join(missing)}; "
                "this is not a scored first-stage arm record"
            )


def check_comparable(arms: list[dict[str, Any]], labels: list[str]) -> None:
    """Reject arms whose union would be meaningless, naming the mismatch.

    Provenance is compared including its absence: records written before `testset_provenance` existed
    carry no `testset_hash`, and treating a missing hash as matching a present one is what would
    silently fuse a 29-question arm into a 42-question table.
    """
    if len(arms) < 2:
        raise IncomparableArmsError("a union needs at least two arms")

    for field in GATED_FIELDS:
        values = [(label, arm.get(field)) for label, arm in zip(labels, arms, strict=True)]
        if len({_hashable(value) for _, value in values}) > 1:
            detail = ", ".join(f"{label}={value!r}" for label, value in values)
            raise IncomparableArmsError(f"arms disagree on {field}: {detail}")

    reference, reference_label = arms[0], labels[0]
    reference_questions = [row["question"] for row in reference["per_question"]]
    for arm, label in zip(arms[1:], labels[1:], strict=True):
        questions = [row["question"] for row in arm["per_question"]]
        if questions != reference_questions:
            raise IncomparableArmsError(
                f"arms disagree on their question list: {reference_label} has "
                f"{len(reference_questions)} questions, {label} has {len(questions)}, "
                "and they do not match in content and order"
            )

    # Ground truth is a set of patterns to the scorer and is sorted before hashing, so only its
    # contents can disagree.
    reference_expected = {row["question"]: sorted(row["expected_sources"]) for row in reference["per_question"]}
    for arm, label in zip(arms[1:], labels[1:], strict=True):
        for row in arm["per_question"]:
            if sorted(row["expected_sources"]) != reference_expected[row["question"]]:
                raise IncomparableArmsError(
                    f"arms disagree on the expected sources for {row['question']!r}: "
                    f"{reference_label} expects {reference_expected[row['question']]}, "
                    f"{label} expects {sorted(row['expected_sources'])}"
                )


def _hashable(value: Any) -> Any:
    return tuple(sorted(value)) if isinstance(value, list) else value


def fusion_bound(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Score each arm and the union of all of them, and report the headroom between them.

    The union is the concatenation of every arm's retrieved sources for a question. Order is
    irrelevant to `compute_recall_at_depth`, which scans the whole list, and a bound has no ordering
    to preserve.

    Components can tie for best, in which case each tied arm gets its own list of questions the union
    gains over it. The headroom is the same number against any of them; which questions supply it is
    not, and picking one tied arm to attribute them to would make that attribution an artifact of
    argument order.
    """
    labels = label_arms(arms)
    for arm, label in zip(arms, labels, strict=True):
        validate_arm(arm, label)
    check_comparable(arms, labels)

    questions = [row["question"] for row in arms[0]["per_question"]]
    expected_by_question = {row["question"]: row["expected_sources"] for row in arms[0]["per_question"]}
    retrieved_by_arm = {
        label: {row["question"]: row["actual_sources"] for row in arm["per_question"]}
        for label, arm in zip(labels, arms, strict=True)
    }

    per_arm_hits = {
        label: {q: compute_recall_at_depth(expected_by_question[q], retrieved[q]) for q in questions}
        for label, retrieved in retrieved_by_arm.items()
    }
    union_hits = {
        q: compute_recall_at_depth(
            expected_by_question[q], [src for retrieved in retrieved_by_arm.values() for src in retrieved[q]]
        )
        for q in questions
    }

    n = len(questions)
    components = {
        label: {"hits": sum(hits.values()), "recall": sum(hits.values()) / n} for label, hits in per_arm_hits.items()
    }
    best_hits = max(stats["hits"] for stats in components.values())
    best_components = [label for label, stats in components.items() if stats["hits"] == best_hits]
    union_total = sum(union_hits.values())

    return {
        "n_questions": n,
        "testset_hash": arms[0].get("testset_hash"),
        "testset_size": arms[0].get("testset_size"),
        "candidate_depth": arms[0]["candidate_depth"],
        "ingested_repositories": arms[0].get("ingested_repositories"),
        "chunk_size": arms[0].get("chunk_size"),
        "chunking_unverified": all(arm.get("chunk_size") is None for arm in arms),
        "components": components,
        "best_components": best_components,
        "best_component_hits": best_hits,
        "union_hits": union_total,
        "union_recall": union_total / n,
        "headroom_questions": union_total - best_hits,
        "questions_gained_by_union": {
            label: [q for q in questions if union_hits[q] and not per_arm_hits[label][q]] for label in best_components
        },
    }


def format_report(result: dict[str, Any]) -> str:
    n = result["n_questions"]
    width = max(len(name) for name in result["components"])
    repositories = result["ingested_repositories"]
    lines = [
        f"test set: {n} questions (hash {result['testset_hash'] or 'none recorded'}), "
        f"candidate depth {result['candidate_depth']}, "
        f"corpus {', '.join(repositories) if repositories else 'none recorded'}, "
        f"chunk size {result['chunk_size'] or 'none recorded'}",
    ]
    if result.get("chunking_unverified"):
        lines.append(
            "  warning: no arm records a chunking, so the gate passed on absence alone. "
            "Two arms scored either side of a re-ingest look identical here and are not."
        )
    tied = len(result["best_components"]) > 1
    for name, stats in result["components"].items():
        marker = (" (tied best)" if tied else " (best)") if name in result["best_components"] else ""
        lines.append(f"  {name:<{width}}  {stats['hits']:>3}/{n}  {stats['recall']:.4f}{marker}")
    lines.append(f"  {'union (upper bound)':<{width}}  {result['union_hits']:>3}/{n}  {result['union_recall']:.4f}")
    lines.append(f"  headroom over best component: {result['headroom_questions']} question(s)")
    for name, gained in result["questions_gained_by_union"].items():
        for question in gained:
            lines.append(f"    gained over {name}: {question}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="Arm names or paths under evals/bench_results/.")
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON instead of a table.")
    args = parser.parse_args()

    try:
        arms = [load_arm(resolve_arm_path(name)) for name in args.arms]
        result = fusion_bound(arms)
    except FusionBoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result, indent=2) if args.json else format_report(result))


if __name__ == "__main__":
    main()
