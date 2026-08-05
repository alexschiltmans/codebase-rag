"""Rescore a frozen first-stage candidate list with a reranker and report what it bought.

The first stage is never re-run: `bench_retrieval.py` saves each configuration's
candidate list to `bench_candidates/`, and every reranker arm reorders the same
saved list. That removes first-stage variance from the comparison and makes the
arms cheap enough to sweep.

Recall of the input list is an upper bound on every arm scored against it, so it
is reported next to the arm's hit rate rather than left implicit. The unranked
list's own hit rate and MRR at the same output depth are reported too: a reranker
that does not beat those has bought nothing.

Usage:
    uv run python evals/rerank_bench.py \\
        --candidates evals/bench_candidates/baai-bge-m3_hybrid_d50_on-bench_baai-bge-m3_1024.json \\
        --reranker cross-encoder/ms-marco-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent))

from retrieval_metrics import compute_recall_at_depth, compute_retrieval_hit_and_reciprocal_rank
from testset_provenance import testset_provenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent
TESTSET_PATH = EVALS_DIR / "testset.json"
RESULTS_DIR = EVALS_DIR / "rerank_results"

DEFAULT_OUTPUT_DEPTH = 10
DEFAULT_BATCH_SIZE = 32

# Listwise defaults follow the sliding-window scheme these rerankers were described with:
# score a window, slide it back by half, so every document is seen in two windows but the
# top of the list is settled last.
DEFAULT_WINDOW = 20
DEFAULT_STRIDE = 10
OLLAMA_URL = "http://127.0.0.1:11434"
# Not `localhost`: that resolves to the Docker container, which has no Metal access.

# The listwise prompt asks for identifiers rather than a rewritten list, so a model that
# paraphrases a chunk instead of ranking it fails loudly instead of scoring itself.
LISTWISE_PROMPT = """You are ranking search results for a question about a codebase.

Question: {question}

Passages:
{passages}

Rank the passages from most to least relevant to the question. Answer with only
the identifiers in rank order, separated by " > ", for example: [3] > [1] > [2].
Include every identifier exactly once. Do not explain."""


def load_testset() -> list[dict[str, Any]]:
    with open(TESTSET_PATH) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def expected_sources_by_question(testset: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each question to its expected sources, dropping the ones the eval does not score."""
    return {q["question"]: q.get("sources", []) for q in testset if not q.get("expected_failure")}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


class CrossEncoderReranker:
    """Pairwise cross-encoder: one (question, passage) forward pass per candidate."""

    def __init__(self, model_name: str, batch_size: int, max_chars: int, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

    def rank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = [(question, truncate(c["page_content"], self.max_chars)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True)
        return [candidate for candidate, _ in ordered]


class ListwiseLLMReranker:
    """Sliding-window listwise reranker against a local Ollama model."""

    def __init__(
        self,
        model_name: str,
        window: int,
        stride: int,
        max_chars: int,
        think: bool = False,
        base_url: str = OLLAMA_URL,
    ) -> None:
        import httpx

        self.model_name = model_name
        self.window = window
        self.stride = stride
        self.max_chars = max_chars
        self.think = think
        self.base_url = base_url
        self.client = httpx.Client(timeout=900.0)

    def _rank_window(self, question: str, window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        passages = "\n\n".join(f"[{i + 1}] {truncate(c['page_content'], self.max_chars)}" for i, c in enumerate(window))
        prompt = LISTWISE_PROMPT.format(question=question, passages=passages)
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                # A reasoning model left to think spends about 188s on one window against 9s with
                # thinking off, for a ranking task where the measured orderings are comparable.
                "think": self.think,
                "options": {"temperature": 0.0},
            },
        )
        response.raise_for_status()
        return self._apply_order(window, response.json()["response"])

    def _apply_order(self, window: list[dict[str, Any]], reply: str) -> list[dict[str, Any]]:
        """Reorder by the identifiers the model returned, keeping unmentioned ones in place.

        A malformed reply degrades to the input order rather than dropping candidates,
        so a model that cannot follow the format scores as no-op instead of as damage.
        """
        seen: list[int] = []
        for token in re.findall(r"\d+", reply):
            index = int(token) - 1
            if 0 <= index < len(window) and index not in seen:
                seen.append(index)
        seen.extend(i for i in range(len(window)) if i not in seen)
        return [window[i] for i in seen]

    def rank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = list(candidates)
        start = max(0, len(ranked) - self.window)
        while True:
            end = start + self.window
            ranked[start:end] = self._rank_window(question, ranked[start:end])
            if start == 0:
                break
            start = max(0, start - self.stride)
        return ranked


def score_ranking(expected: list[str], ranked: list[dict[str, Any]], output_depth: int) -> tuple[int, float]:
    sources = [c["source"] for c in ranked[:output_depth]]
    return cast(tuple[int, float], compute_retrieval_hit_and_reciprocal_rank(expected, sources))


def run_rerank(
    reranker: CrossEncoderReranker | ListwiseLLMReranker | None,
    candidate_lists: list[dict[str, Any]],
    expected_by_question: dict[str, list[str]],
    input_depth: int,
    output_depth: int,
) -> dict[str, Any]:
    """Rescore every question's list and aggregate, alongside the unranked baseline."""
    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []

    for entry in candidate_lists:
        question = entry["question"]
        expected = expected_by_question.get(question)
        if expected is None:
            continue

        candidates = entry["candidates"][:input_depth]
        ceiling = compute_recall_at_depth(expected, [c["source"] for c in candidates])
        base_hit, base_rr = score_ranking(expected, candidates, output_depth)

        if reranker is None:
            ranked, elapsed = candidates, 0.0
        else:
            started = time.perf_counter()
            ranked = reranker.rank(question, candidates)
            elapsed = time.perf_counter() - started

        hit, rr = score_ranking(expected, ranked, output_depth)
        latencies.append(elapsed)
        per_question.append(
            {
                "question": question,
                "input_recall": ceiling,
                "baseline_hit": base_hit,
                "baseline_reciprocal_rank": base_rr,
                "hit": hit,
                "reciprocal_rank": rr,
                "latency_s": elapsed,
                "top_sources": [c["source"] for c in ranked[:output_depth]],
            }
        )

    n = len(per_question)
    if n == 0:
        raise RuntimeError("No questions scored; the candidate file and the test set do not overlap.")

    return {
        "questions_scored": n,
        "input_recall": sum(q["input_recall"] for q in per_question) / n,
        "baseline_hit_rate": sum(q["baseline_hit"] for q in per_question) / n,
        "baseline_mrr": sum(q["baseline_reciprocal_rank"] for q in per_question) / n,
        "hit_rate": sum(q["hit"] for q in per_question) / n,
        "mrr": sum(q["reciprocal_rank"] for q in per_question) / n,
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[min(n - 1, int(0.95 * n))],
            "max": max(latencies),
            "total": sum(latencies),
        },
        "per_question": per_question,
        "per_query_latency_s": latencies,
    }


def build_reranker(args: argparse.Namespace) -> CrossEncoderReranker | ListwiseLLMReranker | None:
    if args.reranker is None:
        return None
    if args.kind == "cross-encoder":
        return CrossEncoderReranker(args.reranker, args.batch_size, args.max_chars, args.device)
    return ListwiseLLMReranker(args.reranker, args.window, args.stride, args.max_chars, args.think)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True, help="Frozen candidate list to rescore.")
    parser.add_argument(
        "--reranker",
        default=None,
        help="Reranker model. Omitted scores the unranked list, which is the no-reranker control.",
    )
    parser.add_argument("--kind", choices=["cross-encoder", "llm"], default="cross-encoder")
    parser.add_argument(
        "--input-depth",
        type=int,
        default=None,
        help="Truncate the candidate list before reranking. Omitted uses the whole saved list.",
    )
    parser.add_argument(
        "--output-depth",
        type=int,
        default=DEFAULT_OUTPUT_DEPTH,
        help="Depth the reranked list is scored at, which is what the application would show.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device for a cross-encoder arm. Omitted lets sentence-transformers pick, which is "
            "MPS here. Pin to cpu to keep an arm off the GPU while a sweep is holding it, and note "
            "that a cpu-measured latency is not the latency the application would see."
        ),
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Let a reasoning model think before ranking. Roughly 20x the latency per window.",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Listwise window size.")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Listwise window stride.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Passage truncation before scoring. Keeps a listwise window inside the model's context.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidate_lists: list[dict[str, Any]] = json.load(f)

    testset = load_testset()
    expected_by_question = expected_sources_by_question(testset)
    saved_depth = max(len(entry["candidates"]) for entry in candidate_lists)
    input_depth = args.input_depth or saved_depth

    arm = (
        f"{slugify(args.reranker) if args.reranker else 'no-reranker'}"
        f"_on-{args.candidates.stem}_in{input_depth}_out{args.output_depth}"
    )
    logger.info("Reranking %s with %s (input depth %d)", args.candidates.name, args.reranker or "nothing", input_depth)

    if args.reranker and args.kind == "llm" and args.stride < 1:
        parser.error("--stride must be at least 1; a smaller value never advances the window.")
    if args.window < 1:
        parser.error("--window must be at least 1.")

    reranker = build_reranker(args)
    result = run_rerank(reranker, candidate_lists, expected_by_question, input_depth, args.output_depth)
    result |= {
        **testset_provenance(testset),
        "arm": arm,
        "candidates_file": str(args.candidates),
        "reranker": args.reranker,
        "kind": args.kind if args.reranker else None,
        "input_depth": input_depth,
        "output_depth": args.output_depth,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or RESULTS_DIR / f"{arm}.json"
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    latency = result["latency_s"]
    logger.info(
        "%s: hit %.4f (was %.4f), MRR %.4f (was %.4f), input recall %.4f, "
        "latency mean %.2fs median %.2fs p95 %.2fs max %.2fs",
        arm,
        result["hit_rate"],
        result["baseline_hit_rate"],
        result["mrr"],
        result["baseline_mrr"],
        result["input_recall"],
        latency["mean"],
        latency["median"],
        latency["p95"],
        latency["max"],
    )
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
