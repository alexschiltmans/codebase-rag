"""Evaluation runner for the Codebase RAG system.

Runs the RAG chain against a curated test set and scores results using ragas.
Optionally logs scores to Langfuse.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --langfuse  # also log to Langfuse

    # By default RAGAS judges answers with the same model that generated them
    # (self-judged, caveated in the reports). Pass a fixed, larger judge model
    # via --judge-model or RAGAS_JUDGE_MODEL to avoid that. Local models are
    # capped at 9B, so 9B is the largest judge to use here:
    uv run python evals/run_eval.py --judge-model qwen3.5:9b

    # Judge concurrency is bounded to what one local Ollama serves in parallel
    # (default 1) via --max-workers or RAGAS_MAX_WORKERS; a run whose coverage
    # falls below --min-coverage/RAGAS_MIN_COVERAGE (default 0.9) fails rather
    # than publishing. Disable a metric outright with a repeatable --skip-metric:
    uv run python evals/run_eval.py --max-workers 2 --skip-metric context_recall

    # Per-judge-job timeout (both the Ollama client and ragas's RunConfig),
    # default 1200s: --judge-timeout <seconds> or RAGAS_JUDGE_TIMEOUT.
    uv run python evals/run_eval.py --judge-timeout 1800

Operational precondition: this harness must not share its Ollama instance with
the running app or another eval — both compete for the same single-threaded
server and inflate every latency figure with queueing time. Run
`docker stop codebase-rag-app` before an eval to guarantee that.

Judge reasoning is explicitly disabled (`reasoning=False` on the judge
ChatOllama, see `run_ragas_evaluation`) regardless of which judge model is
picked. A "thinking" model like `qwen3.5:9b` otherwise burns most of a judge
call's time on a chain-of-thought that ragas discards, and on CPU-only Docker
Ollama (no GPU passthrough) that turned one judge job into ~10 minutes with a
~7h50m projection for a single retriever's judging.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langfuse import Langfuse
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever as Bm25Index
from codebase_rag.retrieval.bm25_search import load_bm25_corpus
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VECTOR_SCORE_THRESHOLD, VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent
TESTSET_PATH = EVALS_DIR / "testset.json"

RETRIEVER_TYPES = ("vector", "bm25", "hybrid")


def build_ragas_metrics(wrapped_llm: Any, wrapped_embeddings: Any) -> list:
    """Build the ragas metrics this harness scores, in report order.

    Single source of truth for the metric set: `RAGAS_METRIC_NAMES` (the
    all-skip guard) and `run_ragas_evaluation` both derive from this list, so
    adding or removing a metric cannot leave the guard or the coverage gate
    out of sync with what actually runs. Passing `None` for the judge and
    embeddings is safe when only the metric names are needed.
    """
    return [
        Faithfulness(llm=wrapped_llm),
        AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
        ContextRecall(llm=wrapped_llm),
    ]


RAGAS_METRIC_NAMES = frozenset(m.name for m in build_ragas_metrics(None, None))


def _resolve_config_value[T](flag: str, env_var: str, default: T, cast: Callable[[str], T]) -> T:
    """Resolve a config value: `--flag value` or `--flag=value` in argv, then `env_var`, then `default`."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
        if arg.startswith(f"{flag}="):
            return cast(arg.split("=", 1)[1])
    env_val = os.getenv(env_var)
    if env_val:
        return cast(env_val)
    return default


def resolve_judge_model_name(generation_model_name: str) -> str:
    """Determine which model should judge the RAGAS metrics.

    Judging your own outputs with the same (often small) model that generated
    them adds self-preference bias and, for a 350M model, questionable
    competence as a judge in the first place. Prefer a fixed, larger model via
    `--judge-model <name>` or the `RAGAS_JUDGE_MODEL` env var. If neither is
    set, falls back to the generation model — callers must caveat scores in
    that case (see `is_self_judged` usage in `main()`).

    Args:
        generation_model_name: The model used to generate the answers being judged.
    """
    return _resolve_config_value("--judge-model", "RAGAS_JUDGE_MODEL", generation_model_name, str)


def resolve_max_workers() -> int:
    """Determine how many concurrent judge calls RAGAS may issue.

    The eval targets one local Ollama container that serves generation
    requests sequentially (`OLLAMA_NUM_PARALLEL=1` by default); RAGAS's own
    default of 16 concurrent jobs would queue behind each other and share a
    single deadline, timing out in batches instead of measuring per-job work.
    Configurable via `--max-workers <n>` or `RAGAS_MAX_WORKERS`, default 1.
    """
    return _resolve_config_value("--max-workers", "RAGAS_MAX_WORKERS", 1, int)


def resolve_min_coverage() -> float:
    """Determine the minimum completed-job share required to publish a metric.

    Below this share of `attempted` judge jobs completing, a metric's score
    is not a trustworthy average and the run should fail rather than publish
    it. Configurable via `--min-coverage <fraction>` or `RAGAS_MIN_COVERAGE`,
    default 0.9.
    """
    return _resolve_config_value("--min-coverage", "RAGAS_MIN_COVERAGE", 0.9, float)


def resolve_judge_timeout_s() -> int:
    """Determine the per-judge-job timeout, in seconds, for both the Ollama client and ragas's `RunConfig`.

    A 9B-class judge model run CPU-only in Docker (no GPU passthrough on this
    Mac) is far slower per call than the 350M generation model; a single
    ragas job chains several LLM calls (statement generation, NLI
    classification, output-format retries) that each take real time even
    with `reasoning=False`. Configurable via `--judge-timeout <seconds>` or
    `RAGAS_JUDGE_TIMEOUT`, default 1200 (20 minutes) — chosen after the
    previous 600s default was hit mid-call by `qwen3.5:9b` with reasoning
    still enabled (see `run_ragas_evaluation`'s `reasoning=False`).
    """
    return _resolve_config_value("--judge-timeout", "RAGAS_JUDGE_TIMEOUT", 1200, int)


def resolve_skip_metrics() -> set[str]:
    """Determine which metrics are explicitly disabled for this run.

    Repeatable `--skip-metric <name>` flag. A skipped metric is absent from
    both `ragas_scores` and the coverage gate, distinguishing "not asked for"
    from "could not measure".
    """
    skipped = set()
    for i, arg in enumerate(sys.argv):
        if arg == "--skip-metric" and i + 1 < len(sys.argv):
            skipped.add(sys.argv[i + 1])
    return skipped


def load_testset() -> list[dict]:
    """Load the evaluation test set."""
    with open(TESTSET_PATH) as f:
        return json.load(f)


def build_retriever(retriever_type: str, qdrant_store: QdrantStore) -> Any:
    """Build the requested retriever (vector-only, BM25-only, or hybrid).

    The hybrid arm applies `VECTOR_SCORE_THRESHOLD` to its vector component,
    matching the configuration `AppRuntime` ships — otherwise the ablation
    would measure a retrieval setup no user actually runs. The vector-only
    arm stays unthresholded on purpose: it isolates the embedding model's
    raw ranking quality, independent of the production relevance cutoff.

    Args:
        retriever_type: One of "vector", "bm25", "hybrid".
        qdrant_store: The Qdrant store backing vector search.

    Returns:
        A retriever exposing a `search(query, k)` method.
    """
    if retriever_type == "vector":
        return VectorRetriever(qdrant_store)

    cache_dir = Path("data/cache")
    corpus = load_bm25_corpus(cache_dir / "bm25_corpus")
    if not corpus:
        raise RuntimeError("No BM25 corpus found in data/cache/bm25_corpus. Run ingestion first.")
    bm25_retriever = Bm25Index(corpus)
    if retriever_type == "bm25":
        return bm25_retriever
    if retriever_type == "hybrid":
        vector_retriever = VectorRetriever(qdrant_store, score_threshold=VECTOR_SCORE_THRESHOLD)
        return HybridRetriever(vector_retriever, bm25_retriever)
    raise ValueError(f"Unknown retriever type: {retriever_type}")


def build_rag_chain(retriever_type: str = "hybrid") -> RAGChain:
    """Initialize the RAG chain with live services.

    Args:
        retriever_type: One of "vector", "bm25", "hybrid" — which retriever
            backs the chain. Defaults to "hybrid", matching the shipped app.
    """
    config = Config.get_instance()

    qdrant_store = QdrantStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.collection_name,
    )
    if not qdrant_store.collection_exists():
        raise RuntimeError("Qdrant collection does not exist. Run ingestion first.")

    retriever = build_retriever(retriever_type, qdrant_store)

    llm = OllamaClient(
        model_name=config.llm_model_name,
        base_url=config.ollama_base_url,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        max_tokens=512,
        timeout=120,
    )
    status = llm.check_connection()
    if status["status"] != "connected":
        raise RuntimeError(f"Cannot connect to Ollama: {status['message']}")

    return RAGChain(
        retriever=retriever,
        llm=llm,
        top_k=5,
        use_conversation_memory=False,
        prompt_budget_chars=llm.prompt_budget_chars,
    )


def run_rag_on_testset(rag_chain: RAGChain, testset: list[dict]) -> list[dict]:
    """Run the RAG chain on each test question and collect results."""
    results = []
    for i, item in enumerate(testset):
        question = item["question"]
        logger.info("(%d/%d) %s", i + 1, len(testset), question)
        start = time.time()
        try:
            output = rag_chain.run(question)
            elapsed = time.time() - start
            contexts = [doc.page_content for doc in output.get("documents", [])]
            results.append(
                {
                    "question": question,
                    "answer": output["answer"],
                    "contexts": contexts,
                    "expected_answer": item.get("expected_answer", ""),
                    "keywords": item.get("keywords", []),
                    "sources_expected": item.get("sources", []),
                    "sources_actual": [s.get("file_path", "") for s in output.get("sources", [])],
                    "difficulty": item.get("difficulty", ""),
                    "category": item.get("category", ""),
                    "expected_failure": item.get("expected_failure", False),
                    "metrics": output.get("metrics", {}),
                    "elapsed": elapsed,
                }
            )
            logger.info(
                "  -> %.1fs, %d docs retrieved, answer length %d", elapsed, len(contexts), len(output["answer"])
            )
        except Exception as e:
            logger.error("  -> FAILED: %s", e)
            results.append(
                {
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "contexts": [],
                    "expected_answer": item.get("expected_answer", ""),
                    "keywords": item.get("keywords", []),
                    "difficulty": item.get("difficulty", ""),
                    "category": item.get("category", ""),
                    "expected_failure": item.get("expected_failure", False),
                    "error": str(e),
                }
            )
    return results


def compute_retrieval_hit_and_reciprocal_rank(expected: list[str], actual: list[str]) -> tuple[int, float]:
    """Score one question's retrieval against its expected sources, independent of the generated answer.

    Args:
        expected: Expected source patterns (e.g. `"enum.py"`), matched as
            case-insensitive substrings, same convention as source precision.
        actual: Retrieved document paths, in rank order.

    Returns:
        `(hit, reciprocal_rank)` — hit is 1 if any expected source matches any
        retrieved document, else 0; reciprocal_rank is `1 / (1-based rank of
        the first match)`, or 0 if there is no match.
    """
    expected_lower = [s.lower() for s in expected]
    for rank, src in enumerate(actual, start=1):
        src_lower = src.lower()
        if any(exp in src_lower for exp in expected_lower):
            return 1, 1 / rank
    return 0, 0.0


def compute_custom_metrics(results: list[dict]) -> dict:
    """Compute custom keyword-based metrics (no LLM judge required)."""
    keyword_recalls = []
    source_precisions = []
    hit_rates = []
    reciprocal_ranks = []

    for r in results:
        if r.get("error"):
            continue
        # Keyword recall: fraction of expected keywords found in answer
        answer_lower = r["answer"].lower()
        keywords = r.get("keywords", [])
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
            keyword_recalls.append(matches / len(keywords))

        # Source precision: fraction of retrieved sources matching expected patterns
        expected = r.get("sources_expected", [])
        actual = r.get("sources_actual", [])
        if actual and expected:
            expected_lower = [s.lower() for s in expected]
            matching = sum(1 for src in actual if any(exp in src.lower() for exp in expected_lower))
            source_precisions.append(matching / len(actual))

        # Hit rate / MRR: retrieval-only, scored against sources_expected regardless of the answer
        if expected and not r.get("expected_failure", False):
            hit, reciprocal_rank = compute_retrieval_hit_and_reciprocal_rank(expected, actual)
            hit_rates.append(hit)
            reciprocal_ranks.append(reciprocal_rank)

    return {
        "avg_keyword_recall": sum(keyword_recalls) / len(keyword_recalls) if keyword_recalls else 0,
        "avg_source_precision": sum(source_precisions) / len(source_precisions) if source_precisions else 0,
        "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
        "avg_mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "questions_answered": sum(1 for r in results if not r.get("error")),
        "questions_failed": sum(1 for r in results if r.get("error")),
        "avg_latency_s": sum(r.get("elapsed", 0) for r in results if not r.get("error"))
        / max(1, sum(1 for r in results if not r.get("error"))),
    }


def compute_ragas_scores_and_coverage(df: Any) -> tuple[dict[str, float | None], dict[str, dict[str, int]]]:
    """Derive per-metric scores and judge-job coverage from a ragas result DataFrame.

    Args:
        df: `EvaluationResult.to_pandas()` output — one row per sample, judge
            metric columns alongside `user_input`/`response`/`retrieved_contexts`/
            `reference`, NaN in a metric's column for a failed judge job.

    Returns:
        `(scores, coverage)`. `scores` maps metric -> rounded mean, or `None` if
        every job for that metric failed. `coverage` maps metric ->
        `{attempted, completed, failed}` job counts.
    """
    score_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    scores: dict[str, float | None] = {}
    coverage: dict[str, dict[str, int]] = {}
    for col in score_cols:
        attempted = len(df[col])
        vals = df[col].dropna()
        completed = len(vals)
        coverage[col] = {"attempted": attempted, "completed": completed, "failed": attempted - completed}
        scores[col] = round(vals.mean(), 4) if not vals.empty else None
    return scores, coverage


def check_coverage_gate(
    ragas_coverage: dict[str, dict[str, int]],
    min_coverage: float,
    requested_metrics: set[str] | None = None,
) -> str | None:
    """Return the name of the first metric below the coverage threshold, or `None` if all pass.

    Args:
        ragas_coverage: metric -> `{attempted, completed, failed}`, as returned by
            `compute_ragas_scores_and_coverage`. A skipped metric is absent here
            entirely and so cannot fail the gate.
        min_coverage: Minimum required `completed / attempted` share.
        requested_metrics: Metric names this run was supposed to measure (not
            skipped). A requested metric missing from `ragas_coverage` entirely
            — the judge phase failed wholesale, or produced no samples to score
            — also fails the gate; an empty `ragas_coverage` only means "pass"
            when nothing was requested in the first place.
    """
    for metric_name in sorted(requested_metrics or set()):
        if metric_name not in ragas_coverage:
            return metric_name
    for metric_name, counts in ragas_coverage.items():
        attempted = counts["attempted"]
        share = counts["completed"] / attempted if attempted else 0.0
        if share < min_coverage:
            return metric_name
    return None


def run_ragas_evaluation(
    results: list[dict],
    judge_model_name: str,
    max_workers: int,
    skip_metrics: set[str],
    judge_timeout_s: int,
) -> dict:
    """Run ragas evaluation metrics on the results.

    Args:
        results: Output of `run_rag_on_testset`.
        judge_model_name: Ollama model to use as the RAGAS judge. See
            `resolve_judge_model_name` — this may or may not be the same
            model that generated the answers being judged.
        max_workers: Maximum concurrent judge calls. See `resolve_max_workers`.
        skip_metrics: Metric names to exclude entirely. See `resolve_skip_metrics`.
        judge_timeout_s: Per-call and per-job timeout in seconds. See
            `resolve_judge_timeout_s`.

    Returns:
        A dict with `scores` (metric -> score, `None` for an enabled metric
        with zero completed jobs), `coverage` (metric ->
        `{attempted, completed, failed}`), and `requested_metrics` (the metric
        names that weren't skipped — what this run was supposed to measure,
        used by `check_coverage_gate` to catch a wholesale judge-phase failure
        that leaves `coverage` empty).

    The judge LLM is built with `reasoning=False`. For a "thinking" model
    (e.g. `qwen3.5:9b`), Ollama's default is to emit a full chain-of-thought
    before the structured answer ragas actually parses; ragas never reads
    that chain-of-thought, so keeping it only costs latency. Measured on this
    CPU-only Docker Ollama (no GPU passthrough), one judge call with
    reasoning left on took ~10 minutes and still hit the previous 600s
    `RunConfig` timeout mid-thought, projecting ~7h50m for one retriever's 48
    judge jobs. `reasoning=False` is harmless for non-reasoning judge models
    too — Ollama ignores `think` for models that don't support it.
    """
    config = Config.get_instance()

    judge_llm = ChatOllama(
        model=judge_model_name,
        base_url=config.ollama_base_url,
        temperature=0.0,
        timeout=judge_timeout_s,
        reasoning=False,
    )
    wrapped_llm = LangchainLLMWrapper(judge_llm)

    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    all_metrics = build_ragas_metrics(wrapped_llm, wrapped_embeddings)
    metrics = [m for m in all_metrics if m.name not in skip_metrics]
    requested_metrics = {m.name for m in metrics}
    if skip_metrics:
        logger.info("Skipping metrics: %s", sorted(skip_metrics))

    # Defensive: main() already rejects an all-skip run before the loop; this guards direct callers too.
    if not metrics:
        logger.info("All ragas metrics skipped — nothing to judge")
        return {"scores": {}, "coverage": {}, "requested_metrics": requested_metrics}

    # Build evaluation dataset from results
    samples = []
    for r in results:
        if r.get("error"):
            continue
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r.get("contexts", []),
            reference=r.get("expected_answer", ""),
        )
        samples.append(sample)

    if not samples:
        logger.warning("No valid samples for ragas evaluation")
        return {"scores": {}, "coverage": {}, "requested_metrics": requested_metrics}

    eval_dataset = EvaluationDataset(samples=samples)

    logger.info("Running ragas evaluation with %d samples, max_workers=%d...", len(samples), max_workers)
    try:
        run_config = RunConfig(timeout=judge_timeout_s, max_retries=2, max_wait=120, max_workers=max_workers)

        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
            raise_exceptions=False,
            show_progress=True,
            run_config=run_config,
        )
        # Extract scores and per-metric judge-job coverage from the pandas DataFrame
        df = eval_result.to_pandas()
        scores, coverage = compute_ragas_scores_and_coverage(df)
        logger.info("ragas scores: %s", scores)
        logger.info("ragas coverage: %s", coverage)
        return {"scores": scores, "coverage": coverage, "requested_metrics": requested_metrics}
    except Exception as e:
        logger.error("ragas evaluation failed: %s", e)
        return {"scores": {"ragas_error": str(e)}, "coverage": {}, "requested_metrics": requested_metrics}


def log_to_langfuse(results: list[dict], custom_metrics: dict, ragas_scores: dict) -> None:
    """Log evaluation scores to Langfuse."""
    config = Config.get_instance()
    if not config.langfuse_enabled:
        logger.info("Langfuse not enabled, skipping logging")
        return

    try:
        lf = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )

        # Log overall evaluation trace
        trace = lf.trace(
            name="rag-evaluation",
            input={"testset_size": len(results)},
            output={
                "custom_metrics": custom_metrics,
                "ragas_scores": ragas_scores,
            },
        )

        # Log individual question scores
        for r in results:
            if r.get("error"):
                continue
            keywords = r.get("keywords", [])
            answer_lower = r["answer"].lower()
            keyword_recall = sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords) if keywords else 0

            trace.span(
                name="eval-question",
                input={"question": r["question"]},
                output={
                    "answer": r["answer"],
                    "keyword_recall": round(keyword_recall, 4),
                    "difficulty": r.get("difficulty", ""),
                    "category": r.get("category", ""),
                    "latency_s": round(r.get("elapsed", 0), 2),
                    "docs_retrieved": len(r.get("contexts", [])),
                },
            )

        lf.flush()
        logger.info("Evaluation scores logged to Langfuse")
    except Exception as e:
        logger.warning("Failed to log to Langfuse: %s", e)


def generate_results_markdown(
    results: list[dict],
    custom_metrics: dict,
    ragas_scores: dict,
    ragas_coverage: dict,
    judge_model_name: str,
    is_self_judged: bool,
    latency_probe_s: float,
) -> str:
    """Generate a markdown report from the evaluation results."""
    lines = ["# Evaluation Results\n"]
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Test set:** {len(results)} questions\n")
    lines.append(
        f"**Latency probe:** {latency_probe_s:.2f}s (single generation timed before the test set ran; "
        "compare `avg_latency_s` only against runs with a similar probe — a high probe means the "
        "run was contended)\n"
    )

    # Overall metrics
    lines.append("## Custom Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k, v in custom_metrics.items():
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

    if ragas_scores and "ragas_error" not in ragas_scores:
        lines.append(f"\n## RAGAS Scores (judge: `{judge_model_name}`)\n")
        if is_self_judged:
            lines.append(
                "> ⚠️ **Self-judged.** No `--judge-model`/`RAGAS_JUDGE_MODEL` was set, so the "
                f"same model that generated these answers (`{judge_model_name}`) also scored them. "
                "This adds self-preference bias, and a model this size is a weak judge to begin "
                "with — treat these numbers as indicative at best. The custom keyword recall / "
                "source precision metrics above don't use an LLM judge and are more trustworthy.\n"
            )
        lines.append("| Metric | Score | Coverage |")
        lines.append("|--------|-------|----------|")
        for k, v in ragas_scores.items():
            score_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            cov = ragas_coverage.get(k)
            cov_str = f"{cov['completed']}/{cov['attempted']}" if cov else "-"
            lines.append(f"| {k} | {score_str} | {cov_str} |")
    elif ragas_scores.get("ragas_error"):
        lines.append(f"\n## RAGAS Scores\n\nFailed: {ragas_scores['ragas_error']}\n")

    # Per-question breakdown
    lines.append("\n## Per-Question Breakdown\n")
    lines.append("| # | Difficulty | Category | Hit | RR | Keyword Recall | Docs | Latency | Expected Failure |")
    lines.append("|---|-----------|----------|-----|----|-----------------|------|---------|------------------|")

    for i, r in enumerate(results):
        if r.get("error"):
            exp_fail = r.get("expected_failure", False)
            diff = r.get("difficulty", "")
            cat = r.get("category", "")
            lines.append(f"| {i + 1} | {diff} | {cat} | - | - | ERROR | - | - | {exp_fail} |")
            continue
        keywords = r.get("keywords", [])
        answer_lower = r["answer"].lower()
        kr = sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords) if keywords else 0
        docs = len(r.get("contexts", []))
        lat = r.get("elapsed", 0)
        exp_fail = r.get("expected_failure", False)
        diff = r.get("difficulty", "")
        cat = r.get("category", "")
        expected = r.get("sources_expected", [])
        if expected and not exp_fail:
            hit, rr = compute_retrieval_hit_and_reciprocal_rank(expected, r.get("sources_actual", []))
            hit_str, rr_str = str(hit), f"{rr:.2f}"
        else:
            hit_str, rr_str = "-", "-"
        lines.append(
            f"| {i + 1} | {diff} | {cat} | {hit_str} | {rr_str} | {kr:.2f} | {docs} | {lat:.1f}s | {exp_fail} |"
        )

    # Failure cases
    failures = [r for r in results if r.get("error") or r.get("expected_failure")]
    if failures:
        lines.append("\n## Failure Cases\n")
        for r in failures:
            lines.append(f"### Q: {r['question']}\n")
            if r.get("error"):
                lines.append(f"**Error:** {r['error']}\n")
            if r.get("expected_failure"):
                lines.append(f"**Expected failure:** {r.get('failure_reason', 'Yes')}\n")
            lines.append(f"**Answer:** {r.get('answer', 'N/A')}\n")

    return "\n".join(lines)


def generate_ablation_markdown(all_metrics: dict[str, dict], testset: list[dict]) -> str:
    """Generate a markdown ablation report comparing retriever configurations.

    Args:
        all_metrics: Mapping of retriever type ("vector", "bm25", "hybrid") to
            its custom_metrics dict from `compute_custom_metrics`.
        testset: The loaded test set, used to report its exact-term vs
            conceptual composition.
    """
    conceptual_count = sum(1 for item in testset if item.get("category") == "conceptual")
    exact_term_count = len(testset) - conceptual_count

    lines = ["# Retrieval Ablation\n"]
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(
        "Same test set (`evals/testset.json`), same LLM, same top_k — only the retriever "
        "feeding the RAG chain changes. Full per-question detail for each configuration is "
        "in `results_<retriever>.md`.\n"
    )
    lines.append(
        f"Test set composition: {exact_term_count} exact-term (keyword/lookup) questions, "
        f"{conceptual_count} conceptual/paraphrased questions ({len(testset)} total). The "
        "conceptual questions avoid quoting source identifiers, so a retriever's hit rate "
        "on them reflects semantic matching rather than keyword overlap.\n"
    )
    lines.append(
        f"The hybrid arm applies the production cosine relevance cutoff "
        f"(`VECTOR_SCORE_THRESHOLD={VECTOR_SCORE_THRESHOLD}`) to its vector component, matching "
        "the app's shipped configuration. The vector-only arm is unthresholded to isolate raw "
        "embedding ranking quality; BM25 scores are never thresholded (zero-overlap documents "
        "are excluded by construction).\n"
    )
    lines.append(
        "Avg Latency figures are comparable only across runs with similar latency probes — see "
        "each configuration's `results_<retriever>.md` for its probe.\n"
    )
    lines.append("| Retriever | Hit Rate | MRR | Keyword Recall | Source Precision | Answered | Failed | Avg Latency |")
    lines.append(
        "|-----------|----------|-----|----------------|-------------------|----------|--------|-------------|"
    )
    for retriever_type in RETRIEVER_TYPES:
        m = all_metrics[retriever_type]
        lines.append(
            f"| {retriever_type} | {m['avg_hit_rate']:.4f} | {m['avg_mrr']:.4f} | "
            f"{m['avg_keyword_recall']:.4f} | {m['avg_source_precision']:.4f} | "
            f"{m['questions_answered']} | {m['questions_failed']} | {m['avg_latency_s']:.1f}s |"
        )
    return "\n".join(lines)


def publish_retriever_results(
    evals_dir: Path,
    retriever_type: str,
    results: list[dict],
    custom_metrics: dict,
    ragas_scores: dict,
    ragas_coverage: dict,
    requested_metrics: set[str],
    latency_probe_s: float,
    judge_model_name: str,
    is_self_judged: bool,
    min_coverage: float,
) -> None:
    """Gate on judge coverage, then write the JSON and markdown reports for one retriever.

    Checks `check_coverage_gate` before any write. On a gate failure, logs the
    failing metric — with its counts if it has any, or the wholesale-failure
    reason from `ragas_scores` if it doesn't — and exits the process non-zero
    without writing `results_<retriever_type>.{json,md}`, leaving whatever was
    previously published there untouched.
    """
    failed_metric = check_coverage_gate(ragas_coverage, min_coverage, requested_metrics)
    if failed_metric:
        counts = ragas_coverage.get(failed_metric)
        if counts is None:
            reason = ragas_scores.get("ragas_error", "the judge phase produced no results to measure coverage from")
            logger.error(
                "Coverage gate failed for retriever=%s metric=%s: %s. Not writing results.",
                retriever_type,
                failed_metric,
                reason,
            )
        else:
            attempted = counts["attempted"]
            share = counts["completed"] / attempted if attempted else 0.0
            logger.error(
                "Coverage gate failed for retriever=%s metric=%s: %d/%d completed (%.1f%%), "
                "below the configured minimum coverage of %.2f. Not writing results.",
                retriever_type,
                failed_metric,
                counts["completed"],
                attempted,
                share * 100,
                min_coverage,
            )
        sys.exit(1)

    results_path = evals_dir / f"results_{retriever_type}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "retriever": retriever_type,
                "results": results,
                "custom_metrics": custom_metrics,
                "latency_probe_s": round(latency_probe_s, 4),
                "ragas_scores": ragas_scores,
                "ragas_coverage": ragas_coverage,
                "ragas_judge_model": judge_model_name,
                "ragas_self_judged": is_self_judged,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Raw results saved to %s", results_path)

    md = generate_results_markdown(
        results, custom_metrics, ragas_scores, ragas_coverage, judge_model_name, is_self_judged, latency_probe_s
    )
    md_path = evals_dir / f"results_{retriever_type}.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("Markdown report saved to %s", md_path)


def main() -> None:
    """Run the full evaluation pipeline across every retriever configuration.

    Runs the test set once per retriever ("vector", "bm25", "hybrid") so the
    hybrid retriever the app actually ships can be compared against its two
    components, and writes a combined ablation report.
    """
    use_langfuse = "--langfuse" in sys.argv

    logger.info("Loading test set from %s", TESTSET_PATH)
    testset = load_testset()
    logger.info("Loaded %d test questions", len(testset))

    config = Config.get_instance()
    judge_model_name = resolve_judge_model_name(config.llm_model_name)
    is_self_judged = judge_model_name == config.llm_model_name
    max_workers = resolve_max_workers()
    min_coverage = resolve_min_coverage()
    skip_metrics = resolve_skip_metrics()
    judge_timeout_s = resolve_judge_timeout_s()
    logger.info(
        "Resolved eval config: max_workers=%d, min_coverage=%.2f, skip_metrics=%s, judge_timeout_s=%d",
        max_workers,
        min_coverage,
        sorted(skip_metrics),
        judge_timeout_s,
    )
    if skip_metrics >= RAGAS_METRIC_NAMES:
        logger.error(
            "--skip-metric disables every ragas metric (%s) — nothing would be judged. "
            "Skip fewer metrics, or drop --skip-metric entirely if you don't want ragas at all.",
            sorted(RAGAS_METRIC_NAMES),
        )
        sys.exit(1)
    if is_self_judged:
        logger.warning(
            "No --judge-model/RAGAS_JUDGE_MODEL set — RAGAS will judge '%s' with itself. "
            "Scores will be marked self-judged in the reports; pass a fixed, larger judge "
            "model to avoid self-preference bias.",
            judge_model_name,
        )
    else:
        logger.info("Using '%s' as a fixed RAGAS judge model", judge_model_name)

    all_custom_metrics: dict[str, dict] = {}

    for retriever_type in RETRIEVER_TYPES:
        logger.info("=== Retriever: %s ===", retriever_type)
        rag_chain = build_rag_chain(retriever_type)

        logger.info("Timing latency probe...")
        probe_start = time.time()
        rag_chain.run("What does this repository do?")
        latency_probe_s = time.time() - probe_start
        logger.info("Latency probe: %.2fs", latency_probe_s)

        logger.info("Running RAG on test set...")
        results = run_rag_on_testset(rag_chain, testset)

        logger.info("Computing custom metrics...")
        custom_metrics = compute_custom_metrics(results)
        logger.info("Custom metrics: %s", custom_metrics)
        all_custom_metrics[retriever_type] = custom_metrics

        logger.info("Running ragas evaluation...")
        ragas_result = run_ragas_evaluation(results, judge_model_name, max_workers, skip_metrics, judge_timeout_s)
        ragas_scores = ragas_result["scores"]
        ragas_coverage = ragas_result["coverage"]
        requested_metrics = ragas_result["requested_metrics"]

        publish_retriever_results(
            EVALS_DIR,
            retriever_type,
            results,
            custom_metrics,
            ragas_scores,
            ragas_coverage,
            requested_metrics,
            latency_probe_s,
            judge_model_name,
            is_self_judged,
            min_coverage,
        )

        if use_langfuse:
            log_to_langfuse(results, custom_metrics, ragas_scores)

        print("\n" + "=" * 60)
        print(f"EVALUATION SUMMARY — {retriever_type}")
        print("=" * 60)
        print(f"Questions: {len(results)}")
        print(f"Answered:  {custom_metrics['questions_answered']}")
        print(f"Failed:    {custom_metrics['questions_failed']}")
        print(f"Avg hit rate:          {custom_metrics['avg_hit_rate']:.4f}")
        print(f"Avg MRR:               {custom_metrics['avg_mrr']:.4f}")
        print(f"Avg keyword recall:   {custom_metrics['avg_keyword_recall']:.4f}")
        print(f"Avg source precision: {custom_metrics['avg_source_precision']:.4f}")
        print(f"Avg latency:          {custom_metrics['avg_latency_s']:.1f}s")
        print(f"Latency probe:        {latency_probe_s:.1f}s")
        if ragas_scores and "ragas_error" not in ragas_scores:
            print(f"\nRAGAS scores (judge: {judge_model_name}):")
            if is_self_judged:
                print("  WARNING: self-judged — same model generated and scored these answers.")
            for k, v in ragas_scores.items():
                cov = ragas_coverage.get(k)
                cov_str = f" ({cov['completed']}/{cov['attempted']})" if cov else ""
                print(f"  {k}: {v:.4f}{cov_str}" if isinstance(v, float) else f"  {k}: {v}{cov_str}")
        print("=" * 60)

    ablation_md = generate_ablation_markdown(all_custom_metrics, testset)
    ablation_path = EVALS_DIR / "ablation.md"
    with open(ablation_path, "w") as f:
        f.write(ablation_md)
    logger.info("Ablation report saved to %s", ablation_path)
    print("\n" + ablation_md)


if __name__ == "__main__":
    main()
