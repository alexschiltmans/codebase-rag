"""Evaluation runner for the Codebase RAG system.

Runs the RAG chain against a curated test set and scores results using ragas.
Optionally logs scores to Langfuse.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --langfuse  # also log to Langfuse
"""

import json
import logging
import sys
import time
from pathlib import Path

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
from codebase_rag.retrieval.vector_search import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent
TESTSET_PATH = EVALS_DIR / "testset.json"
RESULTS_PATH = EVALS_DIR / "results.json"


def load_testset() -> list[dict]:
    """Load the evaluation test set."""
    with open(TESTSET_PATH) as f:
        return json.load(f)


def build_rag_chain() -> RAGChain:
    """Initialize the RAG chain with live services."""
    config = Config.get_instance()

    qdrant_store = QdrantStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.collection_name,
    )
    if not qdrant_store.collection_exists():
        raise RuntimeError("Qdrant collection does not exist. Run ingestion first.")

    vector_retriever = VectorRetriever(qdrant_store)

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
        retriever=vector_retriever,
        llm=llm,
        top_k=5,
        use_conversation_memory=False,
        min_relevance_score=0.15,
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


def compute_custom_metrics(results: list[dict]) -> dict:
    """Compute custom keyword-based metrics (no LLM judge required)."""
    keyword_recalls = []
    source_precisions = []

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

    return {
        "avg_keyword_recall": sum(keyword_recalls) / len(keyword_recalls) if keyword_recalls else 0,
        "avg_source_precision": sum(source_precisions) / len(source_precisions) if source_precisions else 0,
        "questions_answered": sum(1 for r in results if not r.get("error")),
        "questions_failed": sum(1 for r in results if r.get("error")),
        "avg_latency_s": sum(r.get("elapsed", 0) for r in results if not r.get("error"))
        / max(1, sum(1 for r in results if not r.get("error"))),
    }


def run_ragas_evaluation(results: list[dict]) -> dict:
    """Run ragas evaluation metrics on the results.

    Uses the local Ollama LLM as the judge model via LangchainLLMWrapper.
    Returns the ragas scores dict.
    """
    config = Config.get_instance()

    # Use local Ollama as judge LLM
    judge_llm = ChatOllama(
        model=config.llm_model_name,
        base_url=config.ollama_base_url,
        temperature=0.0,
        timeout=300,
    )
    wrapped_llm = LangchainLLMWrapper(judge_llm)

    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

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
        return {}

    eval_dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=wrapped_llm),
        AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
        ContextRecall(llm=wrapped_llm),
    ]

    logger.info("Running ragas evaluation with %d samples...", len(samples))
    try:
        run_config = RunConfig(timeout=600, max_retries=2, max_wait=120)

        eval_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_embeddings,
            raise_exceptions=False,
            show_progress=True,
            run_config=run_config,
        )
        # Extract scores from the pandas DataFrame
        df = eval_result.to_pandas()
        score_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
        scores = {}
        for col in score_cols:
            vals = df[col].dropna()
            if not vals.empty:
                scores[col] = round(vals.mean(), 4)
        logger.info("ragas scores: %s", scores)
        return scores
    except Exception as e:
        logger.error("ragas evaluation failed: %s", e)
        return {"ragas_error": str(e)}


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


def generate_results_markdown(results: list[dict], custom_metrics: dict, ragas_scores: dict) -> str:
    """Generate a markdown report from the evaluation results."""
    lines = ["# Evaluation Results\n"]
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Test set:** {len(results)} questions\n")

    # Overall metrics
    lines.append("## Custom Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k, v in custom_metrics.items():
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

    if ragas_scores and "ragas_error" not in ragas_scores:
        lines.append("\n## RAGAS Scores\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for k, v in ragas_scores.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    elif ragas_scores.get("ragas_error"):
        lines.append(f"\n## RAGAS Scores\n\nFailed: {ragas_scores['ragas_error']}\n")

    # Per-question breakdown
    lines.append("\n## Per-Question Breakdown\n")
    lines.append("| # | Difficulty | Category | Keyword Recall | Docs | Latency | Expected Failure |")
    lines.append("|---|-----------|----------|----------------|------|---------|------------------|")

    for i, r in enumerate(results):
        if r.get("error"):
            exp_fail = r.get("expected_failure", False)
            diff = r.get("difficulty", "")
            cat = r.get("category", "")
            lines.append(f"| {i + 1} | {diff} | {cat} | ERROR | - | - | {exp_fail} |")
            continue
        keywords = r.get("keywords", [])
        answer_lower = r["answer"].lower()
        kr = sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords) if keywords else 0
        docs = len(r.get("contexts", []))
        lat = r.get("elapsed", 0)
        exp_fail = r.get("expected_failure", False)
        diff = r.get("difficulty", "")
        cat = r.get("category", "")
        lines.append(f"| {i + 1} | {diff} | {cat} | {kr:.2f} | {docs} | {lat:.1f}s | {exp_fail} |")

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


def main() -> None:
    """Run the full evaluation pipeline."""
    use_langfuse = "--langfuse" in sys.argv

    logger.info("Loading test set from %s", TESTSET_PATH)
    testset = load_testset()
    logger.info("Loaded %d test questions", len(testset))

    logger.info("Initializing RAG chain...")
    rag_chain = build_rag_chain()

    logger.info("Running RAG on test set...")
    results = run_rag_on_testset(rag_chain, testset)

    logger.info("Computing custom metrics...")
    custom_metrics = compute_custom_metrics(results)
    logger.info("Custom metrics: %s", custom_metrics)

    logger.info("Running ragas evaluation...")
    ragas_scores = run_ragas_evaluation(results)

    # Save raw results
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "results": results,
                "custom_metrics": custom_metrics,
                "ragas_scores": ragas_scores,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Raw results saved to %s", RESULTS_PATH)

    # Generate markdown report
    md = generate_results_markdown(results, custom_metrics, ragas_scores)
    md_path = EVALS_DIR / "results.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("Markdown report saved to %s", md_path)

    if use_langfuse:
        log_to_langfuse(results, custom_metrics, ragas_scores)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions: {len(results)}")
    print(f"Answered:  {custom_metrics['questions_answered']}")
    print(f"Failed:    {custom_metrics['questions_failed']}")
    print(f"Avg keyword recall:   {custom_metrics['avg_keyword_recall']:.4f}")
    print(f"Avg source precision: {custom_metrics['avg_source_precision']:.4f}")
    print(f"Avg latency:          {custom_metrics['avg_latency_s']:.1f}s")
    if ragas_scores and "ragas_error" not in ragas_scores:
        print("\nRAGAS scores:")
        for k, v in ragas_scores.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
