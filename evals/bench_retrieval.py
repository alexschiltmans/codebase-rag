"""Retrieval-only benchmark harness.

Scores a retrieval configuration (embedding model, retriever type, candidate
depth, threshold, optional reranker) against `testset.json` using hit rate,
MRR, and recall at depth. No generation, no LLM judge: these metrics come
from retrieved sources alone, so a model sweep does not need to pay for a
full ragas run per arm.

Deliberately does not import ragas, langfuse, or `create_llm_client` at
module scope, so this runs with the judge stack absent.

Usage:
    uv run python evals/bench_retrieval.py --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --retriever vector --depth 10

The benchmark builds its own Qdrant collection per embedding model
(`bench_<model-slug>_<dim>`) from the persisted BM25 corpus in
`data/cache/bm25_corpus`, rather than re-running ingestion, and tears the
collection down on exit (including on failure) unless `--keep-collections`
is passed.

It re-embeds that corpus but never re-chunks it, so scoring a model at a
different chunk size means pointing `--corpus-dir` at a corpus cut that way.
The chunking is read from the corpus rather than from the command line, and
lands in both the arm record and the arm name.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from corpus_chunking import chunking_suffix, read_chunking_sidecar
from langchain_core.documents import Document
from retrieval_metrics import compute_recall_at_depth, compute_retrieval_hit_and_reciprocal_rank
from testset_provenance import testset_provenance

from codebase_rag.database.embeddings import EmbeddingManager
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.retrieval.bm25_search import BM25Retriever, load_bm25_corpus
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent
TESTSET_PATH = EVALS_DIR / "testset.json"
CORPUS_DIR = Path("data/cache/bm25_corpus")
CANDIDATES_DIR = EVALS_DIR / "bench_candidates"

DEFAULT_BATCH_SIZE = 100


def load_testset() -> list[dict[str, Any]]:
    """Load the evaluation test set."""
    with open(TESTSET_PATH) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def slugify_model_name(model_name: str) -> str:
    """Turn an embedding model name into a filesystem/collection-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def build_vector_store(
    embedding_model: str,
    collection_name: str,
    max_seq_length: int | None = None,
    dtype: str | None = None,
) -> QdrantStore:
    """Build (or reuse) a Qdrant collection bound to the given embedding model."""
    return QdrantStore(
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_max_seq_length=max_seq_length,
        embedding_dtype=dtype,
    )


def embed_corpus(store: QdrantStore, documents: list[Document], batch_size: int = DEFAULT_BATCH_SIZE) -> float:
    """Embed and index the corpus in batches, returning the wall-clock build time in seconds.

    Logs a progress line per batch. A large model that has started swapping still makes
    progress, just orders of magnitude slower, and without a rate in the log that state is
    indistinguishable from a slow model right up until it has burned the night.
    """
    start = time.monotonic()
    total = len(documents)
    n_batches = (total + batch_size - 1) // batch_size

    for batch_index, i in enumerate(range(0, total, batch_size), start=1):
        batch = documents[i : i + batch_size]
        batch_start = time.monotonic()
        store.add_documents(batch)
        batch_s = time.monotonic() - batch_start

        done = min(i + batch_size, total)
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_s = (total - done) / rate if rate > 0 else float("inf")
        logger.info(
            "embed %d/%d chunks (batch %d/%d, %.1fs, %.1f chunks/s, eta %.0fs)",
            done,
            total,
            batch_index,
            n_batches,
            batch_s,
            rate,
            eta_s,
        )

        # MPS holds freed blocks in its allocator; across thousands of batches at a large
        # hidden size that retention is what tips a fitting model into swap.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return time.monotonic() - start


def build_retriever(
    retriever_type: str,
    store: QdrantStore | None,
    corpus: list[Document],
    threshold: float | None,
) -> Any:
    """Build the requested retriever (vector-only, BM25-only, or hybrid)."""
    bm25_retriever = BM25Retriever(corpus) if retriever_type in ("bm25", "hybrid") else None

    if retriever_type == "bm25":
        return bm25_retriever

    if store is None:
        raise ValueError(f"retriever type '{retriever_type}' requires a vector store")

    vector_retriever = VectorRetriever(store, score_threshold=threshold)
    if retriever_type == "vector":
        return vector_retriever
    if retriever_type == "hybrid":
        return HybridRetriever(vector_retriever, bm25_retriever)
    raise ValueError(f"Unknown retriever type: {retriever_type}")


def run_benchmark(
    retriever: Any,
    testset: list[dict[str, Any]],
    depth: int,
) -> dict[str, Any]:
    """Score a retriever against the test set at the given candidate depth.

    Returns a dict with aggregate metrics, per-category breakdown, per-question
    results (for the paired flip-table analysis), and the per-question candidate
    lists (for reranker arms to rescore later).
    """
    per_question: list[dict[str, Any]] = []
    candidate_lists: list[dict[str, Any]] = []
    latencies_s: list[float] = []

    for item in testset:
        # Questions flagged expected_failure have no retrievable answer by design, and the
        # published eval leaves them out of hit rate and MRR. Scoring them here would make
        # every number quietly incomparable to the ablation.
        if item.get("expected_failure", False):
            continue

        question = item["question"]
        expected_sources = item.get("sources", [])

        start = time.monotonic()
        results = retriever.search(question, k=depth)
        latencies_s.append(time.monotonic() - start)

        actual_sources = [doc.metadata.get("source", "") for doc, _ in results]
        hit, rr = compute_retrieval_hit_and_reciprocal_rank(expected_sources, actual_sources)
        # Equal to `hit` by construction here: both score the same list at the same depth.
        # Retained because the field is part of the published result shape, not because it
        # is independent evidence. The number that is genuinely a ceiling lives in
        # `rerank_bench.py`, where a depth-50 input is scored against a depth-10 output.
        recall = compute_recall_at_depth(expected_sources, actual_sources)

        per_question.append(
            {
                "question": question,
                "category": item.get("category", ""),
                "expected_sources": expected_sources,
                "actual_sources": actual_sources,
                "hit": hit,
                "reciprocal_rank": rr,
                "recall_at_depth": recall,
            }
        )
        candidate_lists.append(
            {
                "question": question,
                "candidates": [
                    {"source": doc.metadata.get("source", ""), "page_content": doc.page_content, "score": score}
                    for doc, score in results
                ],
            }
        )

    n = len(per_question) or 1
    hit_rate = sum(r["hit"] for r in per_question) / n
    mrr = sum(r["reciprocal_rank"] for r in per_question) / n
    recall_at_depth = sum(r["recall_at_depth"] for r in per_question) / n

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
        "hit_rate": hit_rate,
        "mrr": mrr,
        "recall_at_depth": recall_at_depth,
        "category_breakdown": category_breakdown,
        "per_question": per_question,
        "candidate_lists": candidate_lists,
        "avg_query_latency_s": sum(latencies_s) / len(latencies_s) if latencies_s else 0.0,
        "per_query_latency_s": latencies_s,
    }


def ingested_repositories(corpus: list[Document]) -> list[str]:
    return sorted({str(doc.metadata.get("repo", "")) for doc in corpus if doc.metadata.get("repo")})


def save_candidate_lists(candidate_lists: list[dict[str, Any]], arm_name: str) -> Path:
    """Persist a frozen first-stage candidate list so reranker arms rescore it rather than re-running retrieval."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES_DIR / f"{arm_name}.json"
    with open(path, "w") as f:
        json.dump(candidate_lists, f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--retriever", choices=["vector", "bm25", "hybrid"], default="vector")
    parser.add_argument("--depth", type=int, default=10, help="Candidate depth (k) to retrieve and score.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Relevance cutoff to apply. Omitted means no threshold: a vector arm measures "
            "ranking quality, and filtering mixes a retrieval decision into that."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help=(
            "Load precision. Omitted means whatever the checkpoint itself stores, which for some "
            "candidates is bfloat16 rather than float32; the arm records what actually loaded."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Sequence-length override. Omitted means the model's own declared value, which for "
            "some candidates is tens of thousands of tokens."
        ),
    )
    parser.add_argument("--keep-collections", action="store_true", help="Do not tear down the benchmark collection.")
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Score an existing collection instead of building one. Nothing is embedded and "
            "the collection is never torn down."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to write the result JSON.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help=(
            "Corpus to embed and score. Defaults to the application's own corpus. A corpus cut at "
            "a different chunk size lives in its own directory and carries a chunking sidecar."
        ),
    )
    args = parser.parse_args()

    testset = load_testset()
    corpus = load_bm25_corpus(args.corpus_dir)
    if not corpus:
        raise RuntimeError(f"No corpus found in {args.corpus_dir}. Run ingestion first.")
    chunking = read_chunking_sidecar(args.corpus_dir)

    store: QdrantStore | None = None
    collection_name: str | None = None
    dimension = None
    build_time_s = 0.0

    try:
        if args.retriever in ("vector", "hybrid"):
            # EmbeddingManager caches by (model, prompts, seq len), so probing the
            # dimension here doesn't cost a second model load.
            dimension = len(
                EmbeddingManager(
                    model_name=args.embedding_model,
                    max_seq_length=args.max_seq_length,
                    dtype=args.dtype,
                ).get_query_embedding("probe")
            )
            if args.collection:
                # Scoring an index someone else built: no embedding, and teardown stays off
                # so a stray run can't delete the application's own collection.
                store = build_vector_store(args.embedding_model, args.collection, args.max_seq_length, args.dtype)
                logger.info("Scoring existing collection '%s' (no index build)", args.collection)
            else:
                slug = slugify_model_name(args.embedding_model)
                # Sequence length and precision change the stored vectors, so they belong in
                # the collection name too: point IDs are deterministic, and a re-run under
                # different settings would otherwise overwrite the previous run's points in
                # place while the binding check, which compares the model, saw nothing wrong.
                # Chunking is worse than a silent overwrite. A smaller corpus does not even
                # cover the points a larger one wrote, so the stale ones survive and are
                # retrieved alongside the new, from a chunking the arm does not claim.
                collection_name = f"bench_{slug}_{dimension}"
                if args.dtype:
                    collection_name += f"_{args.dtype}"
                if args.max_seq_length:
                    collection_name += f"_seq{args.max_seq_length}"
                collection_name += chunking_suffix(chunking)
                store = build_vector_store(args.embedding_model, collection_name, args.max_seq_length, args.dtype)
                build_time_s = embed_corpus(store, corpus, batch_size=args.batch_size)

        threshold = args.threshold
        retriever = build_retriever(args.retriever, store, corpus, threshold)

        result = run_benchmark(retriever, testset, args.depth)

        manager = store.embedding_manager if store else None
        arm_record = {
            **testset_provenance(testset),
            "embedding_model": args.embedding_model,
            "dimension": dimension,
            "max_seq_length": manager.max_seq_length if manager else None,
            "dtype": manager.loaded_dtype if manager else None,
            "requested_dtype": manager.dtype if manager else None,
            "query_prompt": manager.query_prompt if manager else None,
            "document_prompt": manager.document_prompt if manager else None,
            **chunking,
            "retriever_type": args.retriever,
            "candidate_depth": args.depth,
            "applied_threshold": threshold,
            "ingested_repositories": ingested_repositories(corpus),
            "index_build_time_s": build_time_s,
            "batch_size": args.batch_size,
            "hit_rate": result["hit_rate"],
            "mrr": result["mrr"],
            "recall_at_depth": result["recall_at_depth"],
            "category_breakdown": result["category_breakdown"],
            "avg_query_latency_s": result["avg_query_latency_s"],
            "per_query_latency_s": result["per_query_latency_s"],
            "per_question": result["per_question"],
        }

        # Everything that changes the numbers goes in the name. Without dtype and sequence
        # length, an fp16 sweep silently overwrites the fp32 results it should be compared to,
        # and without chunk size a re-run at a new size overwrites the arm it should be
        # compared against, which is how a tracked result was once lost.
        arm_name = f"{slugify_model_name(args.embedding_model)}_{args.retriever}_d{args.depth}"
        if args.dtype:
            arm_name += f"_{args.dtype}"
        if args.max_seq_length:
            arm_name += f"_seq{args.max_seq_length}"
        arm_name += chunking_suffix(chunking)
        if args.threshold is not None:
            arm_name += f"_t{args.threshold}"
        if args.collection:
            arm_name = f"{arm_name}_on-{args.collection}"
        save_candidate_lists(result["candidate_lists"], arm_name)

        output_path = args.output or (EVALS_DIR / "bench_results" / f"{arm_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(arm_record, f, indent=2)
            f.write("\n")

        logger.info(
            "Arm %s: hit_rate=%.4f mrr=%.4f recall@%d=%.4f (written to %s)",
            arm_name,
            result["hit_rate"],
            result["mrr"],
            args.depth,
            result["recall_at_depth"],
            output_path,
        )
    finally:
        if store is not None and collection_name is not None and not args.keep_collections:
            try:
                store.client.delete_collection(collection_name)
                store.client.delete_collection(f"{collection_name}__meta")
            except Exception:
                logger.warning("Failed to tear down benchmark collection '%s'", collection_name)


if __name__ == "__main__":
    main()
