"""Report how much of a corpus a model would silently cut at its sequence limit.

A chunk size that scores better while pushing content past the embedding model's token limit is not
a better chunk size, it is the same measurement with part of the corpus missing. Chunk size moved to
614 characters because 31.34% of chunks exceeded the limit at 1000, so a sweep that reports only hit
rate and MRR would re-argue a question that was already settled on different grounds.

Counting is the tokenizer's, through the same `count_tokens` the ingest path uses, so the document
prompt and special tokens come out of the same budget here as they do in production.

Usage:
    uv run python evals/corpus_truncation.py --corpus-dir data/cache/bm25_corpus_chunk1000 \\
        --model sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from corpus_chunking import read_chunking_sidecar

from codebase_rag.data_ingestion.truncation import format_truncation_report, measure_truncation
from codebase_rag.database.embeddings import EmbeddingManager
from codebase_rag.retrieval.bm25_search import load_bm25_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-dir", type=Path, action="append", required=True, help="Corpus to measure, repeatable."
    )
    parser.add_argument("--model", required=True, help="Embedding model whose tokenizer and limit to measure against.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Sequence-length override for the model.")
    parser.add_argument("--dtype", default=None, help="Load precision. Does not affect token counts.")
    args = parser.parse_args()

    manager = EmbeddingManager(model_name=args.model, max_seq_length=args.max_seq_length, dtype=args.dtype)
    limit = manager.max_seq_length

    for corpus_dir in args.corpus_dir:
        documents = load_bm25_corpus(corpus_dir)
        if not documents:
            raise SystemExit(f"No corpus found in {corpus_dir}.")

        chunking = read_chunking_sidecar(corpus_dir)
        report = measure_truncation(documents, manager.count_tokens, limit)

        logger.info(
            "%s (chunk size %s): %d/%d chunks over the %d-token limit, %.2f%%",
            corpus_dir,
            chunking["chunk_size"] or "not recorded",
            report.over_limit,
            report.chunks,
            limit,
            report.share,
        )
        for line in format_truncation_report(report):
            logger.info("  %s", line)


if __name__ == "__main__":
    main()
