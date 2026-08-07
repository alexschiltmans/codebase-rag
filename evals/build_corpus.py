"""Build a benchmark corpus at an explicit chunk size.

`bench_retrieval.py` re-embeds a persisted corpus but never re-chunks it, so a model sweep varies
the embedder while holding whatever chunk size the corpus was cut at. Measuring chunk size therefore
needs a corpus per size, produced before the benchmark runs.

The size is given rather than derived. `derive_chunk_size` multiplies a model's token window by 1.6,
which for an uncapped Qwen3-Embedding-0.6B is 52428 characters, and a sweep needs points chosen by
the experiment rather than by whichever model happens to be configured.

Chunking runs against the existing checkout under `data/repos/`, through the ingestion path's own
document loading, so the corpus differs from the application's only in how it was cut. The output
goes to its own directory and the application's corpus is never written.

Usage:
    uv run python evals/build_corpus.py --chunk-size 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from corpus_chunking import write_chunking_sidecar

from codebase_rag.config import Config
from codebase_rag.data_ingestion.chunking import DocumentChunker
from codebase_rag.data_ingestion.document_processor import DocumentProcessor
from codebase_rag.data_ingestion.git_loader import GitLoader
from codebase_rag.data_ingestion.pipeline import discover_included_dirs
from codebase_rag.retrieval.bm25_search import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path("data/cache/bm25_corpus")

# The pipeline's own defaults. Passing them explicitly keeps this corpus matched to the
# application's rather than to whatever the constructor defaults drift to.
DEFAULT_INCLUDED_DIRS = ["docs", "src", "tests"]
DEFAULT_INCLUDED_FILES = ["README.md", "pyproject.toml"]


def repos_in(corpus_dir: Path) -> list[str]:
    """Return the repo names a corpus directory holds, one per JSON file."""
    return sorted(path.stem for path in corpus_dir.glob("*.json"))


def build_corpus(repo_names: list[str], chunk_size: int, out_dir: Path) -> dict[str, int]:
    """Chunk each repo's checkout at `chunk_size` and write it to `out_dir`.

    Returns:
        Chunk count per repo name.
    """
    config = Config.get_instance()
    out_dir.mkdir(parents=True, exist_ok=True)

    # A repo file left behind by an earlier, wider build is not inert: `load_bm25_corpus` merges
    # every JSON in the directory, so the arm would score a corpus mixing two builds while its
    # sidecar names one chunking. Only files this build is about to replace may survive.
    stale = [path for path in out_dir.glob("*.json") if path.stem not in repo_names]
    if stale:
        raise SystemExit(
            f"{out_dir} already holds {', '.join(sorted(path.name for path in stale))} from a build "
            "that covered other repositories. Loading merges every file in the directory, so delete "
            "them or build into a fresh directory."
        )

    chunker = DocumentChunker(chunk_size=chunk_size)
    logger.info(
        "Chunking at %d characters with %d overlap",
        chunker.chunk_size,
        chunker.chunk_overlap,
    )

    counts: dict[str, int] = {}
    for repo_name in repo_names:
        local_path = config.repo_local_path / repo_name
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"No checkout for '{repo_name}' at {local_path}. The sweep re-chunks an existing "
                "clone rather than cloning, so ingest the repo before building a corpus for it."
            )

        included_dirs = discover_included_dirs(local_path, DEFAULT_INCLUDED_DIRS)
        processor = DocumentProcessor(
            git_loader=GitLoader(local_path=local_path),
            document_chunker=chunker,
        )
        documents = processor.process(included_dirs=included_dirs, included_files=DEFAULT_INCLUDED_FILES)

        # list_repos() and the per-repo corpus split both key off this, and the processor does not
        # set it: the pipeline tags it after processing.
        for doc in documents:
            doc.metadata["repo"] = repo_name

        BM25Retriever(documents).save_json(out_dir / f"{repo_name}.json")
        counts[repo_name] = len(documents)
        logger.info("%s: %d chunks", repo_name, len(documents))

    write_chunking_sidecar(
        out_dir,
        chunk_size=chunker.chunk_size,
        chunk_overlap=chunker.chunk_overlap,
        # The size was chosen for the sweep, not derived from any model's window. Recording a
        # window here would attribute the size to a model it was never measured against.
        max_seq_length=None,
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-size", type=int, required=True, help="Chunk size in characters.")
    parser.add_argument(
        "--repo",
        action="append",
        default=None,
        help=(
            "Repo to chunk, repeatable. Defaults to whatever the application's corpus holds, so a "
            "swept corpus covers the same repositories as the one it will be compared against."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination. Defaults to data/cache/bm25_corpus_chunk<size>.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise SystemExit(f"chunk size must be positive, got {args.chunk_size}")

    repo_names = args.repo or repos_in(DEFAULT_CORPUS_DIR)
    if not repo_names:
        raise SystemExit(f"No repositories found in {DEFAULT_CORPUS_DIR} and none given with --repo.")

    out_dir = args.out_dir or Path(f"data/cache/bm25_corpus_chunk{args.chunk_size}")
    if out_dir.resolve() == DEFAULT_CORPUS_DIR.resolve():
        raise SystemExit(
            f"Refusing to write {out_dir}: that is the corpus the application serves from. "
            "A swept corpus goes in its own directory."
        )

    start = time.monotonic()
    counts = build_corpus(repo_names, args.chunk_size, out_dir)
    elapsed = time.monotonic() - start

    total = sum(counts.values())
    logger.info("Wrote %d chunks across %d repo(s) to %s in %.1fs", total, len(counts), out_dir, elapsed)


if __name__ == "__main__":
    main()
