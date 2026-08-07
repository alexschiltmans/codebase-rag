"""Record how a persisted BM25 corpus was cut, so a reader can tell which chunking produced it.

A corpus is chunked once, by whichever embedding model is configured at ingest time, and anything
that later scores or re-embeds it inherits that chunking without being told what it was. Nothing on
disk said, which made two populations of results indistinguishable and let a re-run at a new chunk
size overwrite the run it should have been compared against.

The chunking travels with the corpus rather than with the caller: a label passed on the command line
is a label the caller can get wrong, and a mislabelled corpus is worse than an unlabelled one
because it looks authoritative.

This lives beside the ingestion path rather than in `evals/` because the ingestion path writes the
corpus the application serves from, and a sidecar only the benchmark harness knew how to write would
leave that one corpus, the most-used of them all, permanently unrecorded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The sidecar sits one level down rather than beside the corpus files. `load_bm25_corpus` globs
# `*.json` in the corpus directory and reads every match as a list of documents, so a sidecar next
# to them is not ignored, it is a crash. The glob does not recurse, so a subdirectory is invisible
# to it while the sidecar still travels with the corpus it describes.
SIDECAR_DIR = "_meta"
SIDECAR_NAME = "chunking.json"

# A corpus written before the sidecar existed. Distinct from any recorded value, and not a
# default: "unknown" and "the same as ours" are the two things this file exists to separate.
UNRECORDED: dict[str, Any] = {"chunk_size": None, "chunk_overlap": None, "chunk_max_seq_length": None}


def write_chunking_sidecar(
    corpus_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_seq_length: int | None,
) -> Path:
    """Record the chunking a corpus directory was built with.

    Args:
        corpus_dir: The corpus directory, which is created if it does not exist.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap in characters.
        max_seq_length: The token window the size was derived from, or None when the
            size was chosen explicitly rather than derived.

    Returns:
        The path written.
    """
    meta_dir = corpus_dir / SIDECAR_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / SIDECAR_NAME
    payload = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_max_seq_length": max_seq_length,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def read_chunking_sidecar(corpus_dir: Path) -> dict[str, Any]:
    """Return the chunking a corpus was built with, or nulls when it predates the sidecar.

    A missing sidecar is reported as unrecorded rather than guessed at. The chunk size could be
    inferred from the longest chunk present, but a corpus whose files are all shorter than the
    limit would report a size well below the configured one, and that inference is silently wrong
    exactly when the corpus is small.
    """
    path = corpus_dir / SIDECAR_DIR / SIDECAR_NAME
    if not path.exists():
        return dict(UNRECORDED)

    with open(path) as f:
        payload: dict[str, Any] = json.load(f)

    return {field: payload.get(field) for field in UNRECORDED}
