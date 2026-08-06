"""Record how a benchmark corpus was cut, so an arm can say which chunking produced it.

A corpus is chunked once, at ingest time, by whichever embedding model is configured. Every
benchmark arm then re-embeds that same corpus, so a model sweep varies the embedder while holding
the previous model's chunk size fixed. Nothing in a saved arm said which chunking it ran under,
which made two populations of results indistinguishable on disk and let a re-run at a new chunk
size overwrite the arm it should have been compared against.

The chunking travels with the corpus rather than with the caller: a label passed on the command
line is a label the caller can get wrong, and a mislabelled arm is worse than an unlabelled one
because it looks authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def chunking_suffix(chunking: dict[str, Any]) -> str:
    """Return the arm-name fragment for a chunking, empty when it is unrecorded.

    An unrecorded chunking adds nothing to the name, so arms saved before this existed keep the
    names they already have and are not silently renamed into a population they do not belong to.
    """
    chunk_size = chunking.get("chunk_size")
    return f"_chunk{chunk_size}" if chunk_size else ""
