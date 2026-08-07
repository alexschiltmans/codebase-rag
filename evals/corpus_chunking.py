"""Name a benchmark arm after the chunking its corpus was cut at.

The sidecar itself belongs to the ingestion path, which writes the corpus the application serves
from; this adds only the piece that is benchmark-specific, which is how a chunking turns into a
fragment of an arm's filename. Re-exported here so a reader following `bench_retrieval.py`'s imports
finds the whole mechanism in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.data_ingestion.corpus_chunking import (
    SIDECAR_DIR,
    SIDECAR_NAME,
    UNRECORDED,
    read_chunking_sidecar,
    write_chunking_sidecar,
)

__all__ = [
    "SIDECAR_DIR",
    "SIDECAR_NAME",
    "UNRECORDED",
    "chunking_suffix",
    "read_chunking_sidecar",
    "write_chunking_sidecar",
]


def chunking_suffix(chunking: dict[str, Any]) -> str:
    """Return the arm-name fragment for a chunking, empty when it is unrecorded.

    An unrecorded chunking adds nothing to the name, so arms saved before this existed keep the
    names they already have and are not silently renamed into a population they do not belong to.
    """
    chunk_size = chunking.get("chunk_size")
    return f"_chunk{chunk_size}" if chunk_size else ""
