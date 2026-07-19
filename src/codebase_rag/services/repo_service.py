"""Repository listing service with index freshness metadata."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RepoInfo:
    name: str
    last_ingest_time: float | None
    head_sha: str | None


def _read_freshness(cache_dir: Path, repo_name: str) -> tuple[float | None, str | None]:
    """Read freshness metadata for a repo, preferring the incremental-ingest
    freshness file and falling back to the older full-cache metadata file.
    """
    freshness_path = cache_dir / f"{repo_name}_freshness.json"
    if freshness_path.exists():
        try:
            data = json.loads(freshness_path.read_text())
            return data.get("last_ingest_time"), data.get("head_sha")
        except (json.JSONDecodeError, OSError):
            pass

    meta_path = cache_dir / f"{repo_name}_cache_meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            return data.get("timestamp"), data.get("commit_sha")
        except (json.JSONDecodeError, OSError):
            pass

    return None, None


def list_repos(qdrant_store: Any, cache_dir: Path) -> list[RepoInfo]:
    """List every ingested repository with freshness metadata."""
    infos = []
    for name in qdrant_store.list_repos():
        last_ingest_time, head_sha = _read_freshness(cache_dir, name)
        infos.append(RepoInfo(name=name, last_ingest_time=last_ingest_time, head_sha=head_sha))
    return infos


def save_freshness(cache_dir: Path, repo_name: str, head_sha: str | None) -> None:
    """Persist freshness metadata for a repo after an (incremental) ingest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{repo_name}_freshness.json"
    path.write_text(json.dumps({"last_ingest_time": time.time(), "head_sha": head_sha}, indent=2))
