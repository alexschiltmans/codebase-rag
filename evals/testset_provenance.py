"""Stamp every saved arm with which test set it was scored against.

Arms accumulate in `bench_results/` and `rerank_results/` under names that encode the model,
retriever and depth but nothing about the test set. Two files whose names differ only by an
embedder can therefore have been scored against different question sets, and a reader comparing
them has no way to tell. The size catches a grown test set; the hash catches an edit that kept
the count the same, which is the case a size alone reads as comparable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def testset_provenance(testset: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the scored part of a test set as a size and a content hash.

    Args:
        testset: The loaded test set, including any `expected_failure` questions.

    Returns:
        `{"testset_size": ..., "testset_hash": ...}`, over the questions that are actually
        scored. Only the question text and its expected sources feed the hash, because those
        are the only fields retrieval scoring reads; rewording an expected answer leaves the
        arms comparable and should not look like a new test set.
    """
    scored = [q for q in testset if not q.get("expected_failure")]
    payload = sorted([q["question"], sorted(q.get("sources", []))] for q in scored)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"testset_size": len(scored), "testset_hash": digest[:12]}
