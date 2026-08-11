"""Deterministic retrieval regression gate.

Measures recall@k and MRR over a frozen fixture corpus and fails when either metric leaves a
recorded band. It runs the production chunker and the production retrievers, so a change to
chunking, BM25 tokenization, scoring, or rank fusion moves the numbers. It needs no embedding
model, no Qdrant, no Ollama, and no network, which is what lets it sit in the offline gate rather
than in the eval harness that runs by hand a few times a release.

What it does not measure: embedding quality. A model swap is invisible here, and the fusion arm's
vector side is a frozen ranking rather than a real vector search. The encoding-identity binding in
QdrantStore and the embedding boundary contracts cover that; this covers the parts that are exactly
reproducible.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from codebase_rag.data_ingestion.chunking import DocumentChunker
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever

# `evals/` is a sibling of `src/`, not an installed package, so the shared scorer is reached by
# path. Importing it rather than reimplementing the hit rule is the point: two definitions of
# "hit" would let the gate and the published eval numbers drift apart while both looked healthy.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.retrieval_metrics import compute_retrieval_hit_and_reciprocal_rank

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_gate"
CORPUS_DIR = FIXTURE_DIR / "corpus"
QUERIES_PATH = FIXTURE_DIR / "queries.json"
VECTOR_RANKING_PATH = FIXTURE_DIR / "vector_ranking.json"
THRESHOLDS_PATH = FIXTURE_DIR / "thresholds.json"

ARM_BM25 = "bm25"
ARM_FUSION = "fusion"
ARMS = (ARM_BM25, ARM_FUSION)


def _read_json(path: Path) -> Any:
    """Read a fixture file, failing loudly rather than degrading to an empty measurement."""
    if not path.is_file():
        raise FileNotFoundError(f"retrieval gate fixture missing: {path}")
    return json.loads(path.read_text())


def load_corpus() -> list[Document]:
    """Chunk the fixture corpus with the production chunker.

    Files are read in sorted order because BM25 breaks ties on insertion order and its scores
    depend on corpus-wide document frequencies; directory iteration order is not guaranteed, and a
    measurement that depends on it is not reproducible.
    """
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(f"retrieval gate corpus missing: {CORPUS_DIR}")

    paths = sorted(CORPUS_DIR.glob("*"))
    if not paths:
        raise FileNotFoundError(f"retrieval gate corpus is empty: {CORPUS_DIR}")

    chunker = DocumentChunker()
    documents: list[Document] = []
    for path in paths:
        documents.extend(chunker.process_file(path))

    if not documents:
        raise ValueError(f"chunking {len(paths)} fixture files produced no documents")
    return documents


class FrozenVectorRetriever:
    """Replays a recorded ranking in place of a vector search.

    Exists so the fusion arm can exercise RRF without an embedding model. It satisfies the same
    protocol `HybridRetriever` expects and touches nothing in `codebase_rag.database`, which is
    what keeps the arm offline.
    """

    def __init__(self, ranking: dict[str, list[dict[str, Any]]], documents: list[Document]) -> None:
        self._ranking = ranking
        self._by_key = {(Path(d.metadata["source"]).name, d.metadata["chunk_index"]): d for d in documents}
        self._query_to_id: dict[str, str] = {}

    def bind(self, query: str, query_id: str) -> None:
        """Associate a query string with the fixture entry that records its ranking."""
        self._query_to_id[query] = query_id

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Return the recorded ranking for this query, with descending synthetic scores."""
        entries = self._ranking[self._query_to_id[query]]
        limit = len(entries) if k is None else k
        # Scores descend from 1.0 in fixed steps. Only their order matters: RRF fuses by rank, so
        # the magnitudes are never read, and fixing them keeps the arm free of invented precision.
        return [
            (self._by_key[(entry["source"], entry["chunk_index"])], 1.0 - 0.01 * rank)
            for rank, entry in enumerate(entries[:limit])
        ]


def score_arm(retriever: Any, queries: list[dict[str, Any]], k: int) -> dict[str, float]:
    """Run every query through a retriever and return its recall@k and MRR."""
    hits: list[int] = []
    reciprocal_ranks: list[float] = []

    for entry in queries:
        results = retriever.search(entry["query"], k=k)
        retrieved = [str(doc.metadata.get("source", "")) for doc, _ in results]
        hit, reciprocal_rank = compute_retrieval_hit_and_reciprocal_rank(entry["expected_sources"], retrieved)
        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

    return {
        "recall_at_k": sum(hits) / len(hits),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def measure_all_arms() -> dict[str, dict[str, float]]:
    """Score both arms over the fixture, returning metrics per arm."""
    thresholds = _read_json(THRESHOLDS_PATH)
    queries = _read_json(QUERIES_PATH)
    ranking = _read_json(VECTOR_RANKING_PATH)
    documents = load_corpus()
    k = thresholds["k"]

    bm25 = BM25Retriever(documents)

    vector = FrozenVectorRetriever(ranking, documents)
    for entry in queries:
        vector.bind(entry["query"], entry["id"])
    fusion = HybridRetriever(vector_retriever=vector, bm25_retriever=BM25Retriever(documents))

    return {
        ARM_BM25: score_arm(bm25, queries, k),
        ARM_FUSION: score_arm(fusion, queries, k),
    }


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, float]]:
    """Both arms measured once for the module."""
    return measure_all_arms()


@pytest.fixture(scope="module")
def thresholds() -> dict[str, Any]:
    """The recorded band."""
    band: dict[str, Any] = _read_json(THRESHOLDS_PATH)
    return band


@pytest.mark.evaluation
class TestRetrievalGate:
    @pytest.mark.parametrize("arm", ARMS)
    @pytest.mark.parametrize("metric", ["recall_at_k", "mrr"])
    def test_metric_is_inside_the_recorded_band(
        self, arm: str, metric: str, measured: dict[str, dict[str, float]], thresholds: dict[str, Any]
    ) -> None:
        """Both bounds are enforced.

        The ceiling is not redundant. A change that raises these numbers has either improved
        retrieval or started matching more loosely, and those need different responses; failing on
        both edges forces the question to be asked rather than absorbed.
        """
        band = thresholds["arms"][arm][metric]
        value = measured[arm][metric]

        assert value >= band["floor"], (
            f"{arm} {metric} fell to {value:.4f}, below the recorded floor {band['floor']:.4f}. "
            f"Retrieval quality regressed; fix the cause rather than lowering the floor."
        )
        assert value <= band["ceiling"], (
            f"{arm} {metric} rose to {value:.4f}, above the recorded ceiling {band['ceiling']:.4f}. "
            f"Find out why and re-record the band deliberately."
        )

    def test_recorded_arms_match_the_arms_that_run(self, thresholds: dict[str, Any]) -> None:
        """A threshold entry for a deleted arm, or an arm with no band, is a silent hole."""
        assert set(thresholds["arms"]) == set(ARMS)

    def test_every_arm_records_both_metrics(self, thresholds: dict[str, Any]) -> None:
        for arm in ARMS:
            assert set(thresholds["arms"][arm]) == {"recall_at_k", "mrr"}, f"{arm} is missing a metric band"

    def test_measurement_is_reproducible(self) -> None:
        """Two independent measurements of an unchanged tree must agree exactly.

        Run within one test rather than across two, so an ordering dependency fails here instead of
        intermittently in CI on whichever order the runner happened to pick.
        """
        assert measure_all_arms() == measure_all_arms()

    def test_fusion_arm_loads_no_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The arm is only offline if nothing reaches for model weights."""

        def _fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("the retrieval gate must not construct an embedding model")

        monkeypatch.setattr("codebase_rag.database.embeddings.EmbeddingManager.__new__", _fail)
        measure_all_arms()

    def test_fusion_arm_depends_on_both_of_its_inputs(self) -> None:
        """Switching off either ranker must change the fusion arm's numbers.

        This is the test the first version of the fixture failed. Its frozen ranking was generated
        by a TF-IDF scorer, which agreed with BM25 on every query, so the fusion arm scored
        identically to the BM25 arm even with the vector weight at 1.0 and the BM25 weight at 0.
        Without this assertion the arm looked healthy while measuring nothing.
        """
        thresholds = _read_json(THRESHOLDS_PATH)
        queries = _read_json(QUERIES_PATH)
        ranking = _read_json(VECTOR_RANKING_PATH)
        documents = load_corpus()
        k = thresholds["k"]

        vector = FrozenVectorRetriever(ranking, documents)
        for entry in queries:
            vector.bind(entry["query"], entry["id"])

        def fused(vector_weight: float, bm25_weight: float) -> dict[str, float]:
            retriever = HybridRetriever(
                vector_retriever=vector,
                bm25_retriever=BM25Retriever(documents),
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
            return score_arm(retriever, queries, k)

        balanced = fused(0.7, 0.3)

        assert fused(1.0, 0.0) != balanced, "fusion ignores BM25: the frozen ranking dominates"
        assert fused(0.0, 1.0) != balanced, "fusion ignores the vector side: it tracks BM25 alone"

    def test_corpus_chunks_through_the_production_chunker(self) -> None:
        """The measured path includes chunking, so a chunk-boundary change moves the numbers."""
        documents = load_corpus()

        assert len(documents) > len(sorted(CORPUS_DIR.glob("*"))), "corpus produced no more chunks than files"
        assert all("chunk_index" in d.metadata for d in documents)
        assert all("start_line" in d.metadata for d in documents)

    def test_missing_fixture_fails_rather_than_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The removed predecessor skipped when its input was absent, and so never ran at all."""
        monkeypatch.setattr("tests.evaluation.test_retrieval_gate.CORPUS_DIR", CORPUS_DIR.parent / "does-not-exist")

        with pytest.raises(FileNotFoundError, match="corpus missing"):
            load_corpus()
