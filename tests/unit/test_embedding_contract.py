"""Unit tests for the embedding boundary contract and its two call sites."""

import math
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qdrant_client.models import Distance

from codebase_rag.database.embedding_contract import (
    EmbeddingContractError,
    metric_requires_unit_vectors,
    verify_vectors,
)
from codebase_rag.database.qdrant_store import QdrantStore

DIM = 8


def unit_vector(dimension: int = DIM) -> list[float]:
    """A vector of equal components with L2 norm exactly 1."""
    component = 1.0 / math.sqrt(dimension)
    return [component] * dimension


def check(vectors: list[list[float]], *, dimension: int | None = DIM, require_unit_norm: bool = True) -> None:
    """Run the contract with the boilerplate filled in."""
    verify_vectors(
        vectors,
        boundary="index",
        expected_dimension=dimension,
        model_name="test/model",
        require_unit_norm=require_unit_norm,
    )


class TestContract:
    def test_well_formed_batch_passes(self) -> None:
        check([unit_vector(), unit_vector(), unit_vector()])

    def test_empty_batch_passes(self) -> None:
        check([])

    def test_short_vector_is_rejected(self) -> None:
        with pytest.raises(EmbeddingContractError, match="has 4 components but the configured model"):
            check([unit_vector(4)])

    def test_long_vector_is_rejected(self) -> None:
        with pytest.raises(EmbeddingContractError, match="has 16 components"):
            check([unit_vector(16)])

    def test_error_names_the_offending_position_in_the_batch(self) -> None:
        with pytest.raises(EmbeddingContractError, match="vector 2 has"):
            check([unit_vector(), unit_vector(), unit_vector(4)])

    def test_nan_is_rejected(self) -> None:
        vector = unit_vector()
        vector[3] = math.nan
        with pytest.raises(EmbeddingContractError, match="non-finite value nan at position 3"):
            check([vector])

    def test_infinity_is_rejected(self) -> None:
        vector = unit_vector()
        vector[0] = math.inf
        with pytest.raises(EmbeddingContractError, match="non-finite value inf at position 0"):
            check([vector])

    def test_unnormalised_vector_is_rejected(self) -> None:
        with pytest.raises(EmbeddingContractError, match=r"L2 norm 8\.000000, not 1"):
            check([[component * 8 for component in unit_vector()]])

    def test_zero_vector_is_rejected_under_a_magnitude_sensitive_metric(self) -> None:
        with pytest.raises(EmbeddingContractError, match=r"L2 norm 0\.000000"):
            check([[0.0] * DIM])

    def test_missing_declared_dimension_skips_only_the_width_check(self) -> None:
        """A width the model does not declare is not a mismatch, but the rest still applies."""
        check([unit_vector(4)], dimension=None)

        with pytest.raises(EmbeddingContractError, match="L2 norm"):
            check([[1.0] * 4], dimension=None)

    def test_boundary_name_appears_in_the_error(self) -> None:
        with pytest.raises(EmbeddingContractError, match=r"^query:"):
            verify_vectors(
                [unit_vector(4)],
                boundary="query",
                expected_dimension=DIM,
                model_name="test/model",
                require_unit_norm=True,
            )


class TestNormTolerance:
    def test_realistic_float32_normalised_vector_passes(self) -> None:
        """A genuinely normalised vector whose norm is not exactly 1.0 must not fail.

        Built from uneven components and renormalised, so the norm lands near but not on 1 the way
        a real encode does. A tolerance chosen too tightly would fail here.
        """
        raw = [0.31, -0.72, 0.04, 0.55, -0.19, 0.88, -0.41, 0.07]
        norm = math.sqrt(sum(component * component for component in raw))
        vector = [component / norm for component in raw]

        assert vector != unit_vector(), "test vector should not be trivially uniform"
        check([vector])

    def test_a_norm_just_inside_tolerance_passes_and_just_outside_fails(self) -> None:
        inside = [component * (1 + 5e-4) for component in unit_vector()]
        outside = [component * (1 + 5e-3) for component in unit_vector()]

        check([inside])
        with pytest.raises(EmbeddingContractError, match="L2 norm"):
            check([outside])


class TestMetricConditionality:
    def test_cosine_requires_unit_vectors(self) -> None:
        assert metric_requires_unit_vectors(Distance.COSINE) is True

    @pytest.mark.parametrize("distance", [Distance.EUCLID, Distance.DOT, Distance.MANHATTAN, None])
    def test_other_metrics_do_not(self, distance: object) -> None:
        assert metric_requires_unit_vectors(distance) is False

    def test_unnormalised_passes_when_the_metric_does_not_require_unit_length(self) -> None:
        check([[component * 8 for component in unit_vector()]], require_unit_norm=False)

    def test_width_and_finiteness_still_apply_without_the_norm_requirement(self) -> None:
        with pytest.raises(EmbeddingContractError, match="components but the configured model"):
            check([unit_vector(4)], require_unit_norm=False)

        bad = unit_vector()
        bad[1] = math.nan
        with pytest.raises(EmbeddingContractError, match="non-finite"):
            check([bad], require_unit_norm=False)


def _store(client: MagicMock, embeddings: list[list[float]], query_vector: list[float] | None = None) -> QdrantStore:
    """A QdrantStore with a mocked client and an embedding manager returning fixed vectors."""
    with (
        patch("codebase_rag.database.qdrant_store.QdrantClient", return_value=client),
        patch("codebase_rag.database.qdrant_store.EmbeddingManager") as manager_cls,
    ):
        manager = manager_cls.return_value
        manager.model_name = "test/model"
        manager.model.get_sentence_embedding_dimension.return_value = DIM
        manager.get_embeddings.return_value = embeddings
        manager.get_query_embedding.return_value = query_vector if query_vector is not None else unit_vector()
        return QdrantStore()


def _existing_cosine_collection() -> MagicMock:
    """A mocked client reporting one existing collection configured for cosine."""
    client = MagicMock()
    named = MagicMock()
    named.name = "documents"
    client.get_collections.return_value = MagicMock(collections=[named])
    client.get_collection.return_value.config.params.vectors = MagicMock(size=DIM, distance=Distance.COSINE)
    client.collection_exists.return_value = False
    return client


class TestIndexBoundary:
    def test_malformed_batch_writes_nothing(self) -> None:
        client = _existing_cosine_collection()
        store = _store(client, embeddings=[unit_vector(), [component * 5 for component in unit_vector()]])

        with pytest.raises(EmbeddingContractError, match=r"^index:"):
            store.add_documents([Document(page_content="a"), Document(page_content="b")])

        client.upsert.assert_not_called()

    def test_malformed_first_batch_creates_no_collection(self) -> None:
        """The check runs before _ensure_collection, so a bad batch leaves nothing behind."""
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        client.collection_exists.return_value = False
        store = _store(client, embeddings=[unit_vector(4)])

        with pytest.raises(EmbeddingContractError):
            store.add_documents([Document(page_content="a")])

        client.create_collection.assert_not_called()

    def test_well_formed_batch_is_written(self) -> None:
        client = _existing_cosine_collection()
        store = _store(client, embeddings=[unit_vector()])

        store.add_documents([Document(page_content="a")])

        client.upsert.assert_called_once()

    def test_dropping_normalisation_upstream_is_caught(self) -> None:
        """The regression this exists to prevent: an encode path that stops normalising."""
        client = _existing_cosine_collection()
        unnormalised = [[0.9, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6]]
        store = _store(client, embeddings=unnormalised)

        with pytest.raises(EmbeddingContractError, match="L2 norm"):
            store.add_documents([Document(page_content="a")])


class TestQueryBoundary:
    def test_malformed_query_vector_never_reaches_the_store(self) -> None:
        client = _existing_cosine_collection()
        store = _store(client, embeddings=[], query_vector=[c * 3 for c in unit_vector()])
        store._model_binding_verified = True

        with pytest.raises(RuntimeError, match="query:"):
            store.similarity_search_with_score("anything")

        client.query_points.assert_not_called()

    def test_well_formed_query_vector_proceeds(self) -> None:
        client = _existing_cosine_collection()
        client.query_points.return_value = MagicMock(points=[])
        store = _store(client, embeddings=[], query_vector=unit_vector())
        store._model_binding_verified = True

        assert store.similarity_search_with_score("anything") == []
        client.query_points.assert_called_once()


class TestEncodingIdentityCarriesRevision:
    def test_revision_is_part_of_the_recorded_identity(self) -> None:
        client = _existing_cosine_collection()
        store = _store(client, embeddings=[])
        store.embedding_manager.revision = "abc123"

        assert store._encoding_identity()["revision"] == "abc123"

    def test_a_different_revision_is_refused(self) -> None:
        client = _existing_cosine_collection()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [MagicMock(payload={"embedding_model": "test/model", "revision": "old-sha"})]
        store = _store(client, embeddings=[])
        store.embedding_manager.revision = "new-sha"

        with pytest.raises(ValueError, match="revision: recorded 'old-sha', configured 'new-sha'"):
            store._verify_model_binding()

    def test_a_collection_predating_the_revision_record_is_accepted(self) -> None:
        """Absent is not mismatched, or every collection written before this change would fail."""
        client = _existing_cosine_collection()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [MagicMock(payload={"embedding_model": "test/model"})]
        store = _store(client, embeddings=[])
        store.embedding_manager.revision = "new-sha"

        store._verify_model_binding()

        assert store._model_binding_verified is True
