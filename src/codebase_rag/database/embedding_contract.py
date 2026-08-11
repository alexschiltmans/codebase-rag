"""Checks on the embeddings themselves, at the points where they enter and query the store.

Every way an embedding can be wrong here is silent. Qdrant rejects a vector of the wrong width, but
only after the encode has run; a vector that is the right width and the wrong shape is accepted
without complaint. Under cosine, an unnormalised vector ranks by magnitude rather than direction, so
long chunks drift to the top and results get worse with nothing raised. A NaN compares false against
everything and simply disappears from results.

This is deliberately not a quality check. A well-formed vector from a bad model passes, and should:
that is what the retrieval gate and the eval harness are for. This only catches vectors that are
malformed for the store they are going into.
"""

import math

# Normalised float32 vectors do not come back at exactly 1.0. This tolerance is far outside the
# accumulation error of summing 768 to 4096 squared components, and far inside the gap between a
# normalised vector and an unnormalised one, whose norm for a typical transformer output is order
# 1 to 10 rather than 1.001. It is sized to catch the structural error and nothing else.
NORM_TOLERANCE = 1e-3

# The metrics whose ranking depends on vector magnitude, so that unit length is part of the
# contract. Named by their Qdrant string values, so this module needs no qdrant_client import and
# stays testable on plain lists.
MAGNITUDE_SENSITIVE_METRICS = frozenset({"Cosine"})


class EmbeddingContractError(ValueError):
    """Raised when a vector violates the contract for the store it is entering."""


def metric_requires_unit_vectors(distance: object) -> bool:
    """Report whether a collection's distance metric assumes unit-length vectors.

    Accepts the raw value Qdrant reports, which is an enum whose `str` is its wire name. Anything
    unrecognised is treated as not requiring normalisation, because inventing a requirement for a
    metric this project does not create would fail a collection that is working.
    """
    if distance is None:
        return False
    name = getattr(distance, "value", distance)
    return str(name) in MAGNITUDE_SENSITIVE_METRICS


def verify_vectors(
    vectors: list[list[float]],
    *,
    boundary: str,
    expected_dimension: int | None,
    model_name: str,
    require_unit_norm: bool,
) -> None:
    """Check a batch of embeddings against the contract, raising on the first violation.

    Every vector is checked rather than a sample. A batch of 100 at 768 dimensions is 76,800 float
    operations against an encode that just ran a transformer over 100 chunks, so sampling would buy
    nothing measurable while making the check probabilistic, which is a poor property for something
    whose only job is to be a guarantee.

    Args:
        vectors: The embeddings about to be written or used for a search.
        boundary: Which boundary is checking, named in any error so a failure says where it came from.
        expected_dimension: The configured model's declared width, or None when it declares none.
        model_name: The configured model, named in any error.
        require_unit_norm: Whether the target collection's metric assumes unit-length vectors.

    Raises:
        EmbeddingContractError: On the first vector that violates the contract.
    """
    for index, vector in enumerate(vectors):
        _verify_one(
            vector,
            index=index,
            boundary=boundary,
            expected_dimension=expected_dimension,
            model_name=model_name,
            require_unit_norm=require_unit_norm,
        )


def _verify_one(
    vector: list[float],
    *,
    index: int,
    boundary: str,
    expected_dimension: int | None,
    model_name: str,
    require_unit_norm: bool,
) -> None:
    """Check one embedding, raising on the first property it violates."""
    # A width the model does not declare is skipped rather than guessed. `_verify_dimension` takes
    # the same position: two pieces of configuration that cannot both be read are not a mismatch.
    if isinstance(expected_dimension, int) and len(vector) != expected_dimension:
        raise EmbeddingContractError(
            f"{boundary}: vector {index} has {len(vector)} components but the configured model "
            f"'{model_name}' produces {expected_dimension}. The collection and the model disagree "
            f"about vector width, so nothing indexed under one is comparable with the other."
        )

    for position, component in enumerate(vector):
        if not math.isfinite(component):
            raise EmbeddingContractError(
                f"{boundary}: vector {index} has a non-finite value {component!r} at position "
                f"{position}. Similarity against it is meaningless, and it would rank as no match "
                f"against everything rather than failing."
            )

    if not require_unit_norm:
        return

    # fsum rather than sum, so the check does not contribute the drift it is measuring.
    norm = math.sqrt(math.fsum(component * component for component in vector))
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise EmbeddingContractError(
            f"{boundary}: vector {index} has L2 norm {norm:.6f}, not 1 within {NORM_TOLERANCE}. "
            f"The collection compares with a magnitude-sensitive metric, under which an "
            f"unnormalised vector ranks by length rather than direction and degrades retrieval "
            f"silently instead of failing."
        )
