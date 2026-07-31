"""Evaluation of retrieval quality."""

import pickle
from pathlib import Path

import pytest

from codebase_rag.retrieval.hybrid_search import HybridRetriever
from tests.evaluation.test_dataset import get_test_dataset


def load_retriever(pickle_path: str) -> HybridRetriever:
    """Load a retriever from a pickle file.

    Args:
        pickle_path: Path to the pickled retriever.

    Returns:
        HybridRetriever: The loaded retriever.
    """
    with open(pickle_path, "rb") as f:
        retriever: HybridRetriever = pickle.load(f)
        return retriever


def calculate_keyword_recall(keywords: list[str], content: str) -> float:
    """Calculate what fraction of keywords appear in the content.

    Args:
        keywords: List of keywords to look for.
        content: Content to search in.

    Returns:
        float: Recall score (0.0-1.0).
    """
    if not keywords:
        return 0.0

    content = content.lower()
    matches = sum(1 for kw in keywords if kw.lower() in content)
    return matches / len(keywords)


def calculate_source_precision(expected_sources: list[str], actual_sources: list[str]) -> float:
    """Calculate precision of retrieved sources.

    Args:
        expected_sources: List of expected source keywords.
        actual_sources: List of actual sources retrieved.

    Returns:
        float: Precision score (0.0-1.0).
    """
    if not actual_sources:
        return 0.0

    expected_lower = [s.lower() for s in expected_sources]
    actual_lower = [s.lower() for s in actual_sources]

    # Count matches - a source matches if any expected keyword is in it
    matches = sum(1 for src in actual_lower if any(exp in src for exp in expected_lower))

    return matches / len(actual_lower)


@pytest.mark.evaluation
@pytest.mark.parametrize(
    "retriever_path",
    [
        "./data/cache/hybrid_retriever.pkl",
    ],
)
def test_retrieval_quality(retriever_path: str) -> None:
    """Evaluate retrieval quality on the test dataset."""
    resolved_path = Path(retriever_path)
    if not resolved_path.exists():
        pytest.skip(f"Retriever file {resolved_path} not found")

    try:
        retriever = load_retriever(str(resolved_path))
    except Exception as e:
        pytest.fail(f"Failed to load retriever: {e}")

    test_questions = get_test_dataset()

    recalls = []
    precisions = []

    for question_data in test_questions:
        question = question_data["question"]
        expected_keywords = question_data["keywords"]
        expected_sources = question_data["sources"]

        results = retriever.search(question, k=5)

        retrieved_content = " ".join(doc.page_content for doc, _ in results)
        retrieved_sources = [doc.metadata.get("file_name", doc.metadata.get("source", "unknown")) for doc, _ in results]

        recall = calculate_keyword_recall(expected_keywords, retrieved_content)
        precision = calculate_source_precision(expected_sources, retrieved_sources)

        recalls.append(recall)
        precisions.append(precision)

        assert recall >= 0.0, "Recall should be non-negative"
        assert precision >= 0.0, "Precision should be non-negative"

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    assert avg_recall >= 0.5, "Average recall is too low"
    assert avg_precision >= 0.3, "Average precision is too low"
    assert avg_precision >= 0.3, "Average precision is too low"
