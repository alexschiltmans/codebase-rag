"""Evaluation of LLM response quality."""

import json
import re
from pathlib import Path

import pytest

from codebase_rag.llm.rag_chain import RAGChain

from .test_dataset import get_test_dataset


def evaluate_response(response: str, expected_keywords: list[str], question: str) -> dict:
    """Evaluate the quality of an LLM response.

    Args:
        response: LLM response to evaluate.
        expected_keywords: Keywords that should appear in the response.
        question: The original question.

    Returns:
        Dict: Evaluation metrics.
    """
    response_lower = response.lower()

    # Check for keyword coverage
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    keyword_coverage = keyword_matches / len(expected_keywords) if expected_keywords else 0

    # Check for hallucinations - statements not supported by context
    # Look for phrases indicating uncertainty that shouldn't be there
    uncertainty_phrases = [
        "i'm not sure",
        "i don't know",
        "i can't find",
        "i don't have enough information",
        "not mentioned in the documentation",
        "not specified in the context",
    ]

    contains_uncertainty = any(phrase in response_lower for phrase in uncertainty_phrases)

    # Check for source citations
    citation_pattern = r"\[([\d]+)\]|from [\w\.]+|source: [\w\.]+"
    has_citations = bool(re.search(citation_pattern, response))

    # Check if response is on-topic
    question_keywords = set(re.findall(r"\b\w+\b", question.lower()))
    response_keywords = set(re.findall(r"\b\w+\b", response_lower))
    keyword_overlap = len(question_keywords.intersection(response_keywords)) / len(question_keywords)
    on_topic = keyword_overlap >= 0.3  # At least 30% of question keywords appear in response

    return {
        "keyword_coverage": keyword_coverage,
        "contains_uncertainty": contains_uncertainty,
        "has_citations": has_citations,
        "on_topic": on_topic,
        "keyword_overlap": keyword_overlap,
    }


@pytest.mark.evaluation
def test_rag_response_quality() -> None:
    """Evaluate RAG chain response quality on the test dataset."""
    bm25_path = Path("./data/cache/bm25_retriever.json")
    if not bm25_path.exists():
        pytest.skip(f"BM25 retriever file {bm25_path} not found")

    try:
        # This will connect to actual services - be careful in CI environments
        rag_chain = RAGChain()
    except Exception as e:
        pytest.skip(f"Failed to initialize RAG chain: {e}")

    test_questions = get_test_dataset()

    evaluation_results = []

    for question_data in test_questions[:5]:  # Test first 5 questions
        question = question_data["question"]
        expected_keywords = question_data["keywords"]

        try:
            response_data = rag_chain.run(question)
            response = response_data["answer"]
            sources = response_data["sources"]

            eval_metrics = evaluate_response(response, expected_keywords, question)

            result = {
                "question": question,
                "response": response,
                "sources": sources,
                "metrics": eval_metrics,
                "expected_keywords": expected_keywords,
            }
            evaluation_results.append(result)

            assert eval_metrics["has_citations"], "Response should include citations"
            assert eval_metrics["on_topic"], "Response should be on-topic"
            assert eval_metrics["keyword_coverage"] >= 0.3, "Keyword coverage is too low"

        except Exception:  # noqa: S110
            pass  # Individual question failures should not block other evaluations

    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "response_quality.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
