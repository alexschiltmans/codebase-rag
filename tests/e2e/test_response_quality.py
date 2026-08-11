"""End-to-end check that the RAG chain answers questions over a real index.

Needs a live LLM backend and an ingested BM25 cache, so it sits in the e2e tier
rather than the offline evaluation tier.

This asserts that the chain works, not how good its answers are. Answer quality
is measured by `evals/run_eval.py` against a curated test set, which is the only
place the numbers mean anything.
"""

import re
from pathlib import Path
from typing import Any

import pytest

from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from tests.evaluation.test_dataset import get_test_dataset

QUESTION_COUNT = 5


def evaluate_response(response: str, expected_keywords: list[str], question: str) -> dict[str, Any]:
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


@pytest.mark.e2e
def test_rag_response_answers_over_real_index() -> None:
    """Every question routed through the chain comes back answered and sourced."""
    bm25_path = Path("./data/cache/bm25_retriever.json")
    if not bm25_path.exists():
        pytest.skip(f"BM25 retriever file {bm25_path} not found")

    try:
        retriever = BM25Retriever.load_json(bm25_path)
        llm = create_llm_client()
        rag_chain = RAGChain(retriever=retriever, llm=llm)
    except Exception as e:
        pytest.skip(f"Failed to initialize RAG chain: {e}")

    questions = get_test_dataset()[:QUESTION_COUNT]
    assert questions, "Test dataset is empty"

    for question_data in questions:
        question = question_data["question"]
        response_data = rag_chain.run(question)

        answer = response_data["answer"]
        assert answer.strip(), f"Empty answer for {question!r}"
        assert response_data["sources"], f"No sources returned for {question!r}"

        # Keyword coverage is deliberately not asserted: the shared dataset ships placeholder keywords,
        # so it would score zero regardless of how the chain performs.
        metrics = evaluate_response(answer, question_data["keywords"], question)
        assert metrics["on_topic"], f"Answer is off-topic for {question!r} (overlap {metrics['keyword_overlap']:.2f})"
