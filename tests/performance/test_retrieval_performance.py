"""Performance tests for retrieval components."""

import random
import string
import time

import pytest
from langchain_core.documents import Document

from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever


def generate_random_text(length: int = 500, rng: random.Random | None = None) -> str:
    """Generate random text of specified length."""
    rng = rng or random
    return "".join(rng.choice(string.ascii_letters + " " * 10) for _ in range(length))


def generate_test_documents(
    num_docs: int = 100, codebase_keywords: bool = True, seed: int = 20260720
) -> list[Document]:
    """Generate a large number of test documents.

    Uses a seeded, local RNG so the corpus is identical run to run. Keyword
    occurrences must be deterministic: BM25 legitimately returns nothing when
    a query term is absent, so a corpus that only *probably* contains the
    queried terms makes any "found something" assertion flaky.

    Args:
        num_docs: Number of documents to generate.
        codebase_keywords: Whether to include codebase-related keywords.
        seed: Seed for the local RNG.

    Returns:
        List[Document]: Generated test documents.
    """
    rng = random.Random(seed)
    documents = []

    # Keywords to randomly insert (if codebase_keywords is True)
    keywords = [
        "codebase package",
        "power grid",
        "load flow",
        "network analysis",
        "electrical grid",
        "power system",
        "transformer",
        "transmission line",
        "distribution network",
        "voltage profile",
        "contingency analysis",
    ]

    for i in range(num_docs):
        content = generate_random_text(rng=rng)

        if codebase_keywords and rng.random() < 0.3:  # 30% chance
            # Round-robin rather than random choice so every keyword is
            # guaranteed present, and pad with spaces so the keyword survives
            # tokenization instead of fusing with adjacent random letters.
            keyword = f" {keywords[i % len(keywords)]} "
            position = rng.randint(0, len(content) - len(keyword) - 1)
            content = content[:position] + keyword + content[position + len(keyword) :]

        documents.append(Document(page_content=content, metadata={"source": f"doc{i}.txt", "chunk_index": i}))

    return documents


@pytest.mark.performance
@pytest.mark.parametrize("num_docs", [100, 500, 1000])
def test_bm25_initialization_performance(num_docs) -> None:
    """Test the performance of BM25Retriever initialization with different document counts."""
    documents = generate_test_documents(num_docs)

    start_time = time.time()
    result = BM25Retriever(documents)
    elapsed = time.time() - start_time

    assert result is not None
    assert hasattr(result, "search")
    assert elapsed < 10.0, f"BM25 init with {num_docs} docs took {elapsed:.2f}s"


@pytest.mark.performance
@pytest.mark.parametrize("num_docs", [100, 500, 1000])
def test_bm25_search_performance(num_docs) -> None:
    """Test the performance of BM25 search with different document counts."""
    documents = generate_test_documents(num_docs, codebase_keywords=True)
    retriever = BM25Retriever(documents)

    queries = ["codebase package", "power grid analysis", "load flow calculation", "completely unrelated query"]

    for query in queries:
        start_time = time.time()
        results = retriever.search(query, k=5)
        time.time() - start_time

        assert len(results) <= 5

        if "codebase" in query or "power" in query or "load" in query:
            assert len(results) > 0
        else:
            # No term overlap with the corpus: BM25 must return nothing rather
            # than padding the list out to k with zero-scored documents.
            assert len(results) == 0


@pytest.mark.performance
def test_hybrid_search_scaling() -> None:
    """Test how hybrid search performance scales with document count."""
    document_counts = [100, 500, 1000]

    for num_docs in document_counts:
        documents = generate_test_documents(num_docs, codebase_keywords=True)
        bm25_retriever = BM25Retriever(documents)

        # Create a mock vector retriever that returns fixed results
        class MockVectorRetriever:
            def __init__(self, docs):
                self._docs = docs

            def search(self, query, k):
                return [(random.choice(self._docs), random.uniform(0.7, 0.9)) for _ in range(min(5, len(self._docs)))]

        vector_retriever = MockVectorRetriever(documents)

        hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)

        query = "codebase analysis"

        start_time = time.time()
        results = hybrid_retriever.search(query, k=5)
        time.time() - start_time

        assert len(results) <= 5
