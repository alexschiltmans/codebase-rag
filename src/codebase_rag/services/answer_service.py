"""Answer generation service wrapping `RAGChain`, shared by the HTTP API."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)


def _citations_from_documents(documents: list[Any]) -> list[Citation]:
    citations = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        citations.append(
            Citation(
                path=str(metadata.get("source", "unknown")),
                start_line=metadata.get("start_line"),
                end_line=metadata.get("end_line"),
            )
        )
    return citations


def answer(rag_chain: Any, question: str, **kwargs: Any) -> AnswerResult:
    """Run the full RAG chain and return the answer with source citations."""
    result = rag_chain.run(question, **kwargs)
    return AnswerResult(answer=result["answer"], citations=_citations_from_documents(result.get("documents", [])))


def stream_answer(rag_chain: Any, question: str, **kwargs: Any) -> Iterator[str]:
    """Stream the answer's text chunks. Citations are available afterward via
    `last_citations(rag_chain)`, mirroring `RAGChain.stream()`'s `last_result` contract.
    """
    yield from rag_chain.stream(question, **kwargs)


def last_citations(rag_chain: Any) -> list[Citation]:
    """Citations for the most recently streamed answer, once the generator is fully consumed."""
    result = rag_chain.last_result
    if not result:
        return []
    return _citations_from_documents(result.get("documents", []))
