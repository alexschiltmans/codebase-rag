"""`/answer` endpoint: full RAG answer, or an SSE token stream when `stream=true`."""

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from codebase_rag.api.dependencies import get_state
from codebase_rag.api.schemas import AnswerRequest, AnswerResponse
from codebase_rag.api.state import ApiState
from codebase_rag.services.answer_service import answer, last_citations, stream_answer

router = APIRouter()


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(state: ApiState, question: str) -> Iterator[str]:
    rag_chain = state.new_rag_chain()
    for chunk in stream_answer(rag_chain, question):
        yield _sse_event("token", {"text": chunk})

    citations = [
        {"path": c.path, "start_line": c.start_line, "end_line": c.end_line} for c in last_citations(rag_chain)
    ]
    yield _sse_event("done", {"sources": citations})


@router.post("/answer", response_model=None)
def answer_endpoint(request: AnswerRequest, state: ApiState = Depends(get_state)) -> AnswerResponse | StreamingResponse:
    if request.stream:
        return StreamingResponse(_stream_events(state, request.question), media_type="text/event-stream")

    rag_chain = state.new_rag_chain()
    result = answer(rag_chain, request.question)
    return AnswerResponse.model_validate({"answer": result.answer, "sources": result.citations})
