from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from codebase_rag.api.dependencies import get_state
from codebase_rag.api.schemas import SearchRequest, SearchResponse
from codebase_rag.api.state import ApiState
from codebase_rag.services.compact_format import format_compact
from codebase_rag.services.search_service import search

router = APIRouter()


@router.post("/search", response_model=None)
def search_endpoint(request: SearchRequest, state: ApiState = Depends(get_state)) -> SearchResponse | PlainTextResponse:
    results = search(
        state.hybrid_retriever,
        query=request.query,
        k=request.k,
        repo=request.repo,
        token_budget=request.token_budget,
        tokenizer=state.tokenizer,
    )

    if request.format == "compact":
        return PlainTextResponse(format_compact(results))

    return SearchResponse.model_validate({"results": results})
