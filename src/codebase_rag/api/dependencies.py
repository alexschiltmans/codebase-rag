"""FastAPI dependency accessors for the shared `ApiState`."""

from fastapi import Request

from codebase_rag.api.state import ApiState


def get_state(request: Request) -> ApiState:
    return request.app.state.api_state  # type: ignore[no-any-return]
