"""FastAPI app factory for the retrieval HTTP API.

Runs as its own uvicorn process rather than mounted inside Streamlit, binding
to localhost by default. Exposing it beyond localhost requires adding
authentication first; there is none today.
"""

from fastapi import FastAPI

from codebase_rag.api.routers import answer, health, ingest, repos, search
from codebase_rag.api.state import ApiState
from codebase_rag.config import Config


def create_app(config: Config | None = None) -> FastAPI:
    app = FastAPI(title="Codebase RAG API")
    app.state.api_state = ApiState(config)

    app.include_router(search.router)
    app.include_router(answer.router)
    app.include_router(repos.router)
    app.include_router(ingest.router)
    app.include_router(health.router)

    return app
