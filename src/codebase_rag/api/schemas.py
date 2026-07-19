"""Pydantic request/response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel

from codebase_rag.services.search_service import DEFAULT_TOKEN_BUDGET


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    repo: str | None = None
    token_budget: int = DEFAULT_TOKEN_BUDGET
    format: Literal["json", "compact"] = "json"


class SearchResultSchema(BaseModel):
    model_config = {"from_attributes": True}

    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    token_estimate: int


class SearchResponse(BaseModel):
    results: list[SearchResultSchema]


class AnswerRequest(BaseModel):
    question: str
    stream: bool = False


class CitationSchema(BaseModel):
    model_config = {"from_attributes": True}

    path: str
    start_line: int | None = None
    end_line: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[CitationSchema]


class RepoInfoSchema(BaseModel):
    model_config = {"from_attributes": True}

    name: str
    last_ingest_time: float | None
    head_sha: str | None


class IngestRequest(BaseModel):
    source: str


class IngestJobSchema(BaseModel):
    source: str
    state: Literal["running", "succeeded", "failed"]
    error: str | None = None
    result: dict | None = None
