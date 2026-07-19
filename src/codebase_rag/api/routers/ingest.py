from fastapi import APIRouter, Depends, HTTPException

from codebase_rag.api.dependencies import get_state
from codebase_rag.api.ingest_manager import IngestJob
from codebase_rag.api.schemas import IngestJobSchema, IngestRequest
from codebase_rag.api.state import ApiState

router = APIRouter()


def _job_schema(job: IngestJob) -> IngestJobSchema:
    return IngestJobSchema(source=job.source, state=job.state, error=job.error, result=job.result)


@router.post("/ingest", status_code=202)
def ingest_endpoint(request: IngestRequest, state: ApiState = Depends(get_state)) -> IngestJobSchema:
    job = state.ingestion.start(request.source)
    if job is None:
        current = state.ingestion.current_job()
        source = current.source if current else "unknown"
        raise HTTPException(status_code=409, detail=f"An ingest is already running for '{source}'")
    return _job_schema(job)


@router.get("/ingest/status")
def ingest_status_endpoint(state: ApiState = Depends(get_state)) -> IngestJobSchema | None:
    job = state.ingestion.current_job()
    if job is None:
        return None
    return _job_schema(job)
