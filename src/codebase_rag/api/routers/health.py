from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_endpoint() -> dict[str, str]:
    """Report that this process is serving, independent of the vector store or LLM backend."""
    return {"status": "ok"}
