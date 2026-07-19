from fastapi import APIRouter, Depends

from codebase_rag.api.dependencies import get_state
from codebase_rag.api.schemas import RepoInfoSchema
from codebase_rag.api.state import ApiState
from codebase_rag.services.repo_service import list_repos

router = APIRouter()


@router.get("/repos")
def repos_endpoint(state: ApiState = Depends(get_state)) -> list[RepoInfoSchema]:
    infos = list_repos(state.qdrant_store, state.cache_dir)
    return [RepoInfoSchema.model_validate(info) for info in infos]
