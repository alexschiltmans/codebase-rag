"""API tests with FastAPI's TestClient. The retriever, LLM, and Qdrant store
are all mocked — these tests exercise routing, request/response shapes, and
job-lock behavior, not the retrieval stack itself.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from codebase_rag.api.ingest_manager import ApiIngestionManager, IngestJob
from codebase_rag.api.routers import answer, ingest, repos, search


class FakeApiState:
    def __init__(self, tmp_path: Path) -> None:
        self.hybrid_retriever = MagicMock()
        self.tokenizer = None
        self.qdrant_store = MagicMock()
        self.qdrant_store.list_repos.return_value = []
        self.cache_dir = tmp_path
        self.ingestion = ApiIngestionManager()
        self._rag_chain = MagicMock()

    def new_rag_chain(self) -> MagicMock:
        return self._rag_chain

    def refresh_bm25(self) -> None:
        pass


@pytest.fixture
def state(tmp_path: Path) -> FakeApiState:
    return FakeApiState(tmp_path)


@pytest.fixture
def client(state: FakeApiState) -> TestClient:
    app = FastAPI()
    app.state.api_state = state
    app.include_router(search.router)
    app.include_router(answer.router)
    app.include_router(repos.router)
    app.include_router(ingest.router)
    return TestClient(app)


def _doc(source: str, start: int, end: int, content: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "start_line": start, "end_line": end})


class TestSearchEndpoint:
    def test_json_format(self, client: TestClient, state: FakeApiState) -> None:
        state.hybrid_retriever.search.return_value = [(_doc("a.py", 1, 5, "hello world"), 0.9)]

        response = client.post("/search", json={"query": "hello"})

        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["path"] == "a.py"
        assert body["results"][0]["start_line"] == 1

    def test_compact_format_returns_plain_text(self, client: TestClient, state: FakeApiState) -> None:
        state.hybrid_retriever.search.return_value = [(_doc("a.py", 1, 5, "hello world"), 0.9)]

        response = client.post("/search", json={"query": "hello", "format": "compact"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "{" not in response.text

    def test_empty_results(self, client: TestClient, state: FakeApiState) -> None:
        state.hybrid_retriever.search.return_value = []

        response = client.post("/search", json={"query": "nothing"})

        assert response.json() == {"results": []}


class TestAnswerEndpoint:
    def test_non_streaming_answer(self, client: TestClient, state: FakeApiState) -> None:
        state._rag_chain.run.return_value = {
            "answer": "The ingestion pipeline chunks and embeds files.",
            "documents": [_doc("pipeline.py", 1, 10, "...")],
        }

        response = client.post("/answer", json={"question": "explain the ingestion pipeline"})

        assert response.status_code == 200
        body = response.json()
        assert "ingestion pipeline" in body["answer"]
        assert body["sources"][0]["path"] == "pipeline.py"

    def test_streaming_answer_returns_sse(self, client: TestClient, state: FakeApiState) -> None:
        state._rag_chain.stream.return_value = iter(["The ", "answer."])
        state._rag_chain.last_result = {"documents": [_doc("pipeline.py", 1, 10, "...")]}

        response = client.post("/answer", json={"question": "explain it", "stream": True})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: token" in response.text
        assert "event: done" in response.text


class TestReposEndpoint:
    def test_list_repos(self, client: TestClient, state: FakeApiState) -> None:
        state.qdrant_store.list_repos.return_value = ["my-repo"]

        response = client.get("/repos")

        assert response.status_code == 200
        body = response.json()
        assert body[0]["name"] == "my-repo"

    def test_freshness_metadata_is_read_from_cache_dir(self, client: TestClient, state: FakeApiState) -> None:
        state.qdrant_store.list_repos.return_value = ["my-repo"]
        (state.cache_dir / "my-repo_freshness.json").write_text(
            json.dumps({"last_ingest_time": 123.0, "head_sha": "abc123"})
        )

        response = client.get("/repos")

        body = response.json()
        assert body[0]["last_ingest_time"] == 123.0
        assert body[0]["head_sha"] == "abc123"


class TestIngestEndpoint:
    def test_ingest_returns_202_and_running_job(
        self, client: TestClient, state: FakeApiState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "codebase_rag.api.ingest_manager.threading.Thread",
            lambda target, daemon: MagicMock(start=lambda: None),
        )

        response = client.post("/ingest", json={"source": "/some/local/path"})

        assert response.status_code == 202
        assert response.json()["state"] == "running"

    def test_concurrent_ingest_returns_409(
        self, client: TestClient, state: FakeApiState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "codebase_rag.api.ingest_manager.threading.Thread",
            lambda target, daemon: MagicMock(start=lambda: None),
        )

        client.post("/ingest", json={"source": "/some/local/path"})
        response = client.post("/ingest", json={"source": "/another/path"})

        assert response.status_code == 409


def _wait_for_completion(job: IngestJob, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)


class TestApiIngestionManager:
    def test_success_populates_result_and_fires_callback_once(self) -> None:
        callback_states = []
        manager = ApiIngestionManager(on_success=lambda job: callback_states.append(job.state))

        fake_result = MagicMock(
            repo_name="my-repo", files_changed=1, files_deleted=0, files_unchanged=2, chunks_indexed=3, head_sha=None
        )
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline") as pipeline_cls:
            pipeline_cls.return_value.process_repo_incremental.return_value = fake_result
            job = manager.start("/some/path")
            assert job is not None
            _wait_for_completion(job)

        assert job.state == "succeeded"
        assert job.result is not None
        assert job.result["repo_name"] == "my-repo"
        assert callback_states == ["succeeded"]

    def test_failure_records_error_and_skips_callback(self) -> None:
        callback_states = []
        manager = ApiIngestionManager(on_success=lambda job: callback_states.append(job.state))

        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline") as pipeline_cls:
            pipeline_cls.return_value.process_repo_incremental.side_effect = RuntimeError("qdrant down")
            job = manager.start("/some/path")
            assert job is not None
            _wait_for_completion(job)

        assert job.state == "failed"
        assert job.error == "qdrant down"
        assert callback_states == []

    def test_slot_is_free_again_after_completion(self) -> None:
        manager = ApiIngestionManager()
        fake_result = MagicMock(
            repo_name="r", files_changed=0, files_deleted=0, files_unchanged=0, chunks_indexed=0, head_sha=None
        )
        with patch("codebase_rag.data_ingestion.pipeline.IngestPipeline") as pipeline_cls:
            pipeline_cls.return_value.process_repo_incremental.return_value = fake_result
            first = manager.start("/one")
            assert first is not None
            _wait_for_completion(first)
            second = manager.start("/two")

        assert second is not None
