"""API tests driven over a real async transport rather than FastAPI's TestClient.

TestClient runs the ASGI app through a background event-loop portal and blocks on each call, so
every request it makes is serialised no matter how the test is written: the existing
"concurrent ingest" test issues its two requests one after the other and would still pass if the
job slot had no lock at all. These tests hold the app in the test's own loop and issue requests
with `asyncio.gather`, which is the only way the single-slot guard and the streaming response are
exercised as the server actually runs them.

Handlers here are all sync `def`, which FastAPI dispatches to a threadpool. That is precisely why
concurrency is worth asserting: two requests really do run at the same time on different threads,
so the lock in ApiIngestionManager is load-bearing rather than decorative.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from langchain_core.documents import Document

from codebase_rag.api.ingest_manager import ApiIngestionManager
from codebase_rag.api.routers import answer, health, ingest, repos, search
from codebase_rag.api.state import ApiState


class FakeApiState:
    def __init__(self, tmp_path: Path) -> None:
        self.retriever = MagicMock()
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
def app(state: FakeApiState) -> FastAPI:
    application = FastAPI()
    application.state.api_state = state
    application.include_router(search.router)
    application.include_router(answer.router)
    application.include_router(repos.router)
    application.include_router(ingest.router)
    application.include_router(health.router)
    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client speaking ASGI directly, with no portal thread in between."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


def _doc(source: str, start: int, end: int, content: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "start_line": start, "end_line": end})


@pytest.mark.asyncio
class TestAsyncTransport:
    async def test_health_over_async_transport(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_search_serves_concurrent_requests(self, client: httpx.AsyncClient, state: FakeApiState) -> None:
        """Sync handlers dispatched to the threadpool must all complete, not serialise into a timeout."""
        state.retriever.search.return_value = [(_doc("a.py", 1, 2, "x"), 0.9)]

        responses = await asyncio.gather(*(client.post("/search", json={"query": f"q{i}", "k": 1}) for i in range(8)))

        assert [r.status_code for r in responses] == [200] * 8
        assert all(r.json()["results"] for r in responses)


@pytest.mark.asyncio
class TestConcurrentIngest:
    async def test_only_one_of_many_simultaneous_ingests_is_accepted(
        self, client: httpx.AsyncClient, state: FakeApiState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single job slot must hold when the requests genuinely race.

        The pipeline thread is stubbed out so the job stays in "running" for the whole test: what
        is under test is the claim on the slot, not what the pipeline does after claiming it.
        """
        monkeypatch.setattr(
            "codebase_rag.api.ingest_manager.threading.Thread",
            lambda target, daemon: MagicMock(start=lambda: None),
        )

        responses = await asyncio.gather(*(client.post("/ingest", json={"source": f"/repo/{i}"}) for i in range(10)))

        codes = sorted(r.status_code for r in responses)
        assert codes == [202] + [409] * 9, f"expected exactly one winner, got {codes}"

    async def test_status_reports_the_job_that_won_the_slot(
        self, client: httpx.AsyncClient, state: FakeApiState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "codebase_rag.api.ingest_manager.threading.Thread",
            lambda target, daemon: MagicMock(start=lambda: None),
        )

        responses = await asyncio.gather(*(client.post("/ingest", json={"source": f"/repo/{i}"}) for i in range(5)))
        accepted = next(r for r in responses if r.status_code == 202)

        status = await client.get("/ingest/status")

        assert status.status_code == 200
        assert status.json()["source"] == accepted.json()["source"]


@pytest.mark.asyncio
class TestStreamingAnswer:
    async def test_sse_framing_survives_the_async_transport(
        self, client: httpx.AsyncClient, state: FakeApiState
    ) -> None:
        """Every token becomes its own SSE event, and the stream closes with exactly one `done`.

        This checks the wire format the browser parses, over the same transport the server uses.
        It does not check that the events arrive separately in time: httpx's ASGITransport reads
        the response body to completion before handing it back, so any timing claim made here
        would be about the transport rather than the handler. `test_stream_events_is_lazy` covers
        that at the generator, where it can actually be observed.
        """
        state._rag_chain.stream.return_value = iter(["The ", "answer", "."])
        state._rag_chain.last_result = {"documents": [_doc("pipeline.py", 1, 10, "...")]}

        async with client.stream("POST", "/answer", json={"question": "explain it", "stream": True}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join([chunk async for chunk in response.aiter_text()])

        assert body.count("event: token") == 3
        assert body.count("event: done") == 1
        assert body.endswith("\n\n"), "SSE events must be terminated by a blank line"

    async def test_done_event_carries_citations(self, client: httpx.AsyncClient, state: FakeApiState) -> None:
        state._rag_chain.stream.return_value = iter(["ok"])
        state._rag_chain.last_result = {"documents": [_doc("pipeline.py", 4, 9, "...")]}

        async with client.stream("POST", "/answer", json={"question": "q", "stream": True}) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

        done_payload = body.split("event: done\ndata: ")[1].strip()
        sources = json.loads(done_payload)["sources"]

        assert sources == [{"path": "pipeline.py", "start_line": 4, "end_line": 9}]


def test_stream_events_is_lazy(state: FakeApiState) -> None:
    """The handler must emit each token as the chain produces it, not collect them all first.

    Asserted at the generator rather than over HTTP because no test client shows it: TestClient
    and httpx's ASGITransport both read the body to completion, so a handler that buffered every
    token and returned one blob would produce a byte-identical response. Pulling a single event
    and checking how much of the source was consumed is what separates the two.
    """
    produced: list[str] = []

    def _tracking_stream() -> Iterator[str]:
        for token in ["a", "b", "c"]:
            produced.append(token)
            yield token

    state._rag_chain.stream.return_value = _tracking_stream()
    state._rag_chain.last_result = {"documents": []}

    events = answer._stream_events(cast("ApiState", state), "q")
    first = next(events)

    assert first == 'event: token\ndata: {"text": "a"}\n\n'
    assert produced == ["a"], f"one event pulled but {len(produced)} tokens consumed, so it buffers"
