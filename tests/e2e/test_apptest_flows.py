"""End-to-end tests for the Streamlit application.

These run the real `app/main.py` script headlessly through
``streamlit.testing.v1.AppTest``, with only the external services (Qdrant,
Ollama, the RAG chain) mocked out. Unlike calling individual functions
directly, this exercises the session-state and rerun state machine that
Streamlit actually drives. Bugs at that layer don't show up in a plain
function call test.
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from codebase_rag.app.runtime import get_repo_list

APP_PATH = str(Path(__file__).parent.parent.parent / "src" / "codebase_rag" / "app" / "main.py")


@pytest.fixture
def mocked_rag_chain():
    """Patch Qdrant, Ollama, and the RAG chain so `main.py` runs offline.

    `get_runtime` is cached with `st.cache_resource`, and `get_repo_list`
    with `st.cache_data`; both are cleared before and after each test so
    one test's mocks can't leak into the next.
    """
    st.cache_resource.clear()
    get_repo_list.clear()

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_qdrant.list_repos.return_value = []

    mock_llm = MagicMock()
    mock_llm.check_connection.return_value = {"status": "connected", "message": "ok"}
    mock_llm.check_model_availability.return_value = {"status": "available", "message": "ok"}

    mock_rag_chain = MagicMock()
    mock_rag_chain.stream.return_value = iter(["Hello", " world"])
    mock_rag_chain.last_result = {"answer": "Hello world", "sources": []}

    with (
        patch("codebase_rag.app.runtime.QdrantStore", return_value=mock_qdrant),
        patch("codebase_rag.app.runtime.OllamaClient", return_value=mock_llm),
        patch("codebase_rag.app.runtime.RAGChain", return_value=mock_rag_chain),
        patch("codebase_rag.app.runtime._load_or_create_bm25_retriever", return_value=MagicMock()),
    ):
        yield mock_rag_chain

    st.cache_resource.clear()
    get_repo_list.clear()


@pytest.mark.e2e
def test_app_initializes_and_shows_chat_input(mocked_rag_chain: MagicMock) -> None:
    """A healthy backend should initialize successfully and unlock chat input."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert len(at.chat_input) == 1
    assert not at.chat_input[0].disabled


@pytest.mark.e2e
def test_submitting_a_question_streams_answer_and_returns_to_idle(mocked_rag_chain: MagicMock) -> None:
    """Submitting a question should stream the answer and return the query
    lifecycle to IDLE, ready for the next question."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.chat_input[0].set_value("How do I use this codebase?").run()

    mocked_rag_chain.stream.assert_called_once_with("How do I use this codebase?")
    assert at.session_state["query_state"] == "idle"
    assert at.session_state["pending_query"] is None
    assert not at.exception
    assert any("Hello world" in msg["content"] for msg in at.session_state["messages"])


@pytest.mark.e2e
def test_stream_error_shows_failed_card_instead_of_looping(mocked_rag_chain: MagicMock) -> None:
    """Regression test for FE-1: a failure while streaming the answer (e.g.
    Ollama becoming unreachable mid-query) must land in the FAILED state
    with a Retry/Dismiss card, not re-run the same query forever."""
    mocked_rag_chain.stream.side_effect = RuntimeError("backend unreachable")

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("this will fail").run()

    assert at.session_state["query_state"] == "failed"
    assert "backend unreachable" in at.session_state["query_error"]

    # A further, unrelated rerun must not re-attempt the failed query.
    mocked_rag_chain.stream.reset_mock(side_effect=True)
    at.run()
    mocked_rag_chain.stream.assert_not_called()


@pytest.mark.e2e
def test_retry_resubmits_the_failed_query_once(mocked_rag_chain: MagicMock) -> None:
    mocked_rag_chain.stream.side_effect = RuntimeError("backend unreachable")
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("this will fail").run()
    assert at.session_state["query_state"] == "failed"

    mocked_rag_chain.stream.side_effect = None
    mocked_rag_chain.stream.return_value = iter(["Hello", " world"])
    retry_buttons = [b for b in at.button if b.key == "btn_retry_query"]
    assert retry_buttons
    retry_buttons[0].click().run()

    mocked_rag_chain.stream.assert_called_with("this will fail")
    assert at.session_state["query_state"] == "idle"


@pytest.mark.e2e
def test_dismiss_failed_query_returns_to_idle_without_resubmitting(mocked_rag_chain: MagicMock) -> None:
    mocked_rag_chain.stream.side_effect = RuntimeError("backend unreachable")
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("this will fail").run()

    mocked_rag_chain.stream.reset_mock(side_effect=True)
    dismiss_buttons = [b for b in at.button if b.key == "btn_dismiss_query_error"]
    assert dismiss_buttons
    dismiss_buttons[0].click().run()

    assert at.session_state["query_state"] == "idle"
    mocked_rag_chain.stream.assert_not_called()


@pytest.mark.e2e
def test_new_chat_after_failure_clears_error_and_blocks_stale_retry(mocked_rag_chain: MagicMock) -> None:
    """Regression test for UI-3: a failed query's error state must not
    survive navigating to a new chat, and the new chat must not receive
    an orphan assistant answer from the old chat's query."""
    mocked_rag_chain.stream.side_effect = RuntimeError("backend unreachable")
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("this will fail").run()
    assert at.session_state["query_state"] == "failed"
    original_chat_id = at.session_state["current_chat_id"]
    original_chat_length_after_failure = len(at.session_state["chat_histories"][original_chat_id])

    new_chat_buttons = [b for b in at.sidebar.button if b.label == "Start New Chat"]
    assert new_chat_buttons
    new_chat_buttons[0].click().run()

    assert not [b for b in at.button if b.key == "btn_retry_query"]
    assert at.session_state["query_state"] == "idle"
    assert at.session_state["pending_query"] is None
    assert at.session_state["query_error"] is None

    unique_question = f"a normal question {uuid.uuid4()}"
    mocked_rag_chain.stream.side_effect = None
    mocked_rag_chain.stream.return_value = iter(["Hello", " world"])
    at.chat_input[0].set_value(unique_question).run()

    # The failed query never resubmitted anywhere: the original chat gained
    # no new messages (no orphan assistant answer landed there), and the new
    # chat contains the exchange for the question just asked there instead.
    original_chat_history = at.session_state["chat_histories"][original_chat_id]
    assert len(original_chat_history) == original_chat_length_after_failure
    assert not any(m["content"] == unique_question for m in original_chat_history)

    current_chat_id = at.session_state["current_chat_id"]
    assert current_chat_id != original_chat_id
    new_chat_history = at.session_state["chat_histories"][current_chat_id]
    assert not any(m["content"] == "this will fail" for m in new_chat_history)
    assert any(m["content"] == unique_question for m in new_chat_history)
    for i, message in enumerate(new_chat_history):
        if message["role"] == "assistant":
            assert i > 0
            assert new_chat_history[i - 1]["role"] == "user"


@pytest.mark.e2e
def test_invalid_github_url_error_persists_until_dismissed(mocked_rag_chain: MagicMock) -> None:
    """The "Please enter a valid GitHub URL" validation message should
    survive a rerun instead of disappearing within seconds."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "new_repo_url"]
    assert text_inputs
    text_inputs[0].set_value("not-a-github-url").run()

    ingest_buttons = [b for b in at.sidebar.button if b.key == "btn_ingest_repo"]
    assert ingest_buttons
    ingest_buttons[0].click().run()

    def sidebar_text() -> str:
        return " ".join(e.value for e in at.sidebar.error)

    assert "valid GitHub URL" in sidebar_text()

    at.run()  # another rerun — the message must still be there
    assert "valid GitHub URL" in sidebar_text()


@pytest.mark.e2e
def test_local_folder_nonexistent_path_shows_validation_error(mocked_rag_chain: MagicMock) -> None:
    """Regression test for §6.4: submitting a nonexistent local path must
    show an inline validation error and never start a job."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    assert text_inputs
    text_inputs[0].set_value("/definitely/does/not/exist").run()

    assert at.session_state["selected_folder"] == "/definitely/does/not/exist"
    assert not [b for b in at.sidebar.button if b.key == "btn_ingest_local"]
    errors = " ".join(e.value for e in at.sidebar.error)
    assert "does not exist" in errors


@pytest.mark.e2e
def test_local_folder_valid_path_shows_file_count_and_ingest_button(
    mocked_rag_chain: MagicMock, tmp_path: Path
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "1 file(s)" in captions
    assert [b for b in at.sidebar.button if b.key == "btn_ingest_local"]
