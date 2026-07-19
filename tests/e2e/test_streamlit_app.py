"""End-to-end tests for the Streamlit application.

These run the real `app/main.py` script headlessly through
``streamlit.testing.v1.AppTest``, with only the external services (Qdrant,
Ollama, the RAG chain) mocked out. Unlike calling individual functions
directly, this exercises the session-state and rerun state machine that
Streamlit actually drives: a stuck "thinking" flag, a banner that vanishes
before a fragment tick finishes, chat state left inconsistent by a partial
reset. Bugs at that layer don't show up in a plain function call test.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import codebase_rag.app.components as comp

APP_PATH = str(Path(__file__).parent.parent.parent / "src" / "codebase_rag" / "app" / "main.py")


@pytest.fixture
def mocked_rag_chain():
    """Patch Qdrant, Ollama, and the RAG chain so `main.py` runs offline.

    `initialize_app_components` is cached with `st.cache_resource`, so the
    cache is cleared before and after each test to stop one test's mocks
    leaking into the next.
    """
    st.cache_resource.clear()

    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True

    mock_llm = MagicMock()
    mock_llm.check_connection.return_value = {"status": "connected", "message": "ok"}
    mock_llm.check_model_availability.return_value = {"status": "available", "message": "ok"}

    mock_rag_chain = MagicMock()
    mock_rag_chain.stream.return_value = iter(["Hello", " world"])
    mock_rag_chain.last_result = {"answer": "Hello world", "sources": []}

    with (
        patch("codebase_rag.database.qdrant_store.QdrantStore", return_value=mock_qdrant),
        patch("codebase_rag.llm.ollama_client.OllamaClient", return_value=mock_llm),
        patch("codebase_rag.llm.rag_chain.RAGChain", return_value=mock_rag_chain),
    ):
        yield mock_rag_chain

    st.cache_resource.clear()


@pytest.fixture(autouse=True)
def _reset_ingestion_status():
    """Ingestion status and the folder-dialog result are module-level
    state shared across the process; make sure one test's leftovers
    can't leak into the next."""
    original = dict(comp._ingestion_status)
    comp._ingestion_status.clear()
    with comp._folder_dialog_lock:
        comp._folder_dialog_result.clear()
    comp._folder_dialog_thread = None
    yield
    comp._ingestion_status.clear()
    comp._ingestion_status.update(original)
    with comp._folder_dialog_lock:
        comp._folder_dialog_result.clear()
    comp._folder_dialog_thread = None


@pytest.mark.e2e
def test_app_initializes_and_shows_chat_input(mocked_rag_chain: MagicMock) -> None:
    """A healthy backend should initialize successfully and unlock chat input."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert at.session_state["initialized"] is True
    assert len(at.chat_input) == 1
    assert not at.chat_input[0].disabled


@pytest.mark.e2e
def test_submitting_a_question_streams_answer_and_resets_thinking(mocked_rag_chain: MagicMock) -> None:
    """Submitting a question should stream the answer and leave the
    thinking/query_to_process flags clean, ready for the next question."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.chat_input[0].set_value("How do I use this codebase?").run()

    mocked_rag_chain.stream.assert_called_once_with("How do I use this codebase?")
    assert at.session_state["thinking"] is False
    assert at.session_state["query_to_process"] is None
    assert not at.exception
    assert any("Hello world" in msg["content"] for msg in at.session_state["messages"])


@pytest.mark.e2e
def test_stream_error_resets_thinking_instead_of_looping(mocked_rag_chain: MagicMock) -> None:
    """Regression test for FE-1: a failure while streaming the answer (e.g.
    Ollama becoming unreachable mid-query) must not leave `thinking=True`
    with the query still queued, which previously caused the same failing
    query to re-run forever on every subsequent rerun."""
    mocked_rag_chain.stream.side_effect = RuntimeError("backend unreachable")

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.chat_input[0].set_value("this will fail").run()

    assert at.session_state["thinking"] is False
    assert at.session_state["query_to_process"] is None
    assert any("encountered an error" in msg["content"] for msg in at.session_state["messages"])

    # A further, unrelated rerun must not re-attempt the failed query.
    mocked_rag_chain.stream.reset_mock(side_effect=True)
    at.run()
    mocked_rag_chain.stream.assert_not_called()


@pytest.mark.e2e
def test_ingestion_error_banner_persists_until_dismissed(mocked_rag_chain: MagicMock) -> None:
    """Regression test for FE-5: an ingestion failure banner used to be
    wiped from state the instant it was shown, so it vanished on the very
    next 5-second sidebar fragment tick. It should now survive repeated
    reruns until the user explicitly dismisses it."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    comp._ingestion_status.update(running=False, error="disk full")
    at.run()  # simulates the sidebar fragment's first auto-refresh tick

    def sidebar_text() -> str:
        return " ".join(e.value for e in at.sidebar.error) + " ".join(e.value for e in at.sidebar.success)

    assert "disk full" in sidebar_text()

    at.run()  # a second tick, the old code cleared state on the first tick
    assert "disk full" in sidebar_text()

    dismiss_buttons = [b for b in at.sidebar.button if b.key == "btn_dismiss_ingestion_outcome"]
    assert dismiss_buttons, "expected a dismiss button once the outcome banner is showing"
    dismiss_buttons[0].click().run()

    assert "disk full" not in sidebar_text()


@pytest.mark.e2e
def test_invalid_github_url_error_persists_until_dismissed(mocked_rag_chain: MagicMock) -> None:
    """The "Please enter a valid GitHub URL" validation message should also
    survive a fragment tick instead of disappearing within seconds."""
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

    at.run()  # another tick — the message must still be there
    assert "valid GitHub URL" in sidebar_text()


def _click_browse(at: AppTest) -> None:
    browse_buttons = [b for b in at.sidebar.button if b.key == "btn_browse_folder"]
    assert browse_buttons, "expected a Browse… button in the Local Folder tab"
    browse_buttons[0].click().run()
    # The dialog runs in a background thread; wait for it to deliver its
    # result before simulating the next fragment tick.
    assert comp._folder_dialog_thread is not None
    comp._folder_dialog_thread.join(timeout=5)


@pytest.mark.e2e
def test_browse_button_shows_picked_folder(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """The full Browse loop: click → dialog thread → result dict → poll on
    the next fragment tick → selected path rendered with an Ingest button."""
    (tmp_path / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    with patch.object(comp, "_pick_folder_path", return_value=(str(tmp_path), None)):
        _click_browse(at)

    at.run()  # next fragment tick picks the result out of the shared dict

    assert at.session_state["selected_folder"] == str(tmp_path)
    sidebar_md = " ".join(m.value for m in at.sidebar.markdown)
    assert str(tmp_path) in sidebar_md
    assert [b for b in at.sidebar.button if b.key == "btn_ingest_local"]


@pytest.mark.e2e
def test_browse_cancel_shows_nothing(mocked_rag_chain: MagicMock) -> None:
    """Cancelling the dialog should leave the tab unchanged — no path, no
    error banner."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    with patch.object(comp, "_pick_folder_path", return_value=(None, None)):
        _click_browse(at)

    at.run()

    assert at.session_state["selected_folder"] == ""
    assert not [b for b in at.sidebar.button if b.key == "btn_ingest_local"]


@pytest.mark.e2e
def test_browse_failure_surfaces_error(mocked_rag_chain: MagicMock) -> None:
    """A real dialog failure (e.g. missing Automation permission) must be
    shown in the sidebar instead of only being logged on the server."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    with patch.object(comp, "_pick_folder_path", return_value=(None, "Folder dialog failed: boom")):
        _click_browse(at)

    at.run()

    sidebar_errors = " ".join(e.value for e in at.sidebar.error)
    assert "boom" in sidebar_errors
