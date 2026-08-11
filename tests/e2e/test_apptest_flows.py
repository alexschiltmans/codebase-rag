"""End-to-end tests for the Streamlit application.

These run the real `app/main.py` script headlessly through
``streamlit.testing.v1.AppTest``, with only the external services (Qdrant,
Ollama, the RAG chain) mocked out. Unlike calling individual functions
directly, this exercises the session-state and rerun state machine that
Streamlit actually drives. Bugs at that layer don't show up in a plain
function call test.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from codebase_rag.app.runtime import get_repo_list

APP_PATH = str(Path(__file__).parent.parent.parent / "src" / "codebase_rag" / "app" / "main.py")


@pytest.fixture
def mocked_rag_chain() -> Iterator[MagicMock]:
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
        patch("codebase_rag.app.runtime.create_llm_client", return_value=mock_llm),
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
    """A failed query's error state must not
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

    at.run()  # another rerun; the message must still be there
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


@pytest.mark.e2e
def test_refused_github_ingest_shows_banner_and_keeps_url(mocked_rag_chain: MagicMock) -> None:
    """A refused start() must not be discarded:
    the URL input must survive and an error banner must appear."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "new_repo_url"]
    text_inputs[0].set_value("https://github.com/owner/repo").run()

    with patch("codebase_rag.app.runtime.IngestionManager.start", return_value=False):
        ingest_buttons = [b for b in at.sidebar.button if b.key == "btn_ingest_repo"]
        ingest_buttons[0].click().run()

    assert not at.exception
    errors = " ".join(e.value for e in at.sidebar.error)
    assert "already running" in errors
    text_inputs = [ti for ti in at.sidebar.text_input if ti.key == "new_repo_url"]
    assert text_inputs[0].value == "https://github.com/owner/repo"


@pytest.mark.e2e
def test_refused_local_ingest_shows_banner_and_keeps_selection(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """A refused local-folder start() must not
    clear the folder selection or preview."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    with patch("codebase_rag.app.runtime.IngestionManager.start", return_value=False):
        ingest_buttons = [b for b in at.sidebar.button if b.key == "btn_ingest_local"]
        ingest_buttons[0].click().run()

    assert not at.exception
    errors = " ".join(e.value for e in at.sidebar.error)
    assert "already running" in errors
    assert at.session_state["selected_folder"] == str(tmp_path)
    assert [b for b in at.sidebar.button if b.key == "btn_ingest_local"]


@pytest.mark.e2e
def test_same_local_path_resubmittable_after_ingest(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """The edge-triggered path sync used to make
    the form one-shot: the same path couldn't be re-submitted after an
    ingest cleared the selection."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    with patch("codebase_rag.app.runtime.IngestionManager.start", return_value=True):
        ingest_buttons = [b for b in at.sidebar.button if b.key == "btn_ingest_local"]
        ingest_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["selected_folder"] == ""

    text_inputs = [ti for ti in at.sidebar.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    assert not at.exception
    assert at.session_state["selected_folder"] == str(tmp_path)
    assert [b for b in at.sidebar.button if b.key == "btn_ingest_local"]


@pytest.mark.e2e
def test_clearing_typed_path_removes_preview(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """Clearing the typed path must clear the
    selection, preview, and Ingest button, not leave the stale state on
    screen."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()
    assert [b for b in at.sidebar.button if b.key == "btn_ingest_local"]

    text_inputs = [ti for ti in at.sidebar.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value("").run()

    assert not at.exception
    assert at.session_state["selected_folder"] == ""
    assert not [b for b in at.sidebar.button if b.key == "btn_ingest_local"]


@pytest.mark.e2e
def test_cancel_button_signals_the_running_job(mocked_rag_chain: MagicMock) -> None:
    """Clicking Cancel while an ingest runs must call IngestionManager.cancel()."""
    from codebase_rag.app.runtime import IngestJob

    running_job = IngestJob(kind="manual", source="https://github.com/owner/repo")

    at = AppTest.from_file(APP_PATH)
    with (
        patch("codebase_rag.app.runtime.IngestionManager.current_job", return_value=running_job),
        patch("codebase_rag.app.runtime.IngestionManager.cancel") as mock_cancel,
    ):
        at.run()
        cancel_buttons = [b for b in at.sidebar.button if b.key == "btn_cancel_ingestion"]
        assert cancel_buttons
        cancel_buttons[0].click().run()

    assert not at.exception
    mock_cancel.assert_called_once()


@pytest.mark.e2e
def test_cancelled_ingest_shows_info_banner_and_ungates_chat(mocked_rag_chain: MagicMock) -> None:
    """A cancelled job must end with a dismissible info banner (not an
    error) and leave the chat surface ungated."""
    from codebase_rag.app.runtime import IngestJob

    cancelled_job = IngestJob(kind="manual", source="https://github.com/owner/repo", state="cancelled")

    at = AppTest.from_file(APP_PATH)
    with (
        patch("codebase_rag.app.runtime.IngestionManager.current_job", return_value=None),
        patch("codebase_rag.app.runtime.IngestionManager.last_completed", return_value=cancelled_job),
    ):
        at.run()

    assert not at.exception
    assert len(at.chat_input) == 1
    assert not at.chat_input[0].disabled

    infos = " ".join(i.value for i in at.sidebar.info)
    assert "cancelled" in infos
    assert "partially ingested" in infos

    dismiss_buttons = [b for b in at.sidebar.button if b.key == "btn_dismiss_ingestion_cancelled"]
    assert dismiss_buttons
    with patch("codebase_rag.app.runtime.IngestionManager.last_completed", return_value=None):
        dismiss_buttons[0].click().run()

    assert not at.exception
    assert not [i for i in at.sidebar.info if "cancelled" in i.value]


@pytest.mark.e2e
def test_preview_count_updates_after_adding_a_file(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """Regression test for the stale preview cache: adding a file to an
    already-selected folder must be reflected on the next rerun."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "1 file(s)" in captions

    (tmp_path / "src" / "second.py").write_text("print('bye')")
    at.run()

    assert not at.exception
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "2 file(s)" in captions


@pytest.mark.e2e
def test_picker_selection_wins_over_stale_typed_text(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """A folder just returned by the native picker must not be clobbered by
    leftover text still sitting in the typed-path box on the same render."""
    from codebase_rag.services.folder_picker import FolderPickResult

    typed_dir = tmp_path / "typed"
    typed_dir.mkdir()
    picked_dir = tmp_path / "picked"
    picked_dir.mkdir()

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(typed_dir)).run()
    assert at.session_state["selected_folder"] == str(typed_dir)

    at.session_state["folder_dialog_token"] = object()
    with patch(
        "codebase_rag.app.runtime.FolderPicker.poll",
        return_value=FolderPickResult(path=str(picked_dir)),
    ):
        at.run()

    assert not at.exception
    assert at.session_state["selected_folder"] == str(picked_dir)

    # The picked path must stay authoritative on the render after the pick too, not just
    # the one where it landed: the stale text still sitting in the box must not revert it.
    at.run()

    assert not at.exception
    assert at.session_state["selected_folder"] == str(picked_dir)


@pytest.mark.e2e
def test_browse_pick_through_wait_fragment_persists_and_clears_on_empty_input(
    mocked_rag_chain: MagicMock, tmp_path: Path
) -> None:
    """Drives the real Browse flow (button -> open() -> wait fragment ->
    poll() -> close) instead of injecting the token by hand, to catch bugs
    that only show up when the fragment itself consumes the picker result."""
    from codebase_rag.services.folder_picker import FolderPickResult

    picked_dir = tmp_path / "picked"
    picked_dir.mkdir()

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    browse_buttons = [b for b in expander.button if b.key == "btn_browse_folder"]
    assert browse_buttons

    picker_state = {"opened": False, "result_polled": False}

    def fake_open() -> object:
        picker_state["opened"] = True
        return object()

    def fake_is_open() -> bool:
        return picker_state["opened"] and not picker_state["result_polled"]

    def fake_poll(token: object) -> FolderPickResult:
        picker_state["result_polled"] = True
        return FolderPickResult(path=str(picked_dir))

    with (
        patch("codebase_rag.app.runtime.FolderPicker.open", side_effect=fake_open),
        patch("codebase_rag.app.runtime.FolderPicker.is_open", side_effect=fake_is_open),
        patch("codebase_rag.app.runtime.FolderPicker.poll", side_effect=fake_poll),
    ):
        browse_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["selected_folder"] == str(picked_dir)

    # A further, unrelated rerun must keep the picked folder selected.
    at.run()
    assert not at.exception
    assert at.session_state["selected_folder"] == str(picked_dir)

    # Clearing the text box afterward must clear the (now typed-editable) selection.
    text_inputs = [ti for ti in at.sidebar.text_input if ti.key == "typed_folder_path"]
    assert text_inputs[0].value == str(picked_dir)
    text_inputs[0].set_value("").run()

    assert not at.exception
    assert at.session_state["selected_folder"] == ""


@pytest.mark.e2e
def test_wait_fragment_still_renders_when_picker_thread_exits_between_poll_and_gate(
    mocked_rag_chain: MagicMock, tmp_path: Path
) -> None:
    """The picker thread can store its result and exit between the poll near
    the top of `_display_local_folder_tab` and the `is_open()` gate below it,
    so `is_open()` can already read False while the dialog token is still
    set. The wait fragment must still render in that window (gated on the
    token, not on `is_open()` alone), or a picked folder never appears until
    the user happens to touch an unrelated widget."""
    from codebase_rag.services.folder_picker import FolderPickResult

    picked_dir = tmp_path / "picked"
    picked_dir.mkdir()

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    browse_buttons = [b for b in expander.button if b.key == "btn_browse_folder"]
    assert browse_buttons

    poll_calls = {"n": 0}

    def fake_poll(token: object) -> FolderPickResult | None:
        # The first call, from the main body's poll, finds nothing yet: the thread hasn't
        # stored its result. `is_open()` (always False here) simulates the thread already
        # having exited by the time the gate checks it. The fragment's own poll call then
        # picks up the result that landed in between.
        poll_calls["n"] += 1
        if poll_calls["n"] == 1:
            return None
        return FolderPickResult(path=str(picked_dir))

    with (
        patch("codebase_rag.app.runtime.FolderPicker.open", return_value=object()),
        patch("codebase_rag.app.runtime.FolderPicker.is_open", return_value=False),
        patch("codebase_rag.app.runtime.FolderPicker.poll", side_effect=fake_poll),
    ):
        browse_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["selected_folder"] == str(picked_dir)


@pytest.mark.e2e
def test_stranded_dialog_token_is_cleared_instead_of_polled_forever(mocked_rag_chain: MagicMock) -> None:
    """A token with no result and a dialog that isn't open is unrecoverable
    (the picker thread is gone). `_poll_folder_dialog` must clear the token
    and surface an error rather than leaving the wait fragment gated open
    forever. Unlike the round-5 test, `poll()` never succeeds here; the
    token can never resolve on its own."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.session_state["folder_dialog_token"] = object()

    with (
        patch("codebase_rag.app.runtime.FolderPicker.is_open", return_value=False),
        patch("codebase_rag.app.runtime.FolderPicker.poll", return_value=None),
    ):
        at.run()

    assert not at.exception
    assert at.session_state["folder_dialog_token"] is None
    assert at.session_state["folder_dialog_error"]


@pytest.mark.e2e
def test_successful_local_ingest_does_not_crash_the_sidebar(mocked_rag_chain: MagicMock, tmp_path: Path) -> None:
    """A successful start() must clear the form without raising: assigning
    directly to the already-instantiated text-input widget's session-state
    key would otherwise crash the rerun."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")

    at = AppTest.from_file(APP_PATH)
    at.run()

    expander = next(e for e in at.sidebar.expander if e.label == "Add Repository")
    text_inputs = [ti for ti in expander.text_input if ti.key == "typed_folder_path"]
    text_inputs[0].set_value(str(tmp_path)).run()

    with patch("codebase_rag.app.runtime.IngestionManager.start", return_value=True):
        ingest_buttons = [b for b in at.sidebar.button if b.key == "btn_ingest_local"]
        ingest_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["selected_folder"] == ""

    at.run()

    assert not at.exception
    text_inputs = [ti for ti in at.sidebar.text_input if ti.key == "typed_folder_path"]
    assert text_inputs[0].value == ""
