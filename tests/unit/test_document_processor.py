"""Unit tests for data_ingestion/document_processor.py."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from codebase_rag.data_ingestion.document_processor import DocumentProcessor
from codebase_rag.data_ingestion.pipeline import IngestCancelled


class TestDocumentProcessor:
    """Tests for DocumentProcessor.process."""

    def test_uses_provided_collaborators(self) -> None:
        mock_git_loader = MagicMock()
        mock_chunker = MagicMock()

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)

        assert processor.git_loader is mock_git_loader
        assert processor.document_chunker is mock_chunker

    def test_process_clones_and_chunks_each_file(self) -> None:
        mock_git_loader = MagicMock()
        mock_git_loader.get_file_paths.return_value = [Path("a.py"), Path("b.md")]

        mock_chunker = MagicMock()
        mock_chunker.process_file.side_effect = [
            [Document(page_content="a", metadata={})],
            [Document(page_content="b1", metadata={}), Document(page_content="b2", metadata={})],
        ]

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)
        documents = processor.process(included_dirs=["src"], included_files=["README.md"])

        mock_git_loader.clone_or_pull.assert_called_once()
        mock_git_loader.get_file_paths.assert_called_once_with(["src"], ["README.md"])
        assert mock_chunker.process_file.call_count == 2
        assert len(documents) == 3

    def test_process_returns_empty_list_when_no_files(self) -> None:
        mock_git_loader = MagicMock()
        mock_git_loader.get_file_paths.return_value = []
        mock_chunker = MagicMock()

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)
        documents = processor.process()

        assert documents == []
        mock_chunker.process_file.assert_not_called()

    def test_process_reports_monotonic_progress(self) -> None:
        mock_git_loader = MagicMock()
        mock_git_loader.get_file_paths.return_value = [Path("a.py"), Path("b.py"), Path("c.py")]
        mock_chunker = MagicMock()
        mock_chunker.process_file.return_value = [Document(page_content="x", metadata={})]

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)
        calls: list[tuple[str, int, int]] = []
        processor.process(progress_callback=lambda phase, current, total: calls.append((phase, current, total)))

        assert calls == [("processing", 1, 3), ("processing", 2, 3), ("processing", 3, 3)]

    def test_process_raises_ingest_cancelled_when_event_set(self) -> None:
        mock_git_loader = MagicMock()
        mock_git_loader.get_file_paths.return_value = [Path("a.py"), Path("b.py")]
        mock_chunker = MagicMock()
        mock_chunker.process_file.return_value = [Document(page_content="x", metadata={})]

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)
        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(IngestCancelled):
            processor.process(cancel_event=cancel_event)

        mock_chunker.process_file.assert_not_called()

    def test_process_stops_at_next_boundary_once_cancelled_mid_loop(self) -> None:
        mock_git_loader = MagicMock()
        mock_git_loader.get_file_paths.return_value = [Path("a.py"), Path("b.py"), Path("c.py")]
        mock_chunker = MagicMock()
        mock_chunker.process_file.return_value = [Document(page_content="x", metadata={})]

        processor = DocumentProcessor(git_loader=mock_git_loader, document_chunker=mock_chunker)
        cancel_event = threading.Event()

        def _maybe_cancel(phase: str, current: int, total: int) -> None:
            if current == 1:
                cancel_event.set()

        with pytest.raises(IngestCancelled):
            processor.process(progress_callback=_maybe_cancel, cancel_event=cancel_event)

        assert mock_chunker.process_file.call_count == 1
