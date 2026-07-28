"""Module for processing documents from repositories."""

import logging
import threading
from collections.abc import Callable

from langchain_core.documents import Document

from .chunking import DocumentChunker
from .git_loader import GitLoader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents from repositories.

    This class orchestrates the document processing pipeline, including
    loading files from the Git repository, chunking them appropriately,
    and preparing them for indexing.
    """

    def __init__(
        self,
        git_loader: GitLoader | None = None,
        document_chunker: DocumentChunker | None = None,
    ) -> None:
        """Initialize the DocumentProcessor.

        Args:
            git_loader: Optional GitLoader instance.
            document_chunker: Optional DocumentChunker instance.
        """
        self.git_loader = git_loader or GitLoader()
        self.document_chunker = document_chunker or DocumentChunker()

    def process(
        self,
        included_dirs: list[str] | None = None,
        included_files: list[str] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Document]:
        """Process all relevant files from the repository.

        Args:
            included_dirs: List of directory paths to include.
            included_files: List of specific files to include.
            progress_callback: Optional hook called as ``("processing", current, total)``
                after each file is processed.
            cancel_event: Optional event checked before each file; raises
                ``IngestCancelled`` when set.

        Returns:
            List[Document]: Processed and chunked documents ready for indexing.
        """
        self.git_loader.clone_or_pull()

        file_paths = self.git_loader.get_file_paths(included_dirs, included_files)
        total = len(file_paths)

        all_documents = []
        for i, file_path in enumerate(file_paths, start=1):
            if cancel_event is not None and cancel_event.is_set():
                from codebase_rag.data_ingestion.pipeline import IngestCancelled

                raise IngestCancelled("Ingestion cancelled during file processing")

            logger.info("Processing %s", file_path)
            documents = self.document_chunker.process_file(file_path)
            all_documents.extend(documents)
            logger.debug("Added %d chunks from %s", len(documents), file_path)

            if progress_callback is not None:
                progress_callback("processing", i, total)

        logger.info("Processed %d files into %d chunks", len(file_paths), len(all_documents))
        return all_documents
