"""Text chunking strategies that preserve code structure and context."""

import hashlib
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingStrategy(StrEnum):
    """Enumeration of available chunking strategies."""

    CODE = "code"
    MARKDOWN = "markdown"
    DEFAULT = "default"


class DocumentChunker:
    """Class for chunking documents based on their type and content.

    This class implements various chunking strategies to preserve document structure
    and context, particularly for code and documentation files.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the DocumentChunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n", " ", ""],
            add_start_index=True,
        )

        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
                ("#####", "header_5"),
                ("######", "header_6"),
            ],
            strip_headers=False,
        )

        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    def _determine_strategy(self, file_path: Path) -> ChunkingStrategy:
        """Determine the appropriate chunking strategy based on the file type.

        Args:
            file_path: Path to the file being processed.

        Returns:
            ChunkingStrategy: The chunking strategy to use.
        """
        suffix = file_path.suffix.lower()

        if suffix in [".py", ".ipynb"]:
            return ChunkingStrategy.CODE
        if suffix in [".md", ".rst"]:
            return ChunkingStrategy.MARKDOWN
        return ChunkingStrategy.DEFAULT

    def chunk_document(
        self, content: str, metadata: dict[str, Any], strategy: ChunkingStrategy | None = None
    ) -> list[Document]:
        """Split a document into chunks while preserving context.

        Args:
            content: The document content to chunk.
            metadata: Metadata to attach to each chunk.
            strategy: Optional strategy override.

        Returns:
            List[Document]: The chunked documents with metadata.
        """
        strategy = strategy or ChunkingStrategy.DEFAULT

        if strategy == ChunkingStrategy.CODE:
            chunks: list[Document] = list(self.code_splitter.create_documents([content], [metadata]))
        elif strategy == ChunkingStrategy.MARKDOWN:
            chunks = self._chunk_markdown(content, metadata)
        else:
            chunks = list(self.default_splitter.create_documents([content], [metadata]))

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(chunks)
            chunk.metadata["content_hash"] = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            start_line, end_line = self._line_range(content, chunk)
            chunk.metadata["start_line"] = start_line
            chunk.metadata["end_line"] = end_line

        logger.debug("Split document into %d chunks", len(chunks))
        return chunks

    @staticmethod
    def _line_range(content: str, chunk: Document) -> tuple[int, int]:
        """Compute a chunk's 1-indexed line range within the original document.

        Uses the splitter's ``start_index`` metadata when available (accurate
        for single-pass splitting); falls back to locating the chunk text
        directly in ``content`` for two-pass markdown splitting, where
        ``start_index`` is only relative to the header-split slice.
        """
        start_index = chunk.metadata.get("start_index")
        if not isinstance(start_index, int):
            start_index = content.find(chunk.page_content)
        if start_index < 0:
            # MarkdownHeaderTextSplitter strips per-line whitespace, so a chunk
            # containing indented lines won't appear verbatim in the original.
            # Its first line (usually the section header) is unindented and
            # still locates the chunk.
            first_line = chunk.page_content.split("\n", 1)[0].strip()
            start_index = content.find(first_line) if first_line else -1
        start_index = max(start_index, 0)

        start_line = content.count("\n", 0, start_index) + 1
        end_line = start_line + chunk.page_content.count("\n")
        return start_line, end_line

    def _chunk_markdown(self, content: str, metadata: dict[str, Any]) -> list[Document]:
        """Split markdown content using header-aware chunking."""
        md_header_splits = self.markdown_header_splitter.split_text(content)

        chunks: list[Document] = []
        for doc in md_header_splits:
            doc_metadata = metadata.copy()
            for key, value in doc.metadata.items():
                if key.startswith("header_") and value:
                    doc_metadata[key] = value

            if len(doc.page_content) > self.chunk_size:
                sub_chunks = self.markdown_splitter.create_documents([doc.page_content], [doc_metadata])
                for sub_chunk in sub_chunks:
                    # start_index here is relative to doc.page_content (the header-split
                    # slice), not the original content, so it would compute wrong line
                    # numbers if left in place. Drop it and let _line_range fall back
                    # to locating the chunk text directly in the original content.
                    sub_chunk.metadata.pop("start_index", None)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=doc.page_content, metadata=doc_metadata))

        return chunks

    def process_file(self, file_path: Path) -> list[Document]:
        """Process a file into chunked documents with appropriate metadata.

        Args:
            file_path: Path to the file to process.

        Returns:
            List[Document]: The chunked documents with metadata.
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "file_path": str(file_path),
            }

            strategy = self._determine_strategy(file_path)

            return self.chunk_document(content, metadata, strategy)

        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, e)
            return []
