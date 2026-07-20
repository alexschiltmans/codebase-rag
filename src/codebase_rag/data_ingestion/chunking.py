"""Text chunking strategies that preserve code structure and context."""

import hashlib
import json
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
    NOTEBOOK = "notebook"
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
        )

        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n", " ", ""],
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
        )

    def _determine_strategy(self, file_path: Path) -> ChunkingStrategy:
        """Determine the appropriate chunking strategy based on the file type.

        Args:
            file_path: Path to the file being processed.

        Returns:
            ChunkingStrategy: The chunking strategy to use.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".ipynb":
            return ChunkingStrategy.NOTEBOOK
        if suffix == ".py":
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

        logger.debug("Split document into %d chunks", len(chunks))
        return chunks

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
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=doc.page_content, metadata=doc_metadata))

        return chunks

    def _extract_notebook_cells(self, content: str) -> tuple[str, str]:
        """Extract code and markdown cell sources from notebook JSON.

        Args:
            content: Raw .ipynb file content.

        Returns:
            tuple[str, str]: Concatenated (code_text, markdown_text) cell sources.
        """
        notebook = json.loads(content)
        cells = notebook["cells"]

        code_sources: list[str] = []
        markdown_sources: list[str] = []

        for cell in cells:
            cell_type = cell["cell_type"]
            source = cell["source"]
            text = "".join(source) if isinstance(source, list) else source

            if cell_type == "code":
                code_sources.append(text)
            elif cell_type == "markdown":
                markdown_sources.append(text)

        return "\n\n".join(code_sources), "\n\n".join(markdown_sources)

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

            if strategy == ChunkingStrategy.NOTEBOOK:
                try:
                    code_text, markdown_text = self._extract_notebook_cells(content)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("Skipping unparseable notebook %s: %s", file_path, e)
                    return []

                chunks: list[Document] = []
                if code_text:
                    code_chunks = self.chunk_document(code_text, metadata, ChunkingStrategy.CODE)
                    for chunk in code_chunks:
                        chunk.metadata["notebook_cell_type"] = "code"
                    chunks.extend(code_chunks)
                if markdown_text:
                    markdown_chunks = self.chunk_document(markdown_text, metadata, ChunkingStrategy.MARKDOWN)
                    for chunk in markdown_chunks:
                        chunk.metadata["notebook_cell_type"] = "markdown"
                    chunks.extend(markdown_chunks)

                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["chunk_count"] = len(chunks)
                return chunks

            return self.chunk_document(content, metadata, strategy)

        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, e)
            return []
