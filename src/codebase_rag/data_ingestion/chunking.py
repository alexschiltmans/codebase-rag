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

# Characters per token used to turn a model's token budget into a character
# budget. Measured on this corpus with the mpnet tokenizer: the median is
# around 3.0, but the median is the wrong statistic here because a chunk only
# has to be dense once to overflow. This is the 1st percentile of the densest
# file type users query (Markdown tables), which keeps the p99 token length of
# .cpp, .py, .hpp and .md inside a 384-token window.
#
# It is one tokenizer's number applied to every model. A model with a coarser
# vocabulary packs more characters per token, so this under-fills its window
# and produces chunks smaller than they need to be. That errs toward less
# context rather than toward silent truncation, and the ingest-time truncation
# report is what would show the ratio being wrong in the dangerous direction.
CHARS_PER_TOKEN = 1.6

# Overlap as a share of chunk size, carried over from the previous 200/1000.
OVERLAP_RATIO = 0.2

# Used only when no caller knows the configured model's window. The ingestion
# path passes the real value; this is the smallest window this project has run
# against, so guessing it low keeps chunks readable by any model.
FALLBACK_MAX_SEQ_LENGTH = 384


def derive_chunk_size(max_seq_length: int) -> int:
    """Return the character chunk size that fits a model's token window."""
    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    return int(max_seq_length * CHARS_PER_TOKEN)


def chunking_fingerprint(chunker: "DocumentChunker", embedding_model: str) -> str:
    """Identify the chunking that produced a set of chunks.

    Every cache in the ingestion path answers "has this file changed", never
    "were these chunks cut the same way". Without this, changing the chunk size
    or the embedding model leaves a hash-matched file untouched and its
    old-size chunks in the index for good.
    """
    return f"{embedding_model}|{chunker.max_seq_length}|{chunker.chunk_size}|{chunker.chunk_overlap}"


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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        """Initialize the DocumentChunker.

        Args:
            chunk_size: Target size of each chunk in characters. Derived from
                ``max_seq_length`` when omitted.
            chunk_overlap: Number of characters to overlap between chunks.
                Defaults to a fixed share of the chunk size.
            max_seq_length: Token window of the embedding model the chunks are
                destined for. Falls back to the smallest window this project
                has run against when the caller doesn't know it.
        """
        self.max_seq_length = max_seq_length if max_seq_length is not None else FALLBACK_MAX_SEQ_LENGTH
        self.chunk_size = chunk_size if chunk_size is not None else derive_chunk_size(self.max_seq_length)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(self.chunk_size * OVERLAP_RATIO)

        logger.debug(
            "Chunking at %d characters with %d overlap for a %d-token window",
            self.chunk_size,
            self.chunk_overlap,
            self.max_seq_length,
        )

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
