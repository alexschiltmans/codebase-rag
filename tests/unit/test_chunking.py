"""Unit tests for data_ingestion/chunking.py."""

from pathlib import Path

from codebase_rag.data_ingestion.chunking import ChunkingStrategy, DocumentChunker


class TestDetermineStrategy:
    """Tests for DocumentChunker._determine_strategy."""

    def test_python_file_is_code(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("foo.py")) == ChunkingStrategy.CODE

    def test_notebook_file_is_code(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("notebook.ipynb")) == ChunkingStrategy.CODE

    def test_markdown_file_is_markdown(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("README.md")) == ChunkingStrategy.MARKDOWN

    def test_rst_file_is_markdown(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("docs/index.rst")) == ChunkingStrategy.MARKDOWN

    def test_other_file_is_default(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("config.yaml")) == ChunkingStrategy.DEFAULT

    def test_case_insensitive_suffix(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("Foo.PY")) == ChunkingStrategy.CODE


class TestChunkDocument:
    """Tests for DocumentChunker.chunk_document."""

    def test_default_strategy_chunks_and_tags_metadata(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        content = "a" * 120
        chunks = chunker.chunk_document(content, {"source": "test.txt"})

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["chunk_count"] == len(chunks)
            assert chunk.metadata["source"] == "test.txt"
            assert "content_hash" in chunk.metadata

    def test_code_strategy_produces_chunks(self) -> None:
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        content = "def foo():\n    return 1\n"
        chunks = chunker.chunk_document(content, {"source": "foo.py"}, ChunkingStrategy.CODE)

        assert len(chunks) == 1
        assert "def foo" in chunks[0].page_content

    def test_defaults_to_default_strategy_when_none_given(self) -> None:
        chunker = DocumentChunker()
        chunks = chunker.chunk_document("short content", {"source": "x"})
        assert len(chunks) == 1

    def test_markdown_strategy_keeps_headers_and_short_sections_whole(self) -> None:
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=0)
        content = "# Title\n\nSome intro text.\n\n## Installation\n\nRun pip install.\n"
        chunks = chunker.chunk_document(content, {"source": "README.md"}, ChunkingStrategy.MARKDOWN)

        joined = " ".join(c.page_content for c in chunks)
        assert "Installation" in joined
        assert any(c.metadata.get("header_1") == "Title" for c in chunks)

    def test_markdown_strategy_splits_long_sections(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        content = "# Title\n\n" + ("word " * 200)
        chunks = chunker.chunk_document(content, {"source": "README.md"}, ChunkingStrategy.MARKDOWN)

        assert len(chunks) > 1


class TestLineRanges:
    """Tests for start_line/end_line metadata on chunks."""

    def test_code_chunks_get_exact_line_ranges(self) -> None:
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=0)
        content = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
        chunks = chunker.chunk_document(content, {"source": "foo.py"}, ChunkingStrategy.CODE)

        assert chunks[0].metadata["start_line"] == 1
        assert chunks[0].metadata["end_line"] >= chunks[0].metadata["start_line"]

    def test_markdown_chunk_with_indented_content_locates_its_header_line(self) -> None:
        # MarkdownHeaderTextSplitter strips per-line whitespace, so the chunk
        # text won't appear verbatim in the original; the first-line fallback
        # must still find the section header instead of collapsing to line 1.
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=0)
        content = "# Title\n\nintro text\n\n## Section\n\n    indented code\n    more indented\n"
        chunks = chunker.chunk_document(content, {"source": "doc.md"}, ChunkingStrategy.MARKDOWN)

        section_chunk = next(c for c in chunks if "Section" in c.page_content)
        assert section_chunk.metadata["start_line"] == 5
        assert section_chunk.metadata["end_line"] >= 5

    def test_every_chunk_carries_line_metadata(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk_document("a" * 120, {"source": "test.txt"})

        for chunk in chunks:
            assert chunk.metadata["start_line"] >= 1
            assert chunk.metadata["end_line"] >= chunk.metadata["start_line"]


class TestProcessFile:
    """Tests for DocumentChunker.process_file."""

    def test_process_file_reads_and_chunks_content(self, tmp_path: Path) -> None:
        file_path = tmp_path / "example.py"
        file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")

        chunker = DocumentChunker()
        chunks = chunker.process_file(file_path)

        assert len(chunks) == 1
        assert chunks[0].metadata["file_name"] == "example.py"
        assert chunks[0].metadata["file_type"] == ".py"
        assert chunks[0].metadata["source"] == str(file_path)

    def test_process_file_returns_empty_list_on_read_error(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does-not-exist.py"

        chunker = DocumentChunker()
        chunks = chunker.process_file(missing_path)

        assert chunks == []
