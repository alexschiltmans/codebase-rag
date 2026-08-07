"""Unit tests for data_ingestion/chunking.py."""

import json
from pathlib import Path

import pytest

from codebase_rag.data_ingestion.chunking import (
    FALLBACK_MAX_SEQ_LENGTH,
    MAX_DERIVED_CHUNK_SIZE,
    ChunkingStrategy,
    DocumentChunker,
    chunking_fingerprint,
    derive_chunk_size,
)


class TestDeriveChunkSize:
    """Tests for chunk size derivation from the embedding model's window."""

    def test_larger_window_yields_larger_chunks(self) -> None:
        assert derive_chunk_size(2048) > derive_chunk_size(384)

    def test_smaller_window_yields_smaller_chunks(self) -> None:
        assert derive_chunk_size(128) < derive_chunk_size(384)

    def test_chunker_sizes_itself_from_the_window(self) -> None:
        small = DocumentChunker(max_seq_length=384)
        large = DocumentChunker(max_seq_length=2048)

        assert small.chunk_size == derive_chunk_size(384)
        assert large.chunk_size > small.chunk_size

    def test_overlap_scales_with_size(self) -> None:
        small = DocumentChunker(max_seq_length=384)
        large = DocumentChunker(max_seq_length=2048)

        assert 0 < small.chunk_overlap < small.chunk_size
        assert large.chunk_overlap > small.chunk_overlap

    def test_explicit_size_overrides_derivation(self) -> None:
        chunker = DocumentChunker(chunk_size=333, chunk_overlap=11, max_seq_length=2048)

        assert (chunker.chunk_size, chunker.chunk_overlap) == (333, 11)

    def test_unknown_window_falls_back_rather_than_guessing_high(self) -> None:
        assert DocumentChunker().chunk_size == derive_chunk_size(FALLBACK_MAX_SEQ_LENGTH)

    def test_non_positive_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            derive_chunk_size(0)

    def test_a_long_context_window_is_capped(self) -> None:
        """Qwen3-Embedding-0.6B declares 32768 tokens, which derives 52428 characters. Left uncapped,
        configuring it re-cuts the corpus into chunks 85x what ships and retrieval stops locating."""
        assert derive_chunk_size(32768) == MAX_DERIVED_CHUNK_SIZE

    def test_the_cap_does_not_touch_windows_below_it(self) -> None:
        assert derive_chunk_size(384) == 614

    def test_the_chunker_inherits_the_cap(self) -> None:
        chunker = DocumentChunker(max_seq_length=32768)

        assert chunker.chunk_size == MAX_DERIVED_CHUNK_SIZE
        assert chunker.chunk_overlap == 400


class TestDetermineStrategy:
    """Tests for DocumentChunker._determine_strategy."""

    def test_python_file_is_code(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("foo.py")) == ChunkingStrategy.CODE

    def test_notebook_file_is_notebook(self) -> None:
        chunker = DocumentChunker()
        assert chunker._determine_strategy(Path("notebook.ipynb")) == ChunkingStrategy.NOTEBOOK

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


class TestProcessNotebookFile:
    """Tests for DocumentChunker.process_file with .ipynb notebooks."""

    NOTEBOOK_JSON = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["def foo():\n", "    return 1\n"],
                    "outputs": [],
                    "execution_count": 1,
                },
                {
                    "cell_type": "markdown",
                    "source": "# Notebook Title\n\nSome explanation.",
                },
                {
                    "cell_type": "code",
                    "source": "plt.imshow(data)",
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {"image/png": "iVBORw0KGgoAAAANSUhEUgAAAAUA/////wAAAP///w=="},
                        }
                    ],
                    "execution_count": 2,
                },
            ]
        }
    )

    def test_mixed_notebook_splits_code_and_markdown(self, tmp_path: Path) -> None:
        file_path = tmp_path / "example.ipynb"
        file_path.write_text(self.NOTEBOOK_JSON, encoding="utf-8")

        chunker = DocumentChunker()
        chunks = chunker.process_file(file_path)

        code_chunks = [c for c in chunks if c.metadata["notebook_cell_type"] == "code"]
        markdown_chunks = [c for c in chunks if c.metadata["notebook_cell_type"] == "markdown"]

        assert code_chunks
        assert markdown_chunks
        assert any("def foo" in c.page_content for c in code_chunks)
        assert any("plt.imshow" in c.page_content for c in code_chunks)
        assert any("Notebook Title" in c.page_content for c in markdown_chunks)

        for chunk in chunks:
            assert "iVBORw0KGgo" not in chunk.page_content
            assert '"cell_type"' not in chunk.page_content
            assert '"execution_count"' not in chunk.page_content

    def test_notebook_chunks_keep_chunk_document_metadata(self, tmp_path: Path) -> None:
        file_path = tmp_path / "example.ipynb"
        file_path.write_text(self.NOTEBOOK_JSON, encoding="utf-8")

        chunker = DocumentChunker()
        chunks = chunker.process_file(file_path)

        assert chunks
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata
            assert "content_hash" in chunk.metadata

    def test_broken_notebook_json_returns_empty_list(self, tmp_path: Path) -> None:
        file_path = tmp_path / "broken.ipynb"
        file_path.write_text("{not valid json", encoding="utf-8")

        chunker = DocumentChunker()
        chunks = chunker.process_file(file_path)

        assert chunks == []

    def test_notebook_missing_cells_key_returns_empty_list(self, tmp_path: Path) -> None:
        file_path = tmp_path / "no-cells.ipynb"
        file_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")

        chunker = DocumentChunker()
        chunks = chunker.process_file(file_path)

        assert chunks == []

    def test_notebook_chunk_index_is_unique_across_code_and_markdown_groups(self, tmp_path: Path) -> None:
        notebook_json = json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": "x = 1\n" * 40},
                    {"cell_type": "markdown", "source": "word " * 200},
                ]
            }
        )
        file_path = tmp_path / "multi-chunk.ipynb"
        file_path.write_text(notebook_json, encoding="utf-8")

        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.process_file(file_path)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert len(indices) > 2
        assert indices == list(range(len(chunks)))
        assert all(c.metadata["chunk_count"] == len(chunks) for c in chunks)


class TestChunkingFingerprint:
    """Tests for chunking_fingerprint, which caches use to spot resized chunks."""

    def test_same_settings_fingerprint_alike(self) -> None:
        a = chunking_fingerprint(DocumentChunker(max_seq_length=384), "some/model")
        b = chunking_fingerprint(DocumentChunker(max_seq_length=384), "some/model")

        assert a == b

    def test_a_different_window_changes_the_fingerprint(self) -> None:
        small = chunking_fingerprint(DocumentChunker(max_seq_length=384), "some/model")
        large = chunking_fingerprint(DocumentChunker(max_seq_length=2048), "some/model")

        assert small != large

    def test_a_different_model_changes_the_fingerprint(self) -> None:
        one = chunking_fingerprint(DocumentChunker(max_seq_length=384), "some/model")
        other = chunking_fingerprint(DocumentChunker(max_seq_length=384), "other/model")

        assert one != other

    def test_an_explicit_size_change_at_one_window_changes_the_fingerprint(self) -> None:
        derived = chunking_fingerprint(DocumentChunker(max_seq_length=384), "some/model")
        pinned = chunking_fingerprint(DocumentChunker(chunk_size=900, max_seq_length=384), "some/model")

        assert derived != pinned
