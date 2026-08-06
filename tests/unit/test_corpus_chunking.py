"""Tests for the benchmark corpus chunking sidecar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from codebase_rag.retrieval.bm25_search import load_bm25_corpus
from evals.corpus_chunking import (
    SIDECAR_DIR,
    SIDECAR_NAME,
    chunking_suffix,
    read_chunking_sidecar,
    write_chunking_sidecar,
)


class TestWriteAndRead:
    def test_a_written_chunking_reads_back_unchanged(self, tmp_path: Path) -> None:
        write_chunking_sidecar(tmp_path, chunk_size=614, chunk_overlap=122, max_seq_length=384)
        assert read_chunking_sidecar(tmp_path) == {
            "chunk_size": 614,
            "chunk_overlap": 122,
            "chunk_max_seq_length": 384,
        }

    def test_an_explicitly_chosen_size_records_no_sequence_length(self, tmp_path: Path) -> None:
        # A swept size was not derived from any model's window, and claiming one would say the
        # size belongs to a model it was never measured against.
        write_chunking_sidecar(tmp_path, chunk_size=1800, chunk_overlap=360, max_seq_length=None)
        assert read_chunking_sidecar(tmp_path)["chunk_max_seq_length"] is None

    def test_the_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "bm25_corpus_chunk1000"
        write_chunking_sidecar(target, chunk_size=1000, chunk_overlap=200, max_seq_length=None)
        assert (target / SIDECAR_DIR / SIDECAR_NAME).exists()

    def test_the_sidecar_does_not_break_corpus_loading(self, tmp_path: Path) -> None:
        # `load_bm25_corpus` globs `*.json` and reads every match as a document list, so a sidecar
        # written beside the corpus files is a crash rather than a file it ignores.
        with open(tmp_path / "power-grid-model.json", "w") as f:
            json.dump([{"page_content": "hello", "metadata": {"source": "a.py"}}], f)
        write_chunking_sidecar(tmp_path, chunk_size=614, chunk_overlap=122, max_seq_length=384)

        documents = load_bm25_corpus(tmp_path)

        assert [doc.page_content for doc in documents] == ["hello"]


class TestUnrecordedChunking:
    def test_a_corpus_without_a_sidecar_reads_as_unrecorded(self, tmp_path: Path) -> None:
        assert read_chunking_sidecar(tmp_path) == {
            "chunk_size": None,
            "chunk_overlap": None,
            "chunk_max_seq_length": None,
        }

    def test_unrecorded_is_not_confused_with_a_recorded_value(self, tmp_path: Path) -> None:
        recorded = tmp_path / "recorded"
        write_chunking_sidecar(recorded, chunk_size=614, chunk_overlap=122, max_seq_length=384)
        assert read_chunking_sidecar(recorded) != read_chunking_sidecar(tmp_path / "missing")

    def test_a_partial_sidecar_fills_the_absent_fields_with_none(self, tmp_path: Path) -> None:
        (tmp_path / SIDECAR_DIR).mkdir()
        with open(tmp_path / SIDECAR_DIR / SIDECAR_NAME, "w") as f:
            json.dump({"chunk_size": 614}, f)
        assert read_chunking_sidecar(tmp_path) == {
            "chunk_size": 614,
            "chunk_overlap": None,
            "chunk_max_seq_length": None,
        }

    def test_an_unrelated_key_is_not_carried_into_the_arm_record(self, tmp_path: Path) -> None:
        (tmp_path / SIDECAR_DIR).mkdir()
        with open(tmp_path / SIDECAR_DIR / SIDECAR_NAME, "w") as f:
            json.dump({"chunk_size": 614, "chunk_overlap": 122, "unrelated": "value"}, f)
        assert "unrelated" not in read_chunking_sidecar(tmp_path)


class TestArmNameSuffix:
    def test_a_recorded_size_appears_in_the_arm_name(self) -> None:
        assert chunking_suffix({"chunk_size": 614, "chunk_overlap": 122}) == "_chunk614"

    def test_an_unrecorded_chunking_adds_nothing(self) -> None:
        # Arms saved before the field existed keep the names they already have, rather than being
        # renamed into a population they do not belong to.
        assert chunking_suffix({"chunk_size": None, "chunk_overlap": None}) == ""

    def test_two_sizes_produce_different_names(self) -> None:
        # This is the failure the field exists to prevent: without it, the second run overwrites
        # the arm it should have been compared against.
        assert chunking_suffix({"chunk_size": 614}) != chunking_suffix({"chunk_size": 1000})
