"""Tests for the swept-corpus builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document

from codebase_rag.retrieval.bm25_search import BM25Retriever, load_bm25_corpus
from evals import build_corpus
from evals.build_corpus import repos_in
from evals.corpus_chunking import SIDECAR_DIR, read_chunking_sidecar, write_chunking_sidecar


class TestReposIn:
    def test_each_corpus_file_is_one_repo(self, tmp_path: Path) -> None:
        for name in ("power-grid-model", "click"):
            with open(tmp_path / f"{name}.json", "w") as f:
                json.dump([], f)
        assert repos_in(tmp_path) == ["click", "power-grid-model"]

    def test_an_empty_directory_has_no_repos(self, tmp_path: Path) -> None:
        assert repos_in(tmp_path) == []

    def test_the_sidecar_is_not_mistaken_for_a_repo(self, tmp_path: Path) -> None:
        # It lives one level down for exactly this reason, but a builder that listed it as a repo
        # would go looking for a checkout named "chunking" and fail confusingly.
        (tmp_path / SIDECAR_DIR).mkdir()
        with open(tmp_path / SIDECAR_DIR / "chunking.json", "w") as f:
            json.dump({"chunk_size": 614}, f)
        with open(tmp_path / "power-grid-model.json", "w") as f:
            json.dump([], f)
        assert repos_in(tmp_path) == ["power-grid-model"]


class TestBuiltCorpusShape:
    def test_a_built_corpus_is_loadable_and_carries_its_chunking(self, tmp_path: Path) -> None:
        # Guards the round trip the sweep depends on: what build_corpus writes, the benchmark reads,
        # and the chunking it records is the chunking the benchmark will report.
        docs = [Document(page_content="body", metadata={"source": "a.py", "repo": "demo"})]
        BM25Retriever(docs).save_json(tmp_path / "demo.json")
        write_chunking_sidecar(tmp_path, chunk_size=1800, chunk_overlap=360, max_seq_length=None)

        loaded = load_bm25_corpus(tmp_path)

        assert [doc.page_content for doc in loaded] == ["body"]
        assert loaded[0].metadata["repo"] == "demo"
        assert read_chunking_sidecar(tmp_path)["chunk_size"] == 1800


class TestRefusals:
    @pytest.mark.parametrize("size", [0, -1])
    def test_a_nonpositive_chunk_size_is_refused(self, size: int, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["build_corpus.py", "--chunk-size", str(size)])
        with pytest.raises(SystemExit, match="chunk size must be positive"):
            build_corpus.main()

    def test_writing_over_the_application_corpus_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The running app serves BM25 from this directory, and the sweep is not allowed to be the
        # reason it goes missing.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "build_corpus.py",
                "--chunk-size",
                "614",
                "--repo",
                "demo",
                "--out-dir",
                str(build_corpus.DEFAULT_CORPUS_DIR),
            ],
        )
        with pytest.raises(SystemExit, match="that is the corpus the application serves from"):
            build_corpus.main()

    def test_a_repo_file_from_a_wider_build_is_refused(self, tmp_path: Path) -> None:
        """`load_bm25_corpus` merges every JSON in the directory, so a leftover from an earlier,
        wider build gets scored alongside this one while the sidecar names a single chunking."""
        with open(tmp_path / "left-behind.json", "w") as f:
            json.dump([], f)

        with pytest.raises(SystemExit, match=r"left-behind\.json"):
            build_corpus.build_corpus(["demo"], chunk_size=614, out_dir=tmp_path)
