"""Tests for BM25 corpus loading, including loading a subset of the repos present."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from codebase_rag.retrieval.bm25_search import BM25Retriever, load_bm25_corpus


def _write_corpus(corpus_dir: Path, repo: str, count: int = 1) -> None:
    docs = [
        Document(page_content=f"{repo} body {i}", metadata={"source": f"{repo}/file{i}.py", "repo": repo})
        for i in range(count)
    ]
    BM25Retriever(docs).save_json(corpus_dir / f"{repo}.json")


class TestLoadBm25Corpus:
    def test_no_repos_loads_everything(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, "repo-a")
        _write_corpus(tmp_path, "repo-b")

        loaded = load_bm25_corpus(tmp_path)

        assert sorted(doc.metadata["repo"] for doc in loaded) == ["repo-a", "repo-b"]

    def test_named_repos_load_only_those(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, "repo-a", count=2)
        _write_corpus(tmp_path, "repo-b", count=3)

        loaded = load_bm25_corpus(tmp_path, repos=["repo-a"])

        assert {doc.metadata["repo"] for doc in loaded} == {"repo-a"}
        assert len(loaded) == 2

    def test_a_missing_named_repo_returns_what_was_found(self, tmp_path: Path) -> None:
        """Deliberately not an error here. Callers that need a missing repo to stop the run check
        the directory themselves, so the rule lives in one place instead of two."""
        _write_corpus(tmp_path, "repo-a")

        loaded = load_bm25_corpus(tmp_path, repos=["repo-a", "never-ingested"])

        assert {doc.metadata["repo"] for doc in loaded} == {"repo-a"}

    def test_every_named_repo_missing_loads_nothing(self, tmp_path: Path) -> None:
        _write_corpus(tmp_path, "repo-a")

        assert load_bm25_corpus(tmp_path, repos=["never-ingested"]) == []

    def test_an_empty_repo_list_loads_nothing(self, tmp_path: Path) -> None:
        """An empty scope is not the same request as no scope, and must not fall back to loading
        the whole directory: silently widening it is how an unscoped run gets published as scoped."""
        _write_corpus(tmp_path, "repo-a")

        assert load_bm25_corpus(tmp_path, repos=[]) == []

    def test_a_missing_directory_loads_nothing(self, tmp_path: Path) -> None:
        assert load_bm25_corpus(tmp_path / "absent", repos=["repo-a"]) == []

    def test_a_repeated_repo_is_loaded_once(self, tmp_path: Path) -> None:
        """A doubled name doubles every document frequency and the corpus size, so BM25 would score
        against a corpus that does not exist while the report records the scope as satisfied."""
        _write_corpus(tmp_path, "repo-a", count=3)

        loaded = load_bm25_corpus(tmp_path, repos=["repo-a", "repo-a"])

        assert len(loaded) == 3

    def test_scope_order_does_not_change_document_order(self, tmp_path: Path) -> None:
        """BM25 ties break on insertion order, so the same scope written two ways would otherwise
        publish two different top-k lists for one declared corpus."""
        _write_corpus(tmp_path, "repo-a", count=2)
        _write_corpus(tmp_path, "repo-b", count=2)

        forward = load_bm25_corpus(tmp_path, repos=["repo-a", "repo-b"])
        reverse = load_bm25_corpus(tmp_path, repos=["repo-b", "repo-a"])

        assert [doc.metadata["source"] for doc in forward] == [doc.metadata["source"] for doc in reverse]

    def test_scoped_and_unscoped_agree_when_the_scope_is_everything(self, tmp_path: Path) -> None:
        """The property the change's no-op check rests on, pinned so it cannot drift: naming every
        repo present has to give the same corpus, in the same order, as naming none."""
        _write_corpus(tmp_path, "repo-a", count=2)
        _write_corpus(tmp_path, "repo-b", count=2)

        scoped = load_bm25_corpus(tmp_path, repos=["repo-b", "repo-a"])
        unscoped = load_bm25_corpus(tmp_path)

        assert [doc.metadata["source"] for doc in scoped] == [doc.metadata["source"] for doc in unscoped]

    def test_named_repos_skip_the_chunking_sidecar_directory(self, tmp_path: Path) -> None:
        """The unscoped path globs `*.json` and relies on the sidecar sitting one level down; the
        scoped path names its files, so it cannot pick the sidecar up however it is arranged."""
        _write_corpus(tmp_path, "repo-a")
        (tmp_path / "_meta").mkdir()
        (tmp_path / "_meta" / "chunking.json").write_text('{"chunk_size": 614}')

        loaded = load_bm25_corpus(tmp_path, repos=["repo-a"])

        assert {doc.metadata["repo"] for doc in loaded} == {"repo-a"}


class TestRepoRestriction:
    """The `repos` restriction, which is how `--repo` reaches the retriever."""

    @staticmethod
    def _docs() -> list[Document]:
        """Two documents carrying the query term, one per repo, plus filler.

        The filler is load-bearing: BM25 gives a negative IDF to any term appearing in more
        than roughly half the corpus, so a two-document corpus where both match scores every
        result at or below zero and `search` correctly returns nothing.
        """
        matching = [
            Document(page_content="target token", metadata={"source": "a.py", "repo": "repo-a"}),
            Document(page_content="target token", metadata={"source": "b.py", "repo": "repo-b"}),
        ]
        filler = [
            Document(page_content=f"unrelated body {i}", metadata={"source": f"f{i}.py", "repo": "repo-c"})
            for i in range(4)
        ]
        return matching + filler

    def _index(self, repos: list[str] | None) -> BM25Retriever:
        return BM25Retriever(self._docs(), repos=repos)

    def test_unrestricted_returns_every_repo(self) -> None:
        results = self._index(None).search("target", k=5)

        assert {doc.metadata["repo"] for doc, _ in results} == {"repo-a", "repo-b"}

    def test_restriction_drops_out_of_scope_matches(self) -> None:
        results = self._index(["repo-a"]).search("target", k=5)

        assert [doc.metadata["repo"] for doc, _ in results] == ["repo-a"]

    def test_restriction_applies_before_k(self) -> None:
        """`k` means k in-scope results, not whatever survives filtering the global top k.

        The out-of-scope document scores higher here, so a filter applied after the cut to
        k=1 would return nothing at all.
        """
        docs = [
            Document(page_content="target target target", metadata={"source": "b.py", "repo": "repo-b"}),
            Document(page_content="target", metadata={"source": "a.py", "repo": "repo-a"}),
            *(
                Document(page_content=f"unrelated body {i}", metadata={"source": f"f{i}.py", "repo": "repo-c"})
                for i in range(4)
            ),
        ]

        results = BM25Retriever(docs, repos=["repo-a"]).search("target", k=1)

        assert [doc.metadata["source"] for doc, _ in results] == ["a.py"]

    def test_restriction_does_not_move_scores(self) -> None:
        """Scoping selects from what was scored; it does not change what was scored.

        BM25 scores depend on corpus-wide document frequencies and average document length, so a
        restriction that narrowed the index instead would silently rescore the documents it kept.
        """
        unscoped = {doc.metadata["source"]: score for doc, score in self._index(None).search("target", k=5)}
        scoped = self._index(["repo-a"]).search("target", k=5)

        assert scoped
        for doc, score in scoped:
            assert score == unscoped[doc.metadata["source"]]

    def test_restriction_survives_a_round_trip_through_json(self, tmp_path: Path) -> None:
        """`load_json` is how every entry point builds this, so the scope has to be settable there."""
        path = tmp_path / "index.json"
        self._index(None).save_json(path)

        loaded = BM25Retriever.load_json(path, repos=["repo-b"])

        assert [doc.metadata["repo"] for doc, _ in loaded.search("target", k=5)] == ["repo-b"]
