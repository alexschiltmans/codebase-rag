"""Unit tests for evals/bench_retrieval.py's own logic: metric aggregation,
arm-record completeness, batching, collection naming, and teardown. Runnable
offline with a stubbed store; the sweeps themselves are not tests.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.bench_retrieval import (
    embed_corpus,
    ingested_repositories,
    main,
    run_benchmark,
    save_candidate_lists,
    slugify_model_name,
)


class TestSlugifyModelName:
    def test_slashes_and_dots_become_dashes(self) -> None:
        slug = slugify_model_name("sentence-transformers/all-mpnet-base-v2")
        assert slug == "sentence-transformers-all-mpnet-base-v2"

    def test_mixed_case_lowercased(self) -> None:
        assert slugify_model_name("Qwen/Qwen3-Embedding-4B") == "qwen-qwen3-embedding-4b"


class TestEmbedCorpus:
    def test_batches_at_configured_size(self) -> None:
        store = MagicMock()
        docs = [Document(page_content=f"doc {i}") for i in range(250)]

        embed_corpus(store, docs, batch_size=100)

        assert store.add_documents.call_count == 3
        call_sizes = [len(call.args[0]) for call in store.add_documents.call_args_list]
        assert call_sizes == [100, 100, 50]

    def test_returns_positive_build_time(self) -> None:
        store = MagicMock()
        docs = [Document(page_content="doc")]

        build_time = embed_corpus(store, docs, batch_size=10)

        assert build_time >= 0.0


class TestRunBenchmark:
    def test_aggregates_and_category_breakdown(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            (Document(page_content="c", metadata={"source": "enum.py"}), 0.9),
        ]
        testset = [
            {"question": "q1", "sources": ["enum.py"], "category": "factual_lookup"},
            {"question": "q2", "sources": ["missing.py"], "category": "conceptual"},
        ]

        result = run_benchmark(retriever, testset, depth=5)

        assert result["hit_rate"] == 0.5
        assert result["mrr"] == 0.5
        assert set(result["category_breakdown"]) == {"factual_lookup", "conceptual"}
        assert result["category_breakdown"]["factual_lookup"]["hit_rate"] == 1.0
        assert result["category_breakdown"]["conceptual"]["hit_rate"] == 0.0
        assert len(result["per_question"]) == 2
        assert len(result["candidate_lists"]) == 2
        assert len(result["per_query_latency_s"]) == 2

    def test_expected_failure_questions_are_not_scored(self) -> None:
        """The published eval excludes these from hit rate and MRR; scoring them here
        would make every number incomparable to the ablation."""
        retriever = MagicMock()
        retriever.search.return_value = []
        testset: list[dict[str, Any]] = [
            {"question": "q1", "sources": ["a.py"], "category": "x"},
            {"question": "q2", "sources": ["b.py"], "category": "x", "expected_failure": True},
        ]

        result = run_benchmark(retriever, testset, depth=5)

        assert len(result["per_question"]) == 1
        assert result["per_question"][0]["question"] == "q1"

    def test_search_called_with_depth(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = []
        testset = [{"question": "q1", "sources": ["a.py"], "category": "x"}]

        run_benchmark(retriever, testset, depth=50)

        retriever.search.assert_called_once_with("q1", k=50)


class TestIngestedRepositories:
    def test_deduplicates_and_sorts(self) -> None:
        docs = [
            Document(page_content="a", metadata={"repo": "beta"}),
            Document(page_content="b", metadata={"repo": "alpha"}),
            Document(page_content="c", metadata={"repo": "alpha"}),
        ]
        assert ingested_repositories(docs) == ["alpha", "beta"]

    def test_ignores_missing_repo_metadata(self) -> None:
        docs = [Document(page_content="a", metadata={})]
        assert ingested_repositories(docs) == []


class TestSaveCandidateLists:
    def test_writes_json_to_candidates_dir(self, tmp_path: Path) -> None:
        with patch("evals.bench_retrieval.CANDIDATES_DIR", tmp_path):
            path = save_candidate_lists([{"question": "q1", "candidates": []}], "arm-name")

        assert path.exists()
        assert path.parent == tmp_path


class TestMainTeardown:
    @patch("evals.bench_retrieval.load_bm25_corpus")
    @patch("evals.bench_retrieval.load_testset")
    @patch("evals.bench_retrieval.EmbeddingManager")
    @patch("evals.bench_retrieval.QdrantStore")
    def test_benchmark_collection_torn_down_on_success(
        self,
        mock_store_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_load_testset: MagicMock,
        mock_load_corpus: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load_testset.return_value = [{"question": "q1", "sources": ["a.py"], "category": "x"}]
        mock_load_corpus.return_value = [Document(page_content="a", metadata={"source": "a.py", "repo": "r"})]

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1, 0.2]
        mock_emb_cls.return_value = mock_emb

        mock_store = MagicMock()
        mock_store.embedding_manager.max_seq_length = 384
        mock_store.embedding_manager.query_prompt = ""
        mock_store.embedding_manager.document_prompt = ""
        mock_store.embedding_manager.dtype = None
        mock_store.embedding_manager.loaded_dtype = "float32"
        mock_store_cls.return_value = mock_store

        with (
            patch("evals.bench_retrieval.CANDIDATES_DIR", tmp_path),
            patch(
                "sys.argv",
                ["bench_retrieval.py", "--retriever", "vector", "--depth", "5", "--output", str(tmp_path / "o.json")],
            ),
            patch("codebase_rag.retrieval.vector_search.VectorRetriever.search", return_value=[]),
        ):
            main()

        mock_store.client.delete_collection.assert_any_call("bench_sentence-transformers-all-mpnet-base-v2_2")

    @patch("evals.bench_retrieval.load_bm25_corpus")
    @patch("evals.bench_retrieval.load_testset")
    @patch("evals.bench_retrieval.EmbeddingManager")
    @patch("evals.bench_retrieval.QdrantStore")
    def test_keep_collections_skips_teardown(
        self,
        mock_store_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_load_testset: MagicMock,
        mock_load_corpus: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_load_testset.return_value = [{"question": "q1", "sources": ["a.py"], "category": "x"}]
        mock_load_corpus.return_value = [Document(page_content="a", metadata={"source": "a.py", "repo": "r"})]

        mock_emb = MagicMock()
        mock_emb.get_query_embedding.return_value = [0.1, 0.2]
        mock_emb_cls.return_value = mock_emb

        mock_store = MagicMock()
        mock_store.embedding_manager.max_seq_length = 384
        mock_store.embedding_manager.query_prompt = ""
        mock_store.embedding_manager.document_prompt = ""
        mock_store.embedding_manager.dtype = None
        mock_store.embedding_manager.loaded_dtype = "float32"
        mock_store_cls.return_value = mock_store

        with (
            patch("evals.bench_retrieval.CANDIDATES_DIR", tmp_path),
            patch(
                "sys.argv",
                [
                    "bench_retrieval.py",
                    "--retriever",
                    "vector",
                    "--depth",
                    "5",
                    "--keep-collections",
                    "--output",
                    str(tmp_path / "o.json"),
                ],
            ),
            patch("codebase_rag.retrieval.vector_search.VectorRetriever.search", return_value=[]),
        ):
            main()

        mock_store.client.delete_collection.assert_not_called()
