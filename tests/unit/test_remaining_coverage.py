"""Additional tests for rag_chain, retrieval, ollama_client, and git_loader."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest
import requests
from langchain_core.documents import Document

from codebase_rag.data_ingestion.git_loader import GitLoader
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval import bm25_search, vector_search
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever


class TestRAGChainConversationMemory:
    """Tests for RAGChain conversation memory features."""

    def _make_chain(self, **kwargs) -> RAGChain:
        retriever = MagicMock()
        llm = MagicMock()
        return RAGChain(retriever=retriever, llm=llm, **kwargs)

    def test_add_user_message(self) -> None:
        chain = self._make_chain(max_conversation_history=3)
        chain.add_user_message("hello")
        assert len(chain.conversation_history) == 1
        assert chain.conversation_history[0]["role"] == "user"

    def test_add_assistant_message_with_sources(self) -> None:
        chain = self._make_chain()
        sources = [{"id": "1", "file_path": "a.py", "file_name": "a.py"}]
        chain.add_assistant_message("answer", sources)
        assert chain.conversation_history[0]["sources"] == sources

    def test_conversation_memory_disabled(self) -> None:
        chain = self._make_chain(use_conversation_memory=False)
        chain.add_user_message("hello")
        chain.add_assistant_message("world")
        assert len(chain.conversation_history) == 0

    def test_trim_conversation_history(self) -> None:
        chain = self._make_chain(max_conversation_history=2)
        for i in range(5):
            chain.add_user_message(f"q{i}")
            chain.add_assistant_message(f"a{i}")

        user_msgs = [m for m in chain.conversation_history if m["role"] == "user"]
        assert len(user_msgs) <= 2

    def test_format_conversation_history_empty(self) -> None:
        chain = self._make_chain()
        result = chain._format_conversation_history()
        assert "No previous conversation" in result

    def test_trim_removes_orphaned_assistant_reply(self) -> None:
        chain = self._make_chain(max_conversation_history=2)
        chain.conversation_history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        chain._trim_conversation_history()
        assert [m["content"] for m in chain.conversation_history] == ["u2", "a2", "u3", "a3"]

    def test_trim_non_alternating_history(self) -> None:
        chain = self._make_chain(max_conversation_history=2)
        chain.conversation_history = [
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        chain._trim_conversation_history()
        assert [m["content"] for m in chain.conversation_history] == ["u2", "a2", "u3", "a3"]
        assert chain.conversation_history[0]["role"] == "user"

    def test_trim_short_history_untouched(self) -> None:
        chain = self._make_chain(max_conversation_history=2)
        chain.conversation_history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        chain._trim_conversation_history()
        assert [m["content"] for m in chain.conversation_history] == ["u1", "a1"]

    def test_trim_max_conversation_history_one(self) -> None:
        chain = self._make_chain(max_conversation_history=1)
        chain.conversation_history = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        chain._trim_conversation_history()
        assert [m["content"] for m in chain.conversation_history] == ["u2", "a2"]

    def test_trim_history_starting_with_assistant_message(self) -> None:
        chain = self._make_chain(max_conversation_history=1)
        chain.conversation_history = [
            {"role": "assistant", "content": "stray"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        chain._trim_conversation_history()
        assert [m["content"] for m in chain.conversation_history] == ["u2", "a2"]
        assert chain.conversation_history[0]["role"] == "user"

    def test_format_conversation_history_with_messages(self) -> None:
        chain = self._make_chain()
        chain.add_user_message("What is X?")
        chain.add_assistant_message("X is a thing.")
        result = chain._format_conversation_history()
        assert "User: What is X?" in result
        assert "Assistant: X is a thing." in result

    def test_create_context_empty(self) -> None:
        chain = self._make_chain()
        result = chain._create_context([])
        assert "No relevant information" in result

    def test_create_context_with_docs(self) -> None:
        chain = self._make_chain()
        docs = [
            Document(page_content="Code snippet", metadata={"source": "test.py"}),
        ]
        result = chain._create_context(docs)
        assert "Code snippet" in result
        assert "test.py" in result

    def test_build_within_budget_no_budget_set_leaves_docs_untouched(self) -> None:
        chain = self._make_chain()
        docs = [Document(page_content="x" * 100, metadata={"source": f"f{i}.py"}) for i in range(5)]
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(
            "question", docs
        )
        assert used_docs == docs
        assert docs_dropped == 0
        assert history_dropped == 0
        assert question_truncated == 0
        assert "question" in prompt

    def test_build_within_budget_trims_lowest_ranked_docs_first(self) -> None:
        chain = self._make_chain(prompt_budget_chars=600)
        docs = [Document(page_content="x" * 200, metadata={"source": f"f{i}.py"}) for i in range(5)]
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(
            "question", docs
        )
        assert docs_dropped > 0
        assert used_docs == docs[: len(docs) - docs_dropped]
        assert len(prompt) <= chain.prompt_budget_chars
        assert question_truncated == 0
        assert "question" in prompt

    def test_build_within_budget_drops_history_after_docs_exhausted(self) -> None:
        chain = self._make_chain(use_conversation_memory=True, prompt_budget_chars=300)
        chain.add_user_message("a" * 200)
        chain.add_assistant_message("b" * 200)
        docs = [Document(page_content="x" * 200, metadata={"source": "f.py"})]
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(
            "question", docs
        )
        assert docs_dropped == 1
        assert used_docs == []
        assert history_dropped > 0
        assert len(prompt) <= chain.prompt_budget_chars

    def test_build_within_budget_drops_oldest_history_with_uneven_lengths(self) -> None:
        """Regression test: uneven message lengths used to make the drop loop report a fit while still over budget."""
        chain = self._make_chain(use_conversation_memory=True, prompt_budget_chars=350)
        chain.add_user_message("old")
        chain.add_assistant_message("new assistant " + "n" * 500)
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget("q", [])
        assert len(prompt) <= chain.prompt_budget_chars
        assert history_dropped > 0

    def test_build_within_budget_keeps_newest_history_drops_oldest(self) -> None:
        """Regression test: the retention policy is oldest-first, so a partial drop must keep the newest turn."""
        chain = self._make_chain(use_conversation_memory=True, prompt_budget_chars=240)
        chain.add_user_message("oldest question")
        chain.add_assistant_message("oldest answer")
        chain.add_user_message("newest question")
        chain.add_assistant_message("newest answer")
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget("q", [])
        assert len(prompt) <= chain.prompt_budget_chars
        assert 0 < history_dropped < 4
        assert "oldest question" not in prompt
        assert "newest question" in prompt

    def test_build_within_budget_within_budget_untouched(self) -> None:
        chain = self._make_chain(prompt_budget_chars=100_000)
        docs = [Document(page_content="short", metadata={"source": "f.py"})]
        _, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget("question", docs)
        assert used_docs == docs
        assert docs_dropped == 0
        assert history_dropped == 0
        assert question_truncated == 0

    def test_build_within_budget_truncates_question_as_last_resort(self) -> None:
        chain = self._make_chain(prompt_budget_chars=2000)
        query = "q" * 5000
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(query, [])
        assert question_truncated > 0
        assert "[truncated]" in prompt
        assert used_docs == []
        assert query not in prompt
        assert len(prompt) <= chain.prompt_budget_chars

    def test_build_within_budget_truncation_skipped_when_it_would_grow_the_prompt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression test: a query shorter than the elision marker used to grow the prompt instead of shrinking it."""
        chain = self._make_chain(prompt_budget_chars=160)
        query = "hi"
        with caplog.at_level("WARNING", logger="codebase_rag.llm.rag_chain"):
            prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(query, [])
        assert used_docs == []
        assert question_truncated == 0
        assert "hi" in prompt
        assert "[truncated]" not in prompt
        assert len(prompt) > chain.prompt_budget_chars
        assert any("still exceeds budget" in r.message for r in caplog.records)

    def test_build_within_budget_stops_dropping_docs_that_would_grow_the_prompt(self) -> None:
        """Regression test: a doc shorter than the empty-context sentinel used to get dropped, growing the prompt."""
        chain = self._make_chain(prompt_budget_chars=10_000)
        doc = Document(page_content="x", metadata={})
        prompt, used_docs, docs_dropped, history_dropped, question_truncated = chain._build_within_budget(
            "q" * 20_000, [doc]
        )
        assert docs_dropped == 0
        assert used_docs == [doc]
        assert len(prompt) <= chain.prompt_budget_chars

    def test_build_within_budget_excludes_current_query_from_history(self) -> None:
        chain = self._make_chain(use_conversation_memory=True, prompt_budget_chars=100_000)
        chain.add_user_message("current question")
        prompt, _, _, history_dropped, _ = chain._build_within_budget("current question", [])
        assert prompt.count("current question") == 1
        assert history_dropped == 0

    def test_retrieve_documents_calls_search_once(self) -> None:
        """Retrieval goes through the protocol's `search(query, k)` exactly
        once — no attribute probing, no fallback call path."""
        chain = self._make_chain()
        chain.retriever.search.return_value = [(Document(page_content="doc", metadata={}), 0.9)]

        result = chain._retrieve_documents("query", 5)

        assert len(result) == 1
        chain.retriever.search.assert_called_once_with("query", 5)

    def test_retriever_type_error_propagates(self) -> None:
        """Regression test for SE-1: a TypeError raised *inside* a retriever
        must reach the caller instead of being swallowed by argument-dispatch
        logic and silently retried without top_k."""
        chain = self._make_chain()
        chain.retriever.search.side_effect = TypeError("bug inside the retriever")

        with pytest.raises(TypeError, match="bug inside the retriever"):
            chain.run("test query")

        chain.retriever.search.assert_called_once()

    def test_run_with_generation_error(self) -> None:
        chain = self._make_chain()
        chain.retriever.search.return_value = [(Document(page_content="doc", metadata={"source": "a.py"}), 0.9)]
        chain.llm.invoke.side_effect = RuntimeError("LLM error")

        with pytest.raises(RuntimeError, match="LLM error"):
            chain.run("test query")

    def test_format_sources_with_plain_docs(self) -> None:
        chain = self._make_chain()
        docs = [
            Document(page_content="text", metadata={"source": "src/main.py", "repo": "myrepo"}),
        ]
        sources = chain._format_sources(docs)
        assert len(sources) == 1
        assert "[MYREPO]" in sources[0]["file_name"]

    def test_empty_retrieval_returns_refusal_answer(self) -> None:
        """Regression test for AI-1: an out-of-scope query, for which the
        retriever returns no documents, must reach the refusal answer and
        cite no sources — not a hallucinated answer from stale context."""
        chain = self._make_chain()
        chain.retriever.search.return_value = []

        result = chain.run("what's a good lasagna recipe?")

        assert result["documents"] == []
        assert result["sources"] == []
        assert "couldn't find any relevant information" in result["answer"]
        chain.llm.invoke.assert_not_called()


class TestHybridRetrieverExtra:
    """Additional tests for HybridRetriever."""

    def test_search_no_bm25(self) -> None:

        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            (Document(page_content="vec result", metadata={"source": "a.py", "chunk_index": 0}), 0.9),
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector, bm25_retriever=None)
        results = retriever.search("test")

        assert len(results) == 1

    def test_search_error_propagates(self) -> None:

        mock_vector = MagicMock()
        mock_vector.search.side_effect = RuntimeError("error")

        retriever = HybridRetriever(vector_retriever=mock_vector)
        with pytest.raises(RuntimeError, match="error"):
            retriever.search("test")

    def test_empty_bm25_index_does_not_rescale_vector_scores(self) -> None:
        """Regression test: normalization keys off the rankers that actually
        returned results, not off which retriever objects exist.

        Before this fix, a configured-but-empty BM25 index (exactly what the
        app holds before its first ingest) still contributed its weight to
        the denominator, capping every fused score at vector_weight (0.7)
        even though BM25 returned nothing.
        """
        doc = Document(page_content="top hit", metadata={"source": "a.py", "chunk_index": 0})
        mock_vector = MagicMock()
        mock_vector.search.return_value = [(doc, 0.95)]

        with_empty_index = HybridRetriever(mock_vector, BM25Retriever([])).search("query")
        without_bm25 = HybridRetriever(mock_vector, None).search("query")

        assert with_empty_index[0][1] == pytest.approx(1.0)
        assert with_empty_index[0][1] == pytest.approx(without_bm25[0][1])

    def test_both_rankers_contributing_scores_top_doc_1(self) -> None:
        """A document ranked #1 by both rankers still scores exactly 1.0."""
        doc = Document(page_content="top hit", metadata={"source": "a.py", "chunk_index": 0})
        mock_vector = MagicMock()
        mock_vector.search.return_value = [(doc, 0.95)]
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [(doc, 12.0)]

        results = HybridRetriever(mock_vector, mock_bm25).search("query")

        assert results[0][1] == pytest.approx(1.0)

    def test_both_components_empty_returns_empty(self) -> None:
        """Regression test: HybridRetriever no longer filters on fused score,
        so the only way to get [] back is both components returning nothing —
        which is what should happen for an out-of-scope query once the
        vector retriever's own similarity threshold has done its job."""
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(vector_retriever=mock_vector, bm25_retriever=mock_bm25)
        results = retriever.search("off topic query")

        assert results == []

    def test_low_rank_results_are_not_filtered_by_fused_score(self) -> None:
        """A document found only by BM25 at a low rank still comes back:
        fused scores are no longer used for relevance filtering."""
        mock_vector = MagicMock()
        mock_vector.search.return_value = []
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            (Document(page_content=f"doc{i}", metadata={"source": f"{i}.py", "chunk_index": 0}), 1.0 / (i + 1))
            for i in range(5)
        ]

        retriever = HybridRetriever(vector_retriever=mock_vector, bm25_retriever=mock_bm25, top_k=5)
        results = retriever.search("keyword query")

        assert len(results) == 5


class TestVectorRetrieverExtra:
    """Additional tests for VectorRetriever."""

    def test_search_empty_results(self) -> None:

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        retriever = VectorRetriever(mock_store)
        results = retriever.search("query")
        assert results == []

    def test_search_error_propagates(self) -> None:

        mock_store = MagicMock()
        mock_store.similarity_search_with_score.side_effect = RuntimeError("error")

        retriever = VectorRetriever(mock_store)
        with pytest.raises(RuntimeError, match="error"):
            retriever.search("query")

    def test_score_threshold_filters_low_scores(self) -> None:
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="high", metadata={}), 0.8),
            (Document(page_content="mid", metadata={}), 0.3),
            (Document(page_content="low", metadata={}), 0.1),
        ]

        retriever = VectorRetriever(mock_store, score_threshold=0.25)
        results = retriever.search("query")

        assert [doc.page_content for doc, _ in results] == ["high", "mid"]

    def test_k_none_resolves_to_default(self) -> None:
        """Protocol contract: `k=None` means the retriever's own default,
        sourced from the module constant rather than an inline literal."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []

        VectorRetriever(mock_store).search("query")

        mock_store.similarity_search_with_score.assert_called_once_with("query", vector_search.DEFAULT_TOP_K)

    def test_no_threshold_returns_everything(self) -> None:
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="high", metadata={}), 0.8),
            (Document(page_content="low", metadata={}), 0.01),
        ]

        retriever = VectorRetriever(mock_store, score_threshold=None)
        results = retriever.search("query")

        assert len(results) == 2


class TestBM25RetrieverExtra:
    """Additional tests for BM25Retriever."""

    def test_empty_documents(self) -> None:

        retriever = BM25Retriever([])
        assert retriever.bm25 is None

    def test_search_with_empty_index(self) -> None:

        retriever = BM25Retriever([])
        results = retriever.search("query")
        assert results == []

    def test_preprocess_text(self) -> None:

        retriever = BM25Retriever([Document(page_content="test", metadata={})])
        tokens = retriever._preprocess_text("Hello World! Test 123 a")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens
        assert "a" not in tokens

    def test_k_none_resolves_to_default(self) -> None:
        """Protocol contract: `k=None` means the retriever's own default,
        sourced from the module constant rather than an inline literal.

        Five of twenty documents match, so the cap — not the number of
        matches — is what limits the result count. The ratio matters: BM25
        gives a term appearing in more than about half the corpus a negative
        IDF, and those documents are then dropped by the score > 0 filter.
        """
        docs = [Document(page_content=f"transformer winding document {i}", metadata={}) for i in range(5)]
        docs += [Document(page_content=f"unrelated content number {i}", metadata={}) for i in range(15)]
        retriever = BM25Retriever(docs)

        results = retriever.search("transformer winding")

        assert len(results) == bm25_search.DEFAULT_TOP_K
        assert all(score > 0 for _, score in results)

    def test_search_no_term_overlap_returns_empty(self) -> None:
        """Regression test: a query with no term overlap in the corpus must
        return [], not `k` zero-scored documents padded in to fill the
        result list (the bug that kept HybridRetriever's refusal path
        unreachable — see fix-rrf-relevance-thresholds design.md)."""
        retriever = BM25Retriever(
            [
                Document(page_content="power grid model calculation", metadata={}),
                Document(page_content="short circuit analysis", metadata={}),
            ]
        )
        results = retriever.search("xylophone marmalade quokka", k=4)
        assert results == []

    def test_search_partial_overlap_omits_non_matching_docs(self) -> None:
        retriever = BM25Retriever(
            [
                Document(page_content="power grid model calculation", metadata={}),
                Document(page_content="short circuit analysis", metadata={}),
                Document(page_content="completely unrelated text about kazoos", metadata={}),
            ]
        )
        results = retriever.search("power grid", k=5)
        assert len(results) == 1
        assert results[0][0].page_content == "power grid model calculation"
        assert results[0][1] > 0


class TestOllamaClientExtra:
    """Additional tests for OllamaClient edge cases."""

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_non_200(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.llm_model_name = "test"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        client = OllamaClient(model_name="test")
        result = client.check_connection()
        assert result["status"] == "error"
        assert "500" in result["message"]

    @patch("codebase_rag.llm.ollama_client.requests.get")
    @patch("codebase_rag.llm.ollama_client.Config")
    def test_check_connection_request_exception(self, mock_config_cls: MagicMock, mock_get: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.llm_model_name = "test"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_num_ctx = 8192
        mock_config_cls.get_instance.return_value = mock_config

        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        client = OllamaClient(model_name="test")
        result = client.check_connection()
        assert result["status"] == "error"


class TestGitLoaderExtra:
    """Additional tests for GitLoader."""

    @patch("codebase_rag.data_ingestion.git_loader.Config")
    def test_clone_or_pull_no_url_raises(self, mock_config_cls: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.repo_urls = []
        mock_config.repo_local_path = Path("/tmp/nonexistent")
        mock_config_cls.get_instance.return_value = mock_config

        loader = GitLoader(repo_url=None, local_path=Path("/tmp/nonexistent_path"))

        with pytest.raises(ValueError, match="no repo_url"):
            loader.clone_or_pull()

    @patch("codebase_rag.data_ingestion.git_loader.Config")
    def test_clone_or_pull_existing_repo_no_remote(self, mock_config_cls: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.repo_urls = []
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "test-repo"
            # Initialize a bare git repo
            repo = git.Repo.init(local_path)
            repo.git.config("user.email", "test@test.com")
            repo.git.config("user.name", "Test")
            (local_path / "README.md").write_text("hello")
            repo.index.add(["README.md"])
            repo.index.commit("init")

            loader = GitLoader(repo_url="https://example.com/repo.git", local_path=local_path)
            result = loader.clone_or_pull()
            assert result is not None

    @patch("codebase_rag.data_ingestion.git_loader.Config")
    def test_clone_or_pull_local_only(self, mock_config_cls: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.repo_urls = []
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local-repo"
            repo = git.Repo.init(local_path)
            repo.git.config("user.email", "test@test.com")
            repo.git.config("user.name", "Test")
            (local_path / "README.md").write_text("hello")
            repo.index.add(["README.md"])
            repo.index.commit("init")

            loader = GitLoader(repo_url=None, local_path=local_path)
            result = loader.clone_or_pull()
            assert result is not None

    @patch("codebase_rag.data_ingestion.git_loader.Config")
    def test_get_file_paths(self, mock_config_cls: MagicMock) -> None:

        mock_config = MagicMock()
        mock_config.repo_urls = []
        mock_config.repo_local_path = Path("/tmp/repos")
        mock_config_cls.get_instance.return_value = mock_config

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "test-repo"
            repo = git.Repo.init(local_path)
            repo.git.config("user.email", "test@test.com")
            repo.git.config("user.name", "Test")

            (local_path / "README.md").write_text("readme")
            src_dir = local_path / "src"
            src_dir.mkdir()
            (src_dir / "main.py").write_text("print('hello')")
            repo.index.add(["README.md", "src/main.py"])
            repo.index.commit("init")

            loader = GitLoader(repo_url=None, local_path=local_path)
            loader.clone_or_pull()

            paths = loader.get_file_paths(
                included_dirs=["src"],
                included_files=["README.md"],
            )
            filenames = [p.name for p in paths]
            assert "README.md" in filenames
            assert "main.py" in filenames
