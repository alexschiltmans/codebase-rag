"""Process-wide resources for the HTTP API: one Qdrant client, one LLM
client, the configured retriever, and the ingest job manager. Analogous to
`app/runtime.py`'s `AppRuntime`, but without any Streamlit dependency since
this runs as its own uvicorn process.
"""

from __future__ import annotations

from codebase_rag.api.ingest_manager import ApiIngestionManager
from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.provider_factory import create_llm_client
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.retrieval_stack import apply_stages, close_stages, select_base_retriever
from codebase_rag.retrieval.retriever_protocol import RetrieverProtocol
from codebase_rag.retrieval.vector_search import VectorRetriever, resolve_score_threshold
from codebase_rag.services.token_estimator import get_tokenizer


class ApiState:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.get_instance()

        self.qdrant_store = QdrantStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            collection_name=self.config.collection_name,
        )
        self.vector_retriever = VectorRetriever(
            self.qdrant_store, score_threshold=resolve_score_threshold(self.config.embedding_model)
        )
        self.bm25_retriever = self._load_bm25_retriever()
        self.llm = create_llm_client(
            model_name=self.config.llm_model_name,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            timeout=120,
            num_ctx=self.config.ollama_num_ctx,
        )
        # Selected after the LLM exists: the optional rewrite stage needs it.
        self.retriever: RetrieverProtocol = self._select_retriever()
        self.tokenizer = get_tokenizer(self.qdrant_store.embedding_manager)
        self.cache_dir = self.config.cache_dir
        self.ingestion = ApiIngestionManager(on_success=lambda _job: self.refresh_bm25())

    def _select_retriever(self) -> RetrieverProtocol:
        """Resolve the configured retriever and wrap it in the enabled stages.

        The fused retriever is built only when it is the configured one. It used to be
        constructed on every startup and then usually discarded, which cost nothing but
        implied the API had a second standing retriever it could switch to at runtime;
        it never could, because the setting is read once here.
        """
        base = select_base_retriever(self.config, self.bm25_retriever, lambda: self.vector_retriever)
        return apply_stages(base, self.config, self.llm)

    def _load_bm25_retriever(self) -> BM25Retriever:
        bm25_file = self.config.cache_dir / "bm25_retriever.json"
        if bm25_file.exists():
            return BM25Retriever.load_json(bm25_file)
        return BM25Retriever([])

    def new_rag_chain(self) -> RAGChain:
        """Build a fresh RAG chain per request: conversation memory doesn't
        need to persist across stateless HTTP calls."""
        return RAGChain(
            retriever=self.retriever,
            llm=self.llm,
            use_conversation_memory=False,
            prompt_budget_chars=self.llm.prompt_budget_chars,
        )

    def refresh_bm25(self) -> None:
        """Reload the BM25 index from disk after an ingest completes.

        Re-resolving the retriever is what rewires the fused path onto the new index
        under `RETRIEVER=hybrid`; the previous version reached into the standing
        `HybridRetriever` and reassigned its `bm25_retriever` by hand.
        """
        self.bm25_retriever = self._load_bm25_retriever()
        # Release the previous stack's stage resources (e.g. the rewrite thread pool)
        # before rebuilding, so nothing is leaked per ingest.
        close_stages(getattr(self, "retriever", None))
        self.retriever = self._select_retriever()
