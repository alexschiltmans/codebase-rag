"""Process-wide resources for the HTTP API: one Qdrant client, one LLM
client, one hybrid retriever, and the ingest job manager. Analogous to
`app/runtime.py`'s `AppRuntime`, but without any Streamlit dependency since
this runs as its own uvicorn process (see design.md: "Separate uvicorn
process, not mounted in Streamlit").
"""

from __future__ import annotations

from pathlib import Path

from codebase_rag.api.ingest_manager import ApiIngestionManager
from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever
from codebase_rag.services.token_estimator import get_tokenizer


class ApiState:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.get_instance()

        self.qdrant_store = QdrantStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            collection_name=self.config.collection_name,
        )
        self.vector_retriever = VectorRetriever(self.qdrant_store)
        self.bm25_retriever = self._load_bm25_retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            vector_weight=0.7,
            bm25_weight=0.3,
            min_score_threshold=0.1,
        )
        self.llm = OllamaClient(
            model_name=self.config.llm_model_name,
            base_url=self.config.ollama_base_url,
            temperature=0.0,
            top_p=0.9,
            top_k=40,
            max_tokens=1024,
            timeout=120,
        )
        self.tokenizer = get_tokenizer(self.qdrant_store.embedding_manager)
        self.cache_dir = Path("data/cache")
        self.ingestion = ApiIngestionManager(on_success=lambda _job: self.refresh_bm25())

    @staticmethod
    def _load_bm25_retriever() -> BM25Retriever:
        cache_dir = Path("data/cache")
        bm25_file = cache_dir / "bm25_retriever.json"
        if bm25_file.exists():
            return BM25Retriever.load_json(bm25_file)
        return BM25Retriever([])

    def new_rag_chain(self) -> RAGChain:
        """Build a fresh RAG chain per request: conversation memory doesn't
        need to persist across stateless HTTP calls."""
        return RAGChain(retriever=self.hybrid_retriever, llm=self.llm, use_conversation_memory=False)

    def refresh_bm25(self) -> None:
        """Reload the BM25 index from disk after an ingest completes."""
        self.bm25_retriever = self._load_bm25_retriever()
        self.hybrid_retriever.bm25_retriever = self.bm25_retriever
