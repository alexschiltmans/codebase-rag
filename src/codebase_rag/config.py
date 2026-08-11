"""Configuration management for the Codebase RAG application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from dotenv import load_dotenv


def _env(name: str, default: str) -> str:
    """Read an env var, treating an empty value as unset.

    An emptied .env line (LLM_PROVIDER=) sets the variable to "" rather than removing it, and
    os.getenv's own default only applies when the variable is absent. Commenting a value out by
    blanking it is an ordinary edit, so every setting reads through here rather than some of them.
    """
    return os.getenv(name) or default


def _env_int(name: str, default: int) -> int:
    """Read an int env var, treating an empty value as unset.

    Worse than the string case if left unguarded: int("") raises, and because the singleton is
    never assigned when construction fails, that raise repeats on every later get_instance().
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got '{raw}'") from None


def _env_optional_int(name: str, default: int | None) -> int | None:
    """Read an optional int env var, treating an empty value as unset."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got '{raw}'") from None


@dataclass
class Config:
    """Configuration settings for the application.

    Uses the singleton pattern for global access to configuration.
    Repo URLs are configured via environment variables — no defaults are provided.
    """

    # Class variable to store the singleton instance
    _instance: ClassVar[Optional["Config"]] = None

    # Repository settings — configured via REPO_URLS (comma-separated) and REPO_LOCAL_PATH
    repo_urls: list[str] = field(default_factory=list)
    repo_local_path: Path = Path("./data/repos")

    # Local index directory: the BM25 corpus, the combined BM25 index, the document cache, and
    # the freshness sidecars all live under here. Every reader and writer resolves it from this
    # one field so that pointing the app at another directory moves all of them together, and so
    # that the set of paths the process may deserialise from stays enumerable.
    cache_dir: Path = Path("./data/cache")

    # Vector database settings (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # Chat storage settings (SQLite)
    chat_storage_path: Path = Path("./data/chat_history.db")

    # Retriever settings
    retriever: str = "bm25"

    # LLM settings
    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model_name: str = "sam860/LFM2:350m"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    # Hub revision (commit sha or tag) the embedding model resolves to. Empty means the hub's
    # default branch, which is whatever it points at on the day of the run: two runs weeks apart
    # then embed with different weights under the same model name, moving retrieval scores with
    # nothing in the diff to explain it. Set this before publishing any measurement.
    embedding_model_revision: str = ""
    embedding_query_prompt: str = ""
    embedding_document_prompt: str = ""
    embedding_max_seq_length: int | None = None
    embedding_dtype: str = ""
    ollama_num_ctx: int = 8192

    # Default repository for auto-ingestion on first startup
    default_repo_url: str = ""

    # Application settings
    log_level: str = "INFO"

    # HTTP API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Langfuse tracing settings
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    @classmethod
    def get_instance(cls) -> "Config":
        """Get the singleton instance of the Config class.

        Returns:
            Config: The singleton configuration instance.
        """
        if cls._instance is None:
            load_dotenv()

            repo_urls_str = os.getenv("REPO_URLS", "")
            repo_urls = [u.strip() for u in repo_urls_str.split(",") if u.strip()] if repo_urls_str else []

            provider = _env("LLM_PROVIDER", cls.provider)
            if provider not in ("ollama", "openai-compat"):
                raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'openai-compat', got '{provider}'")

            retriever = _env("RETRIEVER", cls.retriever)
            if retriever not in ("bm25", "hybrid"):
                raise ValueError(f"RETRIEVER must be 'bm25' or 'hybrid', got '{retriever}'")

            llm_base_url = _env("LLM_BASE_URL", cls.llm_base_url)
            if provider == "openai-compat" and not llm_base_url:
                # Otherwise this passes here and dies inside OpenAICompatClient.__init__ instead,
                # which for the app means an uncaught exception surfacing as a raw Streamlit
                # traceback rather than the same clear config-time error LLM_PROVIDER gets above.
                raise ValueError("LLM_BASE_URL must be set when LLM_PROVIDER=openai-compat")

            cls._instance = cls(
                repo_urls=repo_urls,
                repo_local_path=Path(_env("REPO_LOCAL_PATH", str(cls.repo_local_path))),
                cache_dir=Path(_env("CACHE_DIR", str(cls.cache_dir))),
                qdrant_host=_env("QDRANT_HOST", cls.qdrant_host),
                qdrant_port=_env_int("QDRANT_PORT", cls.qdrant_port),
                collection_name=_env("COLLECTION_NAME", cls.collection_name),
                chat_storage_path=Path(_env("CHAT_STORAGE_PATH", str(cls.chat_storage_path))),
                retriever=retriever,
                provider=provider,
                ollama_base_url=_env("OLLAMA_BASE_URL", cls.ollama_base_url),
                llm_base_url=llm_base_url,
                llm_api_key=_env("LLM_API_KEY", cls.llm_api_key),
                llm_model_name=_env("LLM_MODEL_NAME", cls.llm_model_name),
                embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
                embedding_model_revision=_env("EMBEDDING_MODEL_REVISION", cls.embedding_model_revision),
                embedding_query_prompt=_env("EMBEDDING_QUERY_PROMPT", cls.embedding_query_prompt),
                embedding_document_prompt=_env("EMBEDDING_DOCUMENT_PROMPT", cls.embedding_document_prompt),
                embedding_max_seq_length=_env_optional_int("EMBEDDING_MAX_SEQ_LENGTH", cls.embedding_max_seq_length),
                embedding_dtype=_env("EMBEDDING_DTYPE", cls.embedding_dtype),
                ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", cls.ollama_num_ctx),
                default_repo_url=_env("DEFAULT_REPO_URL", cls.default_repo_url),
                log_level=_env("LOG_LEVEL", cls.log_level),
                api_host=_env("API_HOST", cls.api_host),
                api_port=_env_int("API_PORT", cls.api_port),
                langfuse_enabled=_env("LANGFUSE_ENABLED", "false").lower() == "true",
                langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY", cls.langfuse_public_key),
                langfuse_secret_key=_env("LANGFUSE_SECRET_KEY", cls.langfuse_secret_key),
                langfuse_host=_env("LANGFUSE_HOST", cls.langfuse_host),
            )

            cls._instance.repo_local_path.mkdir(parents=True, exist_ok=True)

        return cls._instance
