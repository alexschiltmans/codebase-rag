"""Configuration management for the Codebase RAG application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from dotenv import load_dotenv


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

    # Vector database settings (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # Chat storage settings (SQLite)
    chat_storage_path: Path = Path("./data/chat_history.db")

    # LLM settings
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "sam860/LFM2:350m"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

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

            cls._instance = cls(
                repo_urls=repo_urls,
                repo_local_path=Path(os.getenv("REPO_LOCAL_PATH", str(cls.repo_local_path))),
                qdrant_host=os.getenv("QDRANT_HOST", cls.qdrant_host),
                qdrant_port=int(os.getenv("QDRANT_PORT", str(cls.qdrant_port))),
                collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
                chat_storage_path=Path(os.getenv("CHAT_STORAGE_PATH", str(cls.chat_storage_path))),
                provider=os.getenv("LLM_PROVIDER", cls.provider),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
                llm_model_name=os.getenv("LLM_MODEL_NAME", cls.llm_model_name),
                embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
                default_repo_url=os.getenv("DEFAULT_REPO_URL", cls.default_repo_url),
                log_level=os.getenv("LOG_LEVEL", cls.log_level),
                api_host=os.getenv("API_HOST", cls.api_host),
                api_port=int(os.getenv("API_PORT", str(cls.api_port))),
                langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
                langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", cls.langfuse_public_key),
                langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", cls.langfuse_secret_key),
                langfuse_host=os.getenv("LANGFUSE_HOST", cls.langfuse_host),
            )

            cls._instance.repo_local_path.mkdir(parents=True, exist_ok=True)

        return cls._instance
