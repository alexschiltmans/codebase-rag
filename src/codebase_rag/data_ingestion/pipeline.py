"""Ingestion pipeline for loading repository content into the vector database.

Handles the full data ingestion workflow:
1. Clone or update the repository/repositories
2. Process and chunk the documents
3. Create embeddings and store them in Qdrant
4. Initialize BM25 index for hybrid search
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

from codebase_rag.config import Config
from codebase_rag.data_ingestion.document_processor import DocumentProcessor
from codebase_rag.data_ingestion.git_loader import GitLoader
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.retrieval.bm25_search import BM25Retriever
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: Configured logger.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"ingest-{timestamp}.log"

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger("codebase_rag.ingest")
    logger.info(f"Logging initialized at level {log_level}, writing to {log_file}")
    return logger


def save_documents_cache(documents: list, cache_path: Path) -> None:
    """Save processed documents to disk cache.

    Args:
        documents: List of processed documents.
        cache_path: Path to save the cache.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents_cache(cache_path: Path) -> list | None:
    """Load processed documents from disk cache.

    Args:
        cache_path: Path to the cache file.

    Returns:
        Optional[List]: List of documents if cache exists, None otherwise.
    """
    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)  # type: ignore[no-any-return]  # noqa: S301


def display_progress(current: int, total: int, prefix: str = "", length: int = 50) -> None:
    """Display a progress bar in the console.

    Args:
        current: Current progress value.
        total: Total value for 100% completion.
        prefix: Prefix string for the progress bar.
        length: Length of the progress bar in characters.
    """
    percent = min(100.0, (current / total) * 100)
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "░" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


class IngestPipeline:
    """Pipeline for ingesting documents from one or more repositories into the vector database.

    Supports single-repo and multi-repo ingestion. When multiple repos are
    provided, documents from all repos are merged into a single Qdrant
    collection and a single BM25 index.
    """

    def __init__(
        self,
        included_dirs: list[str] | None = None,
        included_files: list[str] | None = None,
        drop_existing: bool = False,
        use_cache: bool = True,
        debug: bool = False,
        repo_url: str | None = None,
        repo_urls: list[str] | None = None,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            included_dirs: List of directories to include.
            included_files: List of specific files to include.
            drop_existing: Whether to drop existing collections.
            use_cache: Whether to use document cache.
            debug: Whether to enable debug mode.
            repo_url: Single GitHub repository URL to ingest.
            repo_urls: List of GitHub repository URLs to ingest.
        """
        log_level = "DEBUG" if debug else "INFO"
        self.logger = setup_logging(log_level)

        self.config = Config.get_instance()
        self.included_dirs = included_dirs or ["docs", "src", "tests"]
        self.included_files = included_files or ["README.md", "pyproject.toml"]
        self.drop_existing = drop_existing
        self.use_cache = use_cache

        self._repo_urls: list[str] = []
        if repo_urls:
            self._repo_urls = list(repo_urls)
        elif repo_url:
            self._repo_urls = [repo_url]

        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = QdrantStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            collection_name=self.config.collection_name,
            recreate_collection=drop_existing,
        )

        self.stats: dict[str, int | float] = {
            "processed_files": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "elapsed_time": 0.0,
        }

    def _repo_name_from_url(self, url: str) -> str:
        """Derive a short repo name from a URL."""
        return url.rstrip("/").split("/")[-1].removesuffix(".git")

    def _cache_path_for_repo(self, repo_name: str) -> Path:
        """Return the document cache path for a specific repo."""
        return self.cache_dir / f"processed_documents_{repo_name}.pkl"

    def _cache_meta_path_for_repo(self, repo_name: str) -> Path:
        """Return the cache metadata path for a specific repo."""
        return self.cache_dir / f"{repo_name}_cache_meta.json"

    def _get_head_sha(self, git_loader: GitLoader) -> str | None:
        """Return the HEAD commit SHA from a GitLoader's repo, or None."""
        if git_loader.repo is None:
            return None
        try:
            return str(git_loader.repo.head.commit.hexsha)
        except Exception:
            return None

    def _is_cache_fresh(self, repo_name: str, head_sha: str | None) -> bool:
        """Check whether the document cache matches the current HEAD SHA."""
        if head_sha is None:
            return False
        meta_path = self._cache_meta_path_for_repo(repo_name)
        if not meta_path.exists():
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return bool(meta.get("commit_sha") == head_sha)
        except (json.JSONDecodeError, OSError):
            return False

    def _save_cache_meta(self, repo_name: str, head_sha: str) -> None:
        """Persist cache metadata for a repo."""
        meta_path = self._cache_meta_path_for_repo(repo_name)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({"commit_sha": head_sha, "timestamp": time.time()}, f, indent=2)

    def _process_single_repo(self, repo_url: str) -> list:
        """Process documents from a single repository.

        Args:
            repo_url: GitHub repository URL or local folder path.

        Returns:
            List of processed document chunks.
        """
        repo_name, local_path, git_loader, is_local_folder = self._resolve_repo_source(repo_url)
        cache_path = self._cache_path_for_repo(repo_name)

        # Always clone/pull first so we can compare HEAD against the cache
        git_loader.clone_or_pull()
        head_sha = self._get_head_sha(git_loader)

        cached = self._try_load_cache(repo_name, cache_path, head_sha)
        if cached is not None:
            return cached

        self.logger.info("Processing repo: %s (local path: %s)", repo_url, local_path)

        # For local folders, scan all subdirectories instead of only docs/src/tests
        if is_local_folder:
            included_dirs = [d.name for d in local_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        else:
            included_dirs = self.included_dirs

        document_processor = DocumentProcessor(git_loader=git_loader)
        start_time = time.time()
        documents = document_processor.process(
            included_dirs=included_dirs,
            included_files=self.included_files,
        )
        processing_time = time.time() - start_time
        self.logger.info("Processed %d chunks from %s in %.2f seconds", len(documents), repo_name, processing_time)

        # Tag every chunk with the repo name so list_repos() can find them
        for doc in documents:
            doc.metadata["repo"] = repo_name

        if self.use_cache:
            save_documents_cache(documents, cache_path)
            if head_sha:
                self._save_cache_meta(repo_name, head_sha)

        return documents

    def _resolve_repo_source(self, repo_url: str) -> tuple[str, Path, GitLoader, bool]:
        """Determine repo name, local path, and GitLoader for a source."""
        source_path = Path(repo_url).resolve()
        if source_path.is_dir():
            return source_path.name, source_path, GitLoader(repo_url=None, local_path=source_path), True
        repo_name = self._repo_name_from_url(repo_url)
        local_path = self.config.repo_local_path / repo_name
        return repo_name, local_path, GitLoader(repo_url=repo_url, local_path=local_path), False

    def _try_load_cache(self, repo_name: str, cache_path: Path, head_sha: str | None) -> list | None:
        """Return cached documents if the cache is fresh, otherwise None."""
        if not (self.use_cache and self._is_cache_fresh(repo_name, head_sha)):
            return None
        cached_docs = load_documents_cache(cache_path)
        if not cached_docs:
            return None
        self.logger.info(
            "Cache is fresh for %s (SHA %s), loaded %d documents",
            repo_name,
            head_sha,
            len(cached_docs),
        )
        for doc in cached_docs:
            doc.metadata.setdefault("repo", repo_name)
        return cached_docs

    def process_documents(self) -> list:
        """Process documents from all configured repositories.

        Returns:
            List: All processed documents across repos.
        """
        if not self._repo_urls:
            raise ValueError(
                "No repository URLs provided. Use --repo or --all-repos to specify repositories to ingest."
            )

        all_documents: list = []
        for url in self._repo_urls:
            docs = self._process_single_repo(url)
            all_documents.extend(docs)
            self.logger.info("Repo %s yielded %d chunks", self._repo_name_from_url(url), len(docs))

        self.stats["chunks_created"] = len(all_documents)
        return all_documents

    def index_documents(self, documents: list) -> None:
        """Index documents in the vector database.

        Args:
            documents: List of processed documents.
        """
        self.logger.info("Indexing documents in Qdrant...")

        # Remove ALL existing chunks for repos being re-ingested so that
        # deleted or shrunk files don't leave orphaned points.
        repos = {doc.metadata.get("repo") for doc in documents if doc.metadata.get("repo")}
        for repo_name in repos:
            deleted = self.vector_store.delete_by_repo(repo_name)
            if deleted:
                self.logger.info("Cleared %d stale chunks for repo '%s'", deleted, repo_name)

        # Index documents in batches to show progress
        start_time = time.time()
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1

            display_progress(batch_num, total_batches, "Indexing: ")

            self.vector_store.add_documents(batch)

        indexing_time = time.time() - start_time
        self.stats["chunks_indexed"] = len(documents)
        self.stats["elapsed_time"] += indexing_time

        self.logger.info(f"Indexed {len(documents)} chunks in {indexing_time:.2f} seconds")

    def save_bm25_index(self, documents: list) -> None:
        """Create and save BM25 index for hybrid search.

        Args:
            documents: List of processed documents.
        """
        self.logger.info("Creating BM25 index...")
        start_time = time.time()

        bm25_retriever = BM25Retriever(documents)

        bm25_cache_path = self.cache_dir / "bm25_retriever.pkl"
        with open(bm25_cache_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

        bm25_time = time.time() - start_time
        self.stats["elapsed_time"] += bm25_time

        self.logger.info(f"Created BM25 index in {bm25_time:.2f} seconds")

    def verify_hybrid_search(self, query: str = "How to use this codebase?") -> None:
        """Verify that hybrid search is working correctly.

        Args:
            query: Test query to use for verification.
        """
        self.logger.info("Verifying hybrid search...")

        try:
            bm25_cache_path = self.cache_dir / "bm25_retriever.pkl"
            with open(bm25_cache_path, "rb") as f:
                bm25_retriever = pickle.load(f)  # noqa: S301

            vector_retriever = VectorRetriever(self.vector_store)

            hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
            )

            results = hybrid_retriever.search(query, k=3)

            if results:
                self.logger.info(f"Hybrid search successful! Found {len(results)} results for query: '{query}'")
                for i, (doc, score) in enumerate(results, 1):
                    source = doc.metadata.get("source", "Unknown")
                    self.logger.info(f"Result {i}: {source} (score: {score:.4f})")
            else:
                self.logger.warning(f"Hybrid search returned no results for query: '{query}'")

        except Exception as e:
            self.logger.error(f"Error verifying hybrid search: {e}")

    def save_stats(self) -> None:
        """Save ingestion statistics to file."""
        stats_path = self.cache_dir / "ingest_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        self.logger.info(f"Statistics saved to {stats_path}")
        self.logger.info(f"Summary: {self.stats}")

    def run(self) -> None:
        """Run the complete ingestion pipeline."""
        self.logger.info("Starting ingestion pipeline...")
        total_start_time = time.time()

        try:
            documents = self.process_documents()
            self.index_documents(documents)
            self.save_bm25_index(documents)
            self.verify_hybrid_search()
            self.stats["elapsed_time"] = time.time() - total_start_time
            self.save_stats()

            self.logger.info(f"Ingestion pipeline completed successfully in {self.stats['elapsed_time']:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error in ingestion pipeline: {e}", exc_info=True)
            raise
