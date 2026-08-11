"""Ingestion pipeline for loading repository content into the vector database.

Handles the full data ingestion workflow:
1. Clone or update the repository/repositories
2. Process and chunk the documents
3. Create embeddings and store them in Qdrant
4. Initialize BM25 index for hybrid search
"""

import hashlib
import json
import logging
import pickle
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from codebase_rag.config import Config
from codebase_rag.data_ingestion.chunking import DocumentChunker, chunking_fingerprint
from codebase_rag.data_ingestion.corpus_chunking import write_chunking_sidecar
from codebase_rag.data_ingestion.document_processor import DocumentProcessor
from codebase_rag.data_ingestion.git_loader import GitLoader
from codebase_rag.data_ingestion.truncation import format_truncation_report, measure_truncation
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.retrieval.bm25_search import BM25Retriever, rebuild_bm25_index
from codebase_rag.retrieval.hybrid_search import HybridRetriever
from codebase_rag.retrieval.vector_search import VectorRetriever
from codebase_rag.services import repo_service

# Top-level directory names skipped during auto-discovery (see
# discover_included_dirs). These are dependency/build/cache directories
# that are large, low-signal for retrieval, and never meant to be
# hand-authored — picking a folder that contains one by accident (e.g. a
# JS project's node_modules) shouldn't silently embed it.
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        "node_modules",
        "venv",
        ".venv",
        "env",
        "dist",
        "build",
        "target",
        "__pycache__",
        "vendor",
        "bin",
        "obj",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        "egg-info",
    }
)


def discover_included_dirs(local_path: Path, fallback: list[str]) -> list[str]:
    """Auto-discover top-level directories to scan for a repo.

    Returns every non-hidden top-level directory on disk except those in
    ``DEFAULT_EXCLUDED_DIRS``, or ``fallback`` if ``local_path`` doesn't
    exist yet.
    """
    if not local_path.is_dir():
        return fallback
    return [
        d.name
        for d in local_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in DEFAULT_EXCLUDED_DIRS
    ]


def count_ingestible_files(local_path: Path) -> tuple[list[str], int]:
    """Preview how many files a local-folder ingest would pick up.

    Runs the same directory discovery and file collection the pipeline
    itself uses, without cloning or chunking anything, so the UI can show
    a "N files found" confirmation before a background ingest starts.
    """
    included_dirs = discover_included_dirs(local_path, ["docs", "src", "tests"])
    loader = GitLoader(repo_url=None, local_path=local_path)
    file_paths = loader.get_file_paths(included_dirs=included_dirs, included_files=["README.md", "pyproject.toml"])
    return included_dirs, len(file_paths)


_INGEST_HANDLER_NAMES = ("codebase_rag.ingest_file", "codebase_rag.ingest_console")

# The level the shared "codebase_rag" logger had before the most recent
# setup_logging() call, restored by _teardown_logging() when the run ends.
# A module global rather than a return value because the logger is a
# process-wide singleton and IngestPipeline.run() is the only place that
# needs to read it back, well after setup_logging() has returned.
_prior_level: int | None = None


def setup_logging(log_level: str = "INFO", add_console: bool | None = None) -> tuple[logging.Logger, Path]:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        add_console: Whether to add a console handler. If None (default), adds one only
            if the root logger has no handlers (fresh CLI process). Set explicitly
            in tests to control console output.

    Returns:
        Tuple of configured logger and log file path.
    """
    global _prior_level

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    now = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    microseconds = int((now % 1) * 1_000_000)
    log_file = logs_dir / f"ingest-{timestamp}-{microseconds:06d}.log"

    suffix = 0
    while log_file.exists() and suffix < 1000:
        suffix += 1
        log_file = logs_dir / f"ingest-{timestamp}-{microseconds:06d}-{suffix}.log"

    logger = logging.getLogger("codebase_rag")

    for handler in logger.handlers[:]:
        if handler.name in _INGEST_HANDLER_NAMES:
            handler.close()
            logger.removeHandler(handler)

    if _prior_level is None:
        _prior_level = logger.level
    logger.setLevel(numeric_level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(format_string))
    file_handler.name = "codebase_rag.ingest_file"
    logger.addHandler(file_handler)

    if add_console is None:
        add_console = not logging.getLogger().hasHandlers()

    if add_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format_string))
        stream_handler.name = "codebase_rag.ingest_console"
        logger.addHandler(stream_handler)

    logger.info("Logging initialized at level %s, writing to %s", log_level, log_file)
    return logger, log_file


def _teardown_logging(logger: logging.Logger) -> None:
    """Detach this run's ingest handlers and restore the logger's prior level.

    Called when a pipeline run finishes, success or failure. Without this,
    the file handler (and the level it forced) stay attached to the shared
    "codebase_rag" logger until the *next* setup_logging() call, so every
    other package logger's records in between land in this run's file and
    its file descriptor is held open indefinitely.
    """
    global _prior_level

    for handler in logger.handlers[:]:
        if handler.name in _INGEST_HANDLER_NAMES:
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(_prior_level if _prior_level is not None else logging.NOTSET)
    _prior_level = None


def save_documents_cache(documents: list[Document], cache_path: Path) -> None:
    """Save processed documents to disk cache.

    Args:
        documents: List of processed documents.
        cache_path: Path to save the cache.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents_cache(cache_path: Path) -> list[Document] | None:
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


@dataclass
class IncrementalIngestResult:
    repo_name: str
    files_changed: int
    files_deleted: int
    files_unchanged: int
    chunks_indexed: int
    head_sha: str | None


class IngestCancelled(Exception):  # noqa: N818 - name fixed by the ingestion-progress spec
    """Raised when a pipeline run is stopped via a cancel event."""


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
        progress_callback: Callable[[str, int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            included_dirs: List of directories to include. If omitted, all
                top-level, non-hidden directories of the ingested repo are
                discovered automatically (both for local folders and cloned
                GitHub repos), instead of being limited to docs/src/tests.
            included_files: List of specific files to include.
            drop_existing: Whether to drop existing collections.
            use_cache: Whether to use document cache.
            debug: Whether to enable debug mode.
            repo_url: Single GitHub repository URL to ingest.
            repo_urls: List of GitHub repository URLs to ingest.
            progress_callback: Optional hook called as ``(phase, current, total)``
                from the file-processing loop and the indexing batch loop. Unused
                by the CLI; the UI wires it to per-job progress state.
            cancel_event: Optional event checked at the same two points; raises
                ``IngestCancelled`` when set. Unused by the CLI.
        """
        log_level = "DEBUG" if debug else "INFO"
        self.logger, self.log_file_path = setup_logging(log_level)

        # A failure below (bad LLM_PROVIDER, unreachable Qdrant) would otherwise never reach run()'s finally.
        try:
            self.config = Config.get_instance()
            self._explicit_included_dirs = included_dirs
            self.included_dirs = included_dirs or ["docs", "src", "tests"]
            self.included_files = included_files or ["README.md", "pyproject.toml"]
            self.drop_existing = drop_existing
            self.use_cache = use_cache
            self.progress_callback = progress_callback
            self.cancel_event = cancel_event

            self._repo_urls: list[str] = []
            if repo_urls:
                self._repo_urls = list(repo_urls)
            elif repo_url:
                self._repo_urls = [repo_url]

            self.cache_dir = self.config.cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.vector_store = QdrantStore(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                collection_name=self.config.collection_name,
                embedding_model=self.config.embedding_model,
                recreate_collection=drop_existing,
            )

            self.stats: dict[str, int | float] = {
                "processed_files": 0,
                "chunks_created": 0,
                "chunks_indexed": 0,
                "elapsed_time": 0.0,
            }
        except Exception:
            _teardown_logging(self.logger)
            raise

        # HEAD SHA per repo processed this run, recorded so run() can persist
        # freshness metadata for GET /repos after indexing succeeds.
        self._ingest_head_shas: dict[str, str | None] = {}

    def _chunking_fingerprint(self, chunker: DocumentChunker) -> str:
        """Identify this run's chunking, so caches can tell it apart from an older one."""
        return chunking_fingerprint(chunker, self.vector_store.embedding_manager.model_name)

    def _build_chunker(self) -> DocumentChunker:
        """Build a chunker sized for the model this run will embed with.

        The token window is read off the loaded embedding model rather than
        assumed, so swapping EMBEDDING_MODEL resizes the chunks with it instead
        of leaving them cut for whatever model was configured when the size was
        last chosen by hand.
        """
        return DocumentChunker(max_seq_length=self.vector_store.embedding_manager.max_seq_length)

    def _report_truncation(self, documents: list[Document]) -> None:
        """Log how much of what is about to be indexed the model cannot read.

        Diagnostic only, so a failure here is reported and stepped over. An
        ingest that embedded everything correctly must not be lost because the
        measurement of it fell over.
        """
        manager = self.vector_store.embedding_manager
        start_time = time.time()
        try:
            report = measure_truncation(documents, manager.count_tokens, manager.max_seq_length)
        except Exception as e:
            self.logger.warning("Truncation check failed, indexing anyway: %s", e)
            return
        for line in format_truncation_report(report):
            self.logger.info(line)
        self.logger.debug("Truncation check took %.2f seconds", time.time() - start_time)

    def _repo_name_from_url(self, url: str) -> str:
        """Derive a short repo name from a URL."""
        return url.rstrip("/").split("/")[-1].removesuffix(".git")

    def _cache_path_for_repo(self, repo_name: str) -> Path:
        """Return the document cache path for a specific repo."""
        return self.cache_dir / f"processed_documents_{repo_name}.pkl"

    def _cache_chunking_path_for_repo(self, repo_name: str) -> Path:
        """Return the sidecar recording how a repo's cached documents were chunked."""
        return self.cache_dir / f"processed_documents_{repo_name}_chunking.json"

    def _cached_chunking(self, repo_name: str) -> str | None:
        """Return the chunking fingerprint a repo's document cache was built with.

        None for a cache written before the fingerprint existed, which is
        treated as a mismatch: chunks of unknown sizing are exactly what must
        not be reused.
        """
        path = self._cache_chunking_path_for_repo(repo_name)
        if not path.exists():
            return None
        try:
            fingerprint = json.loads(path.read_text()).get("chunking")
        except (json.JSONDecodeError, OSError, AttributeError):
            return None
        return str(fingerprint) if fingerprint else None

    def _get_head_sha(self, git_loader: GitLoader) -> str | None:
        """Return the HEAD commit SHA from a GitLoader's repo, or None."""
        if git_loader.repo is None:
            return None
        try:
            return str(git_loader.repo.head.commit.hexsha)
        except Exception:
            return None

    def _is_cache_fresh(self, repo_name: str, head_sha: str | None) -> bool:
        """Check whether the document cache matches the current HEAD SHA and chunking.

        Reads `{repo}_freshness.json`: `run()` writes both the pickle and
        freshness together, and `process_repo_incremental` deletes the pickle
        whenever it re-embeds or deletes anything, so the two always describe
        the same *content*. They can describe different commits: a no-op
        incremental ingest at a newer HEAD advances freshness without touching
        the pickle, which is safe precisely because nothing changed. Without
        this check, a second full `run()` after further upstream commits would
        treat a stale pickle as fresh and silently re-index old content.

        Matching content is not enough on its own. The cached documents are
        chunks, not files, so a pickle written under a different chunk size or
        a different embedding model describes the same commit cut a different
        way, and re-indexing it would put chunks in the collection that no
        current run would produce.
        """
        if head_sha is None:
            return False
        if not self._cache_path_for_repo(repo_name).exists():
            return False
        _, cached_sha = repo_service.read_freshness(self.cache_dir, repo_name)
        if cached_sha != head_sha:
            return False

        cached_chunking = self._cached_chunking(repo_name)
        current_chunking = self._chunking_fingerprint(self._build_chunker())
        if cached_chunking != current_chunking:
            self.logger.info(
                "Document cache for %s was chunked as %s, this run chunks as %s, reprocessing",
                repo_name,
                cached_chunking or "unrecorded",
                current_chunking,
            )
            return False
        return True

    def _process_single_repo(self, repo_url: str) -> list[Document]:
        """Process documents from a single repository.

        Args:
            repo_url: GitHub repository URL or local folder path.

        Returns:
            List of processed document chunks.
        """
        repo_name, local_path, git_loader, _is_local_folder = self._resolve_repo_source(repo_url)
        cache_path = self._cache_path_for_repo(repo_name)

        # Always clone/pull first so we can compare HEAD against the cache, and
        # so local_path exists on disk for directory discovery below.
        git_loader.clone_or_pull()
        head_sha = self._get_head_sha(git_loader)
        self._ingest_head_shas[repo_name] = head_sha

        cached = self._try_load_cache(repo_name, cache_path, head_sha)
        if cached is not None:
            return cached

        self.logger.info("Processing repo: %s (local path: %s)", repo_url, local_path)

        included_dirs = self._discover_included_dirs(local_path)

        chunker = self._build_chunker()
        document_processor = DocumentProcessor(git_loader=git_loader, document_chunker=chunker)
        start_time = time.time()
        documents = document_processor.process(
            included_dirs=included_dirs,
            included_files=self.included_files,
            progress_callback=self.progress_callback,
            cancel_event=self.cancel_event,
        )
        processing_time = time.time() - start_time
        self.logger.info(
            "Processed %d chunks from %s in %.2f seconds (scanned dirs: %s)",
            len(documents),
            repo_name,
            processing_time,
            included_dirs,
        )

        # Tag every chunk with the repo name so list_repos() can find them
        for doc in documents:
            doc.metadata["repo"] = repo_name

        if self.use_cache:
            save_documents_cache(documents, cache_path)
            self._cache_chunking_path_for_repo(repo_name).write_text(
                json.dumps({"chunking": self._chunking_fingerprint(chunker)}, indent=2)
            )

        return documents

    def _discover_included_dirs(self, local_path: Path) -> list[str]:
        """Determine which top-level directories to scan for a repo.

        If the caller explicitly requested a fixed set of directories, honor
        it. Otherwise auto-discover every non-hidden top-level directory on
        disk (skipping common dependency/build directories, see
        ``DEFAULT_EXCLUDED_DIRS``), so cloned GitHub repos get the same
        treatment as local-folder ingestion instead of being limited to
        docs/src/tests (which yields a near-empty index for repos that
        don't follow that layout).
        """
        if self._explicit_included_dirs is not None:
            return self._explicit_included_dirs
        return discover_included_dirs(local_path, self.included_dirs)

    def _resolve_repo_source(self, repo_url: str) -> tuple[str, Path, GitLoader, bool]:
        """Determine repo name, local path, and GitLoader for a source."""
        source_path = Path(repo_url).resolve()
        if source_path.is_dir():
            return source_path.name, source_path, GitLoader(repo_url=None, local_path=source_path), True
        repo_name = self._repo_name_from_url(repo_url)
        local_path = self.config.repo_local_path / repo_name
        return repo_name, local_path, GitLoader(repo_url=repo_url, local_path=local_path), False

    def _try_load_cache(self, repo_name: str, cache_path: Path, head_sha: str | None) -> list[Document] | None:
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

    def _file_hashes_path_for_repo(self, repo_name: str) -> Path:
        """Return the per-file content-hash manifest path for a specific repo."""
        return self.cache_dir / f"{repo_name}_file_hashes.json"

    def _read_manifest(self, manifest_path: Path) -> tuple[dict[str, str], str | None]:
        """Read a per-repo manifest as (file hashes, chunking fingerprint).

        Manifests written before chunking was recorded are a bare
        ``{path: hash}`` mapping. Those get a fingerprint of None, which never
        matches, so the first run after an upgrade re-chunks the repo rather
        than trusting chunks whose sizing is unknown.
        """
        if not manifest_path.exists():
            return {}, None
        try:
            data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}, None

        # A manifest that parsed but isn't a mapping is as unusable as one that didn't parse.
        if not isinstance(data, dict):
            return {}, None
        if isinstance(data.get("files"), dict):
            return data["files"], data.get("chunking")
        return data, None

    def _bm25_corpus_path_for_repo(self, repo_name: str) -> Path:
        return self.cache_dir / "bm25_corpus" / f"{repo_name}.json"

    def process_repo_incremental(self, source: str) -> IncrementalIngestResult:
        """Ingest a single source (git URL or local path), re-embedding only
        files whose content changed since the last ingest.

        Diffs by content hash against a per-repo manifest. Unchanged files are
        left untouched in both Qdrant and the BM25 corpus; changed or deleted
        files have their existing chunks removed (by source path) before any
        new chunks for changed files are indexed.
        """
        repo_name, local_path, git_loader, _is_local_folder = self._resolve_repo_source(source)
        git_loader.clone_or_pull()
        head_sha = self._get_head_sha(git_loader)

        included_dirs = self._discover_included_dirs(local_path)
        file_paths = git_loader.get_file_paths(included_dirs=included_dirs, included_files=self.included_files)

        current_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in file_paths}

        manifest_path = self._file_hashes_path_for_repo(repo_name)
        previous_hashes, previous_chunking = self._read_manifest(manifest_path)

        chunker = self._build_chunker()
        chunking = self._chunking_fingerprint(chunker)

        if previous_chunking != chunking:
            # The unchanged files are only safe to skip while the chunking that
            # produced their indexed chunks still matches. It doesn't, so every
            # file is stale regardless of its content hash.
            self.logger.info(
                "Chunking changed for %s (%s -> %s), re-chunking every file",
                repo_name,
                previous_chunking or "unrecorded",
                chunking,
            )
            changed_paths = list(file_paths)
        else:
            changed_paths = [p for p in file_paths if current_hashes[str(p)] != previous_hashes.get(str(p))]

        deleted_sources = [path for path in previous_hashes if path not in current_hashes]
        unchanged_count = len(file_paths) - len(changed_paths)
        new_documents = []
        for path in changed_paths:
            docs = chunker.process_file(path)
            for doc in docs:
                doc.metadata["repo"] = repo_name
            new_documents.extend(docs)

        self._report_truncation(new_documents)

        stale_sources = [str(p) for p in changed_paths] + deleted_sources
        for source_path in stale_sources:
            self.vector_store.delete_by_source(source_path)
        if new_documents:
            self.vector_store.add_documents(new_documents)

        self._update_bm25_corpus_incremental(repo_name, stale_sources, new_documents)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"chunking": chunking, "files": current_hashes}, indent=2))

        if changed_paths or deleted_sources:
            # A later run() must not load a document-cache pickle built from an
            # older commit than the one this incremental ingest just wrote.
            self._cache_path_for_repo(repo_name).unlink(missing_ok=True)
            self._cache_chunking_path_for_repo(repo_name).unlink(missing_ok=True)

        repo_service.save_freshness(self.cache_dir, repo_name, head_sha)

        return IncrementalIngestResult(
            repo_name=repo_name,
            files_changed=len(changed_paths),
            files_deleted=len(deleted_sources),
            files_unchanged=unchanged_count,
            chunks_indexed=len(new_documents),
            head_sha=head_sha,
        )

    def _update_bm25_corpus_incremental(
        self, repo_name: str, stale_sources: list[str], new_documents: list[Document]
    ) -> None:
        """Update this repo's BM25 corpus in place: drop chunks belonging to
        `stale_sources`, add `new_documents`, then rebuild the combined index.
        """
        if not stale_sources and not new_documents:
            return

        corpus_path = self._bm25_corpus_path_for_repo(repo_name)
        existing_docs = BM25Retriever.load_json(corpus_path).documents if corpus_path.exists() else []

        stale_set = set(stale_sources)
        kept_docs = [doc for doc in existing_docs if str(doc.metadata.get("source", "")) not in stale_set]
        kept_docs.extend(new_documents)

        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        BM25Retriever(kept_docs).save_json(corpus_path)

        # The incremental path refuses to run when the chunking has moved, so this records the
        # chunking rather than changing it. It is here so a corpus that has only ever been updated
        # incrementally still ends up describing itself.
        chunker = self._build_chunker()
        write_chunking_sidecar(
            corpus_path.parent,
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            max_seq_length=chunker.max_seq_length,
        )
        rebuild_bm25_index(self.cache_dir)

    def process_documents(self) -> list[Document]:
        """Process documents from all configured repositories.

        Returns:
            List: All processed documents across repos.
        """
        if not self._repo_urls:
            raise ValueError(
                "No repository URLs provided. Use --repo or --all-repos to specify repositories to ingest."
            )

        all_documents: list[Document] = []
        for url in self._repo_urls:
            docs = self._process_single_repo(url)
            all_documents.extend(docs)
            self.logger.info("Repo %s yielded %d chunks", self._repo_name_from_url(url), len(docs))

        self.stats["chunks_created"] = len(all_documents)
        return all_documents

    def index_documents(self, documents: list[Document]) -> None:
        """Index documents in the vector database.

        Args:
            documents: List of processed documents.
        """
        self.logger.info("Indexing documents in Qdrant...")

        # Checked here, not just in the batch loop below: without this, a cancel that lands
        # between the end of processing and the start of indexing would still let delete_by_repo
        # run, wiping a repo's existing chunks with nothing indexed to replace them.
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise IngestCancelled("Ingestion cancelled before indexing")

        self._report_truncation(documents)

        # Remove ALL existing chunks for repos being re-ingested so that
        # deleted or shrunk files don't leave orphaned points.
        repos = {str(doc.metadata["repo"]) for doc in documents if doc.metadata.get("repo")}
        for repo_name in repos:
            deleted = self.vector_store.delete_by_repo(repo_name)
            if deleted:
                self.logger.info("Cleared %d stale chunks for repo '%s'", deleted, repo_name)

        # Index documents in batches to show progress
        start_time = time.time()
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise IngestCancelled("Ingestion cancelled during indexing")

            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1

            display_progress(batch_num, total_batches, "Indexing: ")
            if self.progress_callback is not None:
                self.progress_callback("indexing", batch_num, total_batches)

            self.vector_store.add_documents(batch)

        indexing_time = time.time() - start_time
        self.stats["chunks_indexed"] = len(documents)
        self.stats["elapsed_time"] += indexing_time

        self.logger.info("Indexed %d chunks in %.2f seconds", len(documents), indexing_time)

    def save_bm25_index(self, documents: list[Document]) -> None:
        """Update this run's repo(s) in the BM25 corpus and rebuild the combined index.

        Each repo's documents are persisted as their own JSON corpus file under
        `data/cache/bm25_corpus/`, so accumulating repos never overwrite each
        other. The combined index is then rebuilt from every corpus file on
        disk (not just the documents from this run) and persisted as JSON.

        Args:
            documents: List of processed documents from this ingest run.
        """
        self.logger.info("Updating BM25 corpus...")
        start_time = time.time()

        corpus_dir = self.cache_dir / "bm25_corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        repos = {str(doc.metadata["repo"]) for doc in documents if doc.metadata.get("repo")}
        for repo_name in repos:
            repo_docs = [doc for doc in documents if doc.metadata.get("repo") == repo_name]
            BM25Retriever(repo_docs).save_json(corpus_dir / f"{repo_name}.json")

        # Written every run rather than once, because a re-ingest under a different embedding model
        # re-cuts the corpus and the sidecar has to follow it. Without this the application's own
        # corpus is the one corpus nothing can name a chunk size for, which is also the one every
        # default benchmark invocation scores against.
        chunker = self._build_chunker()
        write_chunking_sidecar(
            corpus_dir,
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            max_seq_length=chunker.max_seq_length,
        )

        bm25_retriever = rebuild_bm25_index(self.cache_dir)

        bm25_time = time.time() - start_time
        self.stats["elapsed_time"] += bm25_time

        self.logger.info(
            "Rebuilt BM25 index with %d documents across %d repo(s) in %.2f seconds",
            len(bm25_retriever.documents),
            len(repos),
            bm25_time,
        )

    def verify_hybrid_search(self, query: str = "How to use this codebase?") -> None:
        """Verify that hybrid search is working correctly.

        Args:
            query: Test query to use for verification.
        """
        self.logger.info("Verifying hybrid search...")

        try:
            bm25_cache_path = self.cache_dir / "bm25_retriever.json"
            bm25_retriever = BM25Retriever.load_json(bm25_cache_path)

            vector_retriever = VectorRetriever(self.vector_store)

            hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
            )

            results = hybrid_retriever.search(query, k=3)

            if results:
                self.logger.info("Hybrid search successful! Found %d results for query: '%s'", len(results), query)
                for i, (doc, score) in enumerate(results, 1):
                    source = doc.metadata.get("source", "Unknown")
                    self.logger.info("Result %d: %s (score: %.4f)", i, source, score)
            else:
                self.logger.warning("Hybrid search returned no results for query: '%s'", query)

        except Exception as e:
            self.logger.error("Error verifying hybrid search: %s", e)

    def save_stats(self) -> None:
        """Save ingestion statistics to file."""
        stats_path = self.cache_dir / "ingest_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        self.logger.info("Statistics saved to %s", stats_path)
        self.logger.info("Summary: %s", self.stats)

    def run(self) -> None:
        """Run the complete ingestion pipeline."""
        self.logger.info("Starting ingestion pipeline...")
        total_start_time = time.time()

        try:
            documents = self.process_documents()
            self.index_documents(documents)
            self.save_bm25_index(documents)
            for repo_name, head_sha in self._ingest_head_shas.items():
                repo_service.save_freshness(self.cache_dir, repo_name, head_sha)
            self.verify_hybrid_search()
            self.stats["elapsed_time"] = time.time() - total_start_time
            self.save_stats()

            self.logger.info("Ingestion pipeline completed successfully in %.2f seconds", self.stats["elapsed_time"])

        except Exception as e:
            self.logger.exception("Error in ingestion pipeline: %s", e)
            raise
        finally:
            _teardown_logging(self.logger)
