"""BM25 keyword search implementation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Number of documents `search` returns when the caller passes no `k`. The
# protocol's `k=None` resolves to this, so the value lives here rather than
# in the signature (where `None` is the declared default) or in the docstring.
DEFAULT_TOP_K = 4


def _doc_to_dict(doc: Document) -> dict[str, Any]:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def _dict_to_doc(data: dict[str, Any]) -> Document:
    return Document(page_content=data["page_content"], metadata=data.get("metadata", {}))


class BM25Retriever:
    """BM25 keyword-based retriever.

    This class implements a keyword-based search using the BM25 algorithm,
    which is effective for finding documents containing specific terms.
    """

    def __init__(self, documents: list[Document], repos: list[str] | None = None) -> None:
        """Initialize the BM25 retriever with documents.

        Args:
            documents: List of documents to index.
            repos: Optional repository names to restrict every search's *results* to.
                None returns matches from everything indexed, which is what the app
                and the API do. Note this narrows what comes back, it does not narrow
                what is scored: BM25 scores depend on corpus-wide document frequencies
                and average document length, so restricting the index instead would
                change the scores of the documents that survive. Callers that want a
                repo scored on its own build the index from that repo's corpus alone
                (see `load_bm25_corpus`).
        """
        self.documents = documents
        self.repos = repos
        self._initialize_index()
        if repos is not None:
            logger.info("Restricted to repositories: %s", ", ".join(repos))

    def _preprocess_text(self, text: str) -> list[str]:
        """Preprocess text for BM25 indexing.

        Args:
            text: Text to preprocess.

        Returns:
            List[str]: List of preprocessed tokens.
        """
        text = text.lower()

        tokens = re.findall(r"\w+", text)

        return [token for token in tokens if len(token) > 1]

    def _initialize_index(self) -> None:
        """Initialize the BM25 index."""
        if not self.documents:
            logger.warning("No documents provided for BM25 indexing. Creating empty index.")
            self.corpus = []
            self.bm25 = None
            return

        self.corpus = [self._preprocess_text(doc.page_content) for doc in self.documents]

        self.bm25 = BM25Okapi(self.corpus)
        logger.info("Initialized BM25 index with %d documents", len(self.documents))

    def _in_scope(self, doc: Document) -> bool:
        """Whether `doc` belongs to one of the repositories this retriever is restricted to."""
        if self.repos is None:
            return True
        return doc.metadata.get("repo") in self.repos

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Search for documents matching the query.

        Documents scoring at or below 0 are excluded rather than padded in
        to reach `k`: such a score is not evidence of relevance, and
        returning it as a "match" would make every search look non-empty
        regardless of the query.

        Two distinct cases score at or below zero, and both should be
        excluded. A document containing none of the query terms scores
        exactly 0. Separately, BM25 assigns a *negative* IDF to any term
        appearing in more than roughly half the corpus (such a term
        carries no discriminative signal), so documents matching only
        those terms can score slightly below 0.

        A repository restriction is applied before the cut to ``k``, so ``k`` is
        ``k`` in-scope results rather than however many of the global top ``k``
        happen to be in scope.

        Args:
            query: Search query.
            k: Number of results to return. ``None`` uses ``DEFAULT_TOP_K``.

        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples.
        """
        if self.bm25 is None or not self.documents:
            logger.warning("No documents in index, returning empty result")
            return []

        query_tokens = self._preprocess_text(query)

        if not query_tokens:
            logger.warning("No valid tokens in query, returning empty result")
            return []

        k_value = k if k is not None else DEFAULT_TOP_K
        scores = self.bm25.get_scores(query_tokens)

        matches = [
            (doc, score)
            for doc, score in zip(self.documents, scores, strict=False)
            if score > 0 and self._in_scope(doc)
        ]
        results = sorted(matches, key=lambda x: x[1], reverse=True)[:k_value]

        logger.info("BM25 search for '%s' returned %d results", query, len(results))
        return results

    def save_json(self, path: Path) -> None:
        """Persist the indexed documents to a JSON file.

        Args:
            path: File to write the documents to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([_doc_to_dict(doc) for doc in self.documents], f)

    @classmethod
    def load_json(cls, path: Path, repos: list[str] | None = None) -> "BM25Retriever":
        """Load a BM25 retriever from a JSON file of documents, rebuilding the index.

        Args:
            path: File previously written by `save_json`.
            repos: Optional repository restriction, as in `__init__`.
        """
        with open(path) as f:
            data = json.load(f)
        return cls([_dict_to_doc(d) for d in data], repos=repos)


def load_bm25_corpus(corpus_dir: Path, repos: list[str] | None = None) -> list[Document]:
    """Load and merge per-repo BM25 corpora from a directory.

    `repos` selects which files are read rather than filtering the merged result: BM25 scores
    depend on corpus-wide document frequencies and average document length, so two repos loaded
    out of three score differently from the same two loaded on their own. A named repo with no
    corpus file is skipped rather than raised on; callers that need a missing repo to stop the
    run check the directory themselves, so that rule lives in one place.

    Args:
        corpus_dir: Directory containing one JSON file per repo.
        repos: Optional repository names to load, order-insensitive and de-duplicated. None
            loads every file present.

    Returns:
        The combined list of documents across the selected repos.
    """
    if not corpus_dir.exists():
        return []
    if repos is None:
        corpus_paths = sorted(corpus_dir.glob("*.json"))
    else:
        # Sorted and de-duplicated: BM25 ties break on insertion order, and a repeated name would
        # otherwise load one repo twice and double every document frequency it contributes.
        named = {corpus_dir / f"{repo}.json" for repo in repos}
        corpus_paths = sorted(path for path in named if path.exists())
    documents: list[Document] = []
    for corpus_path in corpus_paths:
        with open(corpus_path) as f:
            data = json.load(f)
        documents.extend(_dict_to_doc(d) for d in data)
    return documents


def delete_bm25_corpus(corpus_dir: Path, repo_name: str) -> bool:
    """Delete a single repo's BM25 corpus file, if present.

    Args:
        corpus_dir: Directory containing one JSON file per repo.
        repo_name: Name of the repo whose corpus should be removed.

    Returns:
        True if a corpus file was found and removed.
    """
    corpus_path = corpus_dir / f"{repo_name}.json"
    if corpus_path.exists():
        corpus_path.unlink()
        return True
    return False


def rebuild_bm25_index(cache_dir: Path) -> "BM25Retriever":
    """Rebuild the combined BM25 index from all per-repo corpora and persist it.

    Args:
        cache_dir: The `data/cache` directory holding `bm25_corpus/` and the
            combined index file.

    Returns:
        The rebuilt BM25Retriever, covering every repo with a saved corpus.
    """
    documents = load_bm25_corpus(cache_dir / "bm25_corpus")
    retriever = BM25Retriever(documents)
    retriever.save_json(cache_dir / "bm25_retriever.json")
    return retriever
