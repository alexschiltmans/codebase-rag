"""BM25 keyword search implementation."""

import json
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _doc_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def _dict_to_doc(data: dict) -> Document:
    return Document(page_content=data["page_content"], metadata=data.get("metadata", {}))


class BM25Retriever:
    """BM25 keyword-based retriever.

    This class implements a keyword-based search using the BM25 algorithm,
    which is effective for finding documents containing specific terms.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the BM25 retriever with documents.

        Args:
            documents: List of documents to index.
        """
        self.documents = documents
        self._initialize_index()

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

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Search for documents matching the query.

        Documents scoring exactly 0 (no query term appears in them at all)
        are excluded rather than padded in to reach `k`: a 0 score is not
        evidence of relevance, and returning it as a "match" would make
        every search look non-empty regardless of the query.

        Args:
            query: Search query.
            k: Number of results to return.

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

        scores = self.bm25.get_scores(query_tokens)

        matches = [(doc, score) for doc, score in zip(self.documents, scores, strict=False) if score > 0]
        results = sorted(matches, key=lambda x: x[1], reverse=True)[:k]

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
    def load_json(cls, path: Path) -> "BM25Retriever":
        """Load a BM25 retriever from a JSON file of documents, rebuilding the index.

        Args:
            path: File previously written by `save_json`.
        """
        with open(path) as f:
            data = json.load(f)
        return cls([_dict_to_doc(d) for d in data])


def load_bm25_corpus(corpus_dir: Path) -> list[Document]:
    """Load and merge all per-repo BM25 corpora from a directory.

    Args:
        corpus_dir: Directory containing one JSON file per repo.

    Returns:
        The combined list of documents across all repos.
    """
    if not corpus_dir.exists():
        return []
    documents: list[Document] = []
    for corpus_path in sorted(corpus_dir.glob("*.json")):
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
