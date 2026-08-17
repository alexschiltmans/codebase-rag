"""Compose the optional rerank and rewrite stages around a base retriever.

Single source of truth for stage ordering and enablement, shared by the
Streamlit runtime, the HTTP API, and the eval harness. Without this the
composition was copied into each entry point and would drift the moment a third
stage or an ordering change landed; worse, the eval harness had no composition
at all, so a reranked run measured the bare first stage.

Ordering is deliberate: rewrite is outermost. It expands the query before any
retrieval happens, so it must run before the rerank stage pulls and rescores
candidates. Both stages bind to the `Retriever` protocol, so they compose over
BM25, vector, or hybrid without any of them knowing they are wrapped.
"""

from typing import Any

from codebase_rag.config import Config

from .retriever_protocol import RetrieverProtocol


def apply_stages(retriever: RetrieverProtocol, config: Config, llm: Any) -> RetrieverProtocol:
    """Wrap `retriever` with the rerank and rewrite stages enabled in `config`.

    Returns the base retriever unchanged when neither stage is enabled, so a
    default configuration pays nothing for stages it does not use. The reranker
    model still loads lazily on first `search()`, so even an enabled-but-unused
    stack costs nothing until a query runs through it.

    Args:
        retriever: The base first-stage retriever.
        config: Feature flags and model/depth/timeout settings.
        llm: Local model client for the rewrite stage's expansion call. Only
            used when `rewrite_enabled`.

    Returns:
        The composed retriever, still satisfying `RetrieverProtocol`.
    """
    if config.rerank_enabled:
        from .rerank import RerankingRetriever

        retriever = RerankingRetriever(
            retriever,
            model_name=config.rerank_model,
            candidate_depth=config.rerank_candidate_depth,
        )
    if config.rewrite_enabled:
        from .rewrite import RewritingRetriever

        retriever = RewritingRetriever(retriever, llm, timeout_s=config.rewrite_timeout_s)
    return retriever


def close_stages(retriever: RetrieverProtocol | None) -> None:
    """Release any resources held by the stages wrapping `retriever`.

    Walks the composed chain via each stage's `retriever` attribute and calls
    `close()` on the stages this module added (currently only
    `RewritingRetriever`, which owns a thread pool). Keeping the walk here rather
    than in each caller means stage ordering and which stages hold resources stay
    knowledge this module owns, alongside `apply_stages` that built the chain.
    Safe on `None` and on a bare base retriever with no stages.

    Only stage types are closed, never the base retriever the chain terminates
    in. Closing whatever happens to expose a `close()` would reach the base the
    moment one grows the method, and a base retriever holds the process-wide
    Qdrant client: an ingest would then tear down the connection every open
    session is still using.

    Callers should expect an in-flight reader to survive this. A request holding
    the old stack keeps working, and its rewrite stage falls back to the
    unexpanded query rather than raising, which is the right trade because the
    stack being closed is the one whose index is now stale anyway.
    """
    from .rewrite import RewritingRetriever

    closable = (RewritingRetriever,)

    current: Any = retriever
    seen = 0
    # Bounded walk: the chain is at most base + the stages apply_stages can add,
    # so a small cap guards against a cycle without hardcoding the stage count.
    while current is not None and seen < 8:
        if isinstance(current, closable):
            current.close()
        current = getattr(current, "retriever", None)
        seen += 1
