# Architecture Decision Records

Each ADR documents a significant design choice: the context that drove it, the alternatives considered, and the rationale for the decision taken.

| ADR | Summary |
|-----|---------|
| [Chunking Strategy](adr-chunking-strategy.md) | Chunk size derived from the embedding model's token window (614 chars at 20% overlap for the shipped model); language-specific splitting for Python (class/function boundaries) and Markdown (header-based). |
| [Why Qdrant](adr-qdrant.md) | Replaced Milvus + MinIO + etcd (3 containers, >2 GB RAM, fragile startup) with Qdrant (single container, ~200 MB) and SQLite for chat storage. |
| [Why LangChain](adr-langchain.md) | Used for document loading, code-aware text splitting, and the Ollama client. Higher-level abstractions (chains, retrievers) are bypassed in favour of direct control. |
| [Why Langfuse](adr-langfuse.md) | Optional tracing layer that records retrieval results, prompt text, and generation output per query, making it possible to pinpoint whether failures are in retrieval, prompting, or generation. |
| [Chat Storage](chat_storage.md) | SQLite-backed chat history with no extra infrastructure; messages serialised as JSON in a single `chats` table. |
