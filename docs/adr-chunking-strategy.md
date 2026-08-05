# ADR: Document Chunking Strategy

## Status

Accepted

## Context

When ingesting codebases for RAG, documents must be split into chunks small enough for embedding models to process effectively, yet large enough to retain meaningful context. The chunking strategy affects retrieval quality: chunks that are too small lose context, chunks that are too large dilute the signal.

## Decision

### Chunk size: derived from the embedding model's token window

- The chunker takes the configured model's `max_seq_length` and multiplies it by a measured characters-per-token ratio (`CHARS_PER_TOKEN`, 1.6). For `all-mpnet-base-v2` at 384 tokens that gives **614 characters**, with a **122-character overlap** at a fixed 20% of size so context at chunk boundaries isn't lost.
- The ratio is deliberately taken from the low tail of what was measured, not the median. Chars-per-token on this corpus has a median near 3.0 but drops to 1.6 for dense Markdown and below that for data formats, and a chunk only has to be dense once to overflow. Sizing at the median would produce chunks that fit on average and truncate wherever the content is packed.
- The ingestion path reports the share of chunks that exceed the model's limit, per file type. Truncation is otherwise invisible: the model silently drops the tail, and the resulting vector says nothing about what was left out.

This replaces a fixed 1000/200, which was chosen against an estimate of 384 tokens ≈ 1500 characters. Measured with the model's own tokenizer, 1000-character chunks put 31% of the corpus over the limit, including 9% of `.cpp` chunks.

### Language-specific splitting

Three strategies are used based on file type:

1. **Python files** (`.py`, `.ipynb`) → `RecursiveCharacterTextSplitter.from_language(language="python")`
   - Splits on Python-specific boundaries: class definitions, function definitions, decorators, blank lines between top-level blocks.
   - Preserves complete logical units (e.g., a full method) whenever possible.
   - Prevents splitting mid-expression or mid-docstring.

2. **Markdown/RST files** (`.md`, `.rst`) → `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
   - First pass: splits by headers (`#`, `##`, `###`, etc.) to capture document hierarchy.
   - Header metadata is preserved in chunk metadata (e.g., `header_1: "API Reference"`, `header_2: "Authentication"`).
   - Second pass: further splits oversized sections using the recursive splitter with markdown-aware separators.
   - This two-pass approach ensures that retrieval results include the section hierarchy, which helps when answering questions about doc structure.

3. **All other files** → `RecursiveCharacterTextSplitter` with default separators
   - Falls back to splitting on `\n\n`, `\n`, ` `, then character-level as a last resort.
   - Suitable for config files, YAML, TOML, plain text, etc.

### Content hashing

Each chunk includes a SHA-256 hash of its content in metadata (`content_hash`). This enables:
- Detecting whether a chunk has changed between ingestion runs.
- Future deduplication or change-tracking optimizations.

### Deterministic chunk IDs

Chunk point IDs in the vector store are deterministic, derived from `source_path + chunk_index` via UUID5. This means:
- Re-ingesting the same file overwrites existing chunks in place (idempotent).
- No duplicate chunks accumulate across runs.

## Consequences

- The chunking parameters are tuned for Python codebases and English documentation. Other languages (Java, Rust, etc.) would benefit from adding language-specific splitters.
- Changing the embedding model resizes the chunks automatically, but it also invalidates the index: chunk boundaries feed the deterministic point IDs, so a model swap is a full re-ingest either way.
- The characters-per-token ratio was measured on one corpus. A corpus of a different character (minified assets, a non-English language) could sit below it, which the ingest-time truncation report is there to surface.
- Markdown header preservation means markdown chunks carry richer metadata than code chunks, which can improve retrieval for documentation-heavy repos.
