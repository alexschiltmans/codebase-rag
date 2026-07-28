"""RAG (Retrieval-Augmented Generation) chain implementation.

This module implements a RAG chain that combines document retrieval with
language model generation to produce answers grounded in a knowledge base.
"""

import logging
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document

from codebase_rag.config import Config
from codebase_rag.retrieval.retriever_protocol import RetrieverProtocol

logger = logging.getLogger(__name__)

# Lazy-initialized Langfuse client
_langfuse = None


def _get_langfuse() -> Any:
    """Get or initialize the Langfuse client (lazy singleton)."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse

    config = Config.get_instance()
    if not config.langfuse_enabled:
        return None

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
        )
        logger.info("Langfuse tracing initialized")
        return _langfuse
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s", e)
        return None


class RAGChain:
    """Retrieval-Augmented Generation (RAG) chain.

    This class implements a RAG chain that combines document retrieval with
    language model generation to produce factual answers grounded in the
    retrieved knowledge.

    Uses the Chain of Responsibility pattern to process the query through
    multiple steps (retrieval, prompt construction, generation).
    """

    _HISTORY_PREFIX = "Previous conversation:\n"

    def __init__(
        self,
        retriever: RetrieverProtocol,
        llm: Any,
        prompt_template: str | None = None,
        top_k: int = 5,
        use_conversation_memory: bool = True,
        max_conversation_history: int = 5,
        prompt_budget_chars: int | None = None,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            retriever: Document retriever component (any `RetrieverProtocol`).
            llm: Language model for generation.
            prompt_template: Optional custom prompt template.
            top_k: Number of documents to retrieve.
            use_conversation_memory: Whether to use conversation memory.
            max_conversation_history: Maximum number of conversation turns to keep.
            prompt_budget_chars: Maximum prompt length in characters, derived from
                the LLM's `num_ctx`. `None` disables budget enforcement.
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.use_conversation_memory = use_conversation_memory
        self.conversation_history: list[dict[str, Any]] = []
        self.max_conversation_history = max_conversation_history
        self.prompt_budget_chars = prompt_budget_chars
        # Populated by stream() once its generator is fully consumed, since
        # st.write_stream() only returns the concatenated text — callers that
        # need sources/metrics read them from here afterward.
        self.last_result: dict[str, Any] | None = None

        if prompt_template is None:
            self.prompt_template = (
                "You are a helpful coding assistant. "
                "Answer the question based on the context below.\n\n"
                "{conversation_history}\n\n"
                "Context information:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer: "
            )
        else:
            self.prompt_template = prompt_template

        logger.info("Initialized RAG chain with top_k=%d, use_conversation_memory=%s", top_k, use_conversation_memory)

    def run(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Run the RAG chain on the given query.

        Args:
            query: The user query.
            **kwargs: Additional parameters for retrieval or generation.

        Returns:
            Dict containing the answer and source documents.
        """
        langfuse = _get_langfuse()
        trace = langfuse.trace(name="rag-chain", input={"query": query}) if langfuse else None

        try:
            start_time = time.time()

            if self.use_conversation_memory:
                self.add_user_message(query)

            # Retrieve relevant documents
            top_k = kwargs.get("top_k", self.top_k)
            retrieval_span = trace.span(name="retrieval", input={"query": query, "top_k": top_k}) if trace else None
            documents = self._retrieve_documents(query, top_k)
            documents_retrieved = len(documents)
            retrieval_time = time.time() - start_time
            logger.debug("Retrieved %d documents in %.2f seconds", documents_retrieved, retrieval_time)

            if retrieval_span:
                retrieval_span.end(
                    output={"documents_retrieved": documents_retrieved, "retrieval_time": retrieval_time}
                )

            if not documents:
                return self._empty_retrieval_result(start_time, retrieval_time, trace)

            prompt, documents, docs_dropped, history_dropped, question_truncated_chars = self._build_within_budget(
                query, documents
            )

            generation_span = trace.span(name="generation", input={"prompt_length": len(prompt)}) if trace else None
            generation_start = time.time()
            answer = self.llm.invoke(prompt)
            generation_time = time.time() - generation_start
            logger.debug("Generated answer in %.2f seconds", generation_time)

            if generation_span:
                generation_span.end(output={"answer_length": len(answer), "generation_time": generation_time})

            sources = self._format_sources(documents)

            if self.use_conversation_memory:
                self.add_assistant_message(answer, sources)

            total_time = time.time() - start_time
            logger.info("RAG chain completed in %.2f seconds", total_time)

            result: dict[str, Any] = {
                "answer": answer,
                "sources": sources,
                "documents": documents,
                "prompt": prompt,
                "metrics": {
                    "total_time": total_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "documents_retrieved": documents_retrieved,
                    "context_docs_dropped": docs_dropped,
                    "history_messages_dropped": history_dropped,
                    "question_truncated_chars": question_truncated_chars,
                },
            }
            if trace:
                trace.update(output=result)
            return result
        except Exception as e:
            logger.error("Error running RAG chain: %s", e)
            raise

    def stream(self, query: str, **kwargs: Any) -> Iterator[str]:
        """Run the RAG chain on the given query, streaming the answer as it's generated.

        Retrieval happens synchronously first (same as `run()`), then generation is
        yielded chunk by chunk. Once the generator is fully consumed, `self.last_result`
        holds the same dict shape `run()` returns — callers that need sources or
        metrics (rather than just the displayed text) should read it from there
        afterward, since a generator can't both yield text and return a value.

        Args:
            query: The user query.
            **kwargs: Additional parameters for retrieval or generation.

        Yields:
            Successive text chunks of the generated answer.
        """
        langfuse = _get_langfuse()
        trace = langfuse.trace(name="rag-chain", input={"query": query}) if langfuse else None

        try:
            start_time = time.time()

            if self.use_conversation_memory:
                self.add_user_message(query)

            # Retrieve relevant documents
            top_k = kwargs.get("top_k", self.top_k)
            retrieval_span = trace.span(name="retrieval", input={"query": query, "top_k": top_k}) if trace else None
            documents = self._retrieve_documents(query, top_k)
            documents_retrieved = len(documents)
            retrieval_time = time.time() - start_time
            logger.debug("Retrieved %d documents in %.2f seconds", documents_retrieved, retrieval_time)

            if retrieval_span:
                retrieval_span.end(
                    output={"documents_retrieved": documents_retrieved, "retrieval_time": retrieval_time}
                )

            if not documents:
                self.last_result = self._empty_retrieval_result(start_time, retrieval_time, trace)
                yield str(self.last_result["answer"])
                return

            prompt, documents, docs_dropped, history_dropped, question_truncated_chars = self._build_within_budget(
                query, documents
            )

            generation_span = trace.span(name="generation", input={"prompt_length": len(prompt)}) if trace else None
            generation_start = time.time()

            chunks: list[str] = []
            for chunk in self._stream_llm(prompt):
                chunks.append(chunk)
                yield chunk
            answer = "".join(chunks)

            generation_time = time.time() - generation_start
            logger.debug("Generated answer in %.2f seconds", generation_time)

            if generation_span:
                generation_span.end(output={"answer_length": len(answer), "generation_time": generation_time})

            sources = self._format_sources(documents)

            if self.use_conversation_memory:
                self.add_assistant_message(answer, sources)

            total_time = time.time() - start_time
            logger.info("RAG chain (streamed) completed in %.2f seconds", total_time)

            result: dict[str, Any] = {
                "answer": answer,
                "sources": sources,
                "documents": documents,
                "prompt": prompt,
                "metrics": {
                    "total_time": total_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "documents_retrieved": documents_retrieved,
                    "context_docs_dropped": docs_dropped,
                    "history_messages_dropped": history_dropped,
                    "question_truncated_chars": question_truncated_chars,
                },
            }
            if trace:
                trace.update(output=result)
            self.last_result = result
        except Exception as e:
            logger.error("Error streaming RAG chain: %s", e)
            raise

    def _stream_llm(self, prompt: str) -> Iterator[str]:
        """Stream text chunks from the LLM, falling back to one chunk if it can't stream."""
        if hasattr(self.llm, "stream"):
            yield from self.llm.stream(prompt)
        else:
            yield self.llm.invoke(prompt)

    def _retrieve_documents(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents for a query through the retriever protocol.

        Relevance filtering happens inside the retriever itself (see
        `VectorRetriever.search`), not here — after RRF fusion, a fused
        score can no longer distinguish relevant from irrelevant results.
        Exceptions from the retriever propagate to `run()`/`stream()`
        rather than being caught and retried.
        """
        return [doc for doc, _ in self.retriever.search(query, top_k)]

    def _empty_retrieval_result(self, start_time: float, retrieval_time: float, trace: Any) -> dict[str, Any]:
        """Build the response dict when no relevant documents are found."""
        default_answer = (
            "I couldn't find any relevant information in the ingested codebases to answer "
            "this question. This could mean:\n\n"
            "- The topic isn't covered in the ingested repositories\n"
            "- Try rephrasing your question with different keywords\n"
            "- The relevant code or documentation may not have been ingested yet"
        )

        if self.use_conversation_memory:
            self.add_assistant_message(default_answer)

        result: dict[str, Any] = {
            "answer": default_answer,
            "sources": [],
            "documents": [],
            "prompt": "",
            "metrics": {
                "total_time": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "generation_time": 0,
                "documents_retrieved": 0,
                "context_docs_dropped": 0,
                "history_messages_dropped": 0,
                "question_truncated_chars": 0,
            },
        }
        if trace:
            trace.update(output=result)
        return result

    def _create_context(self, documents: list[Document]) -> str:
        """Create context string from retrieved documents.

        Args:
            documents: List of retrieved documents.

        Returns:
            String containing the document contents.
        """
        if not documents:
            return "No relevant information found."

        context_parts = []

        for i, doc in enumerate(documents):
            context_parts.append(self._doc_block(doc, i + 1))

        return "\n\n".join(context_parts)

    def _doc_block(self, doc: Document, index: int) -> str:
        """Render one document the way `_create_context` labels it, so estimates account for `index`'s digit width."""
        content = getattr(doc, "page_content", "") or getattr(doc, "content", "")
        metadata = getattr(doc, "metadata", {}) or {}
        source_info = f"Source: {metadata['source']}" if "source" in metadata else ""
        return f"[Document {index}] {content}\n{source_info}\n"

    def _format_sources(self, documents: list[Any]) -> list[dict[str, str]]:
        """Format sources for citation in the response.

        This function formats document sources for display in the UI, ensuring that
        paths are properly formatted for the codebase repositories.

        Args:
            documents: Either a list of Documents, or a list of (Document, score) tuples.

        Returns:
            List of source dictionaries with ID, file path, and file name.
        """
        sources = []
        for i, doc_item in enumerate(documents):
            doc = doc_item[0] if isinstance(doc_item, tuple) and len(doc_item) == 2 else doc_item

            source = doc.metadata.get("source", "unknown")

            file_name = doc.metadata.get("file_name", "")
            if not file_name and source != "unknown":
                file_name = source.split("/")[-1] if "/" in source else source

            repo = doc.metadata.get("repo", "")
            if repo and not file_name.startswith(f"[{repo.upper()}]"):
                file_name = f"[{repo.upper()}] {file_name}"

            sources.append(
                {
                    "id": str(i + 1),
                    "file_path": source,
                    "file_name": file_name,
                }
            )
        return sources

    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history.

        Args:
            message: The user's message
        """
        if not self.use_conversation_memory:
            return

        self.conversation_history.append({"role": "user", "content": message})
        self._trim_conversation_history()

    def add_assistant_message(self, message: str, sources: list[dict[str, str]] | None = None) -> None:
        """Add an assistant message to the conversation history.

        Args:
            message: The assistant's response
            sources: Optional list of sources used in the response
        """
        if not self.use_conversation_memory:
            return

        assistant_message: dict[str, Any] = {"role": "assistant", "content": message}

        if sources:
            assistant_message["sources"] = sources

        self.conversation_history.append(assistant_message)
        self._trim_conversation_history()

    def _trim_conversation_history(self) -> None:
        """Trim conversation history to maximum allowed turns."""
        if not self.conversation_history or self.max_conversation_history <= 0:
            return

        user_indices = [i for i, msg in enumerate(self.conversation_history) if msg["role"] == "user"]

        if len(user_indices) > self.max_conversation_history:
            cutoff = user_indices[-self.max_conversation_history]
            self.conversation_history = self.conversation_history[cutoff:]

    def _format_conversation_history(self, history: list[dict[str, Any]] | None = None) -> str:
        """Format the conversation history for inclusion in the prompt.

        Args:
            history: Messages to format. Defaults to `self.conversation_history`.

        Returns:
            Formatted conversation history string
        """
        if history is None:
            history = self.conversation_history

        if not self.use_conversation_memory or not history:
            return "No previous conversation."

        formatted_messages = [self._history_line(message) for message in history]

        return self._HISTORY_PREFIX + "\n\n".join(formatted_messages)

    def _history_line(self, message: dict[str, Any]) -> str:
        """Render one history message the way `_format_conversation_history` does, keeping length estimates in sync."""
        return f"{message['role'].capitalize()}: {message['content']}"

    def _build_within_budget(self, query: str, documents: list[Document]) -> tuple[str, list[Document], int, int, int]:
        """Assemble a prompt that fits `prompt_budget_chars`, if set.

        Drops the lowest-ranked (last) context document first, then the
        oldest conversation history message. If the prompt still doesn't
        fit once both are exhausted, the question itself is truncated
        (head kept, tail cut, with an elision marker) as a last resort —
        the alternative is Ollama truncating it silently server-side.
        The prompt template is never altered. Per-item character costs
        are computed once so drops don't require rebuilding the whole
        prompt on every iteration; the final prompt is assembled once.

        Returns:
            Tuple of (prompt, documents used, docs dropped, history
            messages dropped, question characters truncated).
        """
        docs = list(documents)
        history = list(self.conversation_history) if self.use_conversation_memory else []
        # run()/stream() just added the current query as the last history entry; drop it so it isn't counted twice.
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        def build(docs_subset: list[Document], history_subset: list[dict[str, Any]], question: str) -> str:
            context = self._create_context(docs_subset)
            conversation_history = (
                self._format_conversation_history(history_subset) if self.use_conversation_memory else ""
            )
            return self.prompt_template.format(
                question=question, context=context, conversation_history=conversation_history
            )

        if self.prompt_budget_chars is None:
            return build(docs, history, query), docs, 0, 0, 0

        fixed_len = len(self.prompt_template.format(question=query, context="", conversation_history=""))
        doc_lens = [len(self._doc_block(d, i + 1)) for i, d in enumerate(docs)]
        no_context_len = len(self._create_context([]))
        msg_lens = [len(self._history_line(m)) for m in history]
        no_history_len = len(self._format_conversation_history([])) if self.use_conversation_memory else 0

        n_docs = len(docs)
        n_hist = len(history)
        # Running totals updated incrementally as items drop, instead of re-summing the surviving slice each time.
        context_total = no_context_len if n_docs == 0 else sum(doc_lens) + 2 * (n_docs - 1)
        if n_hist == 0 or not self.use_conversation_memory:
            history_total = no_history_len
        else:
            history_total = len(self._HISTORY_PREFIX) + sum(msg_lens) + 2 * (n_hist - 1)
        total = fixed_len + context_total + history_total

        docs_dropped = 0
        history_dropped = 0

        while total > self.prompt_budget_chars and n_docs > 0:
            # Dropping the last doc swaps the whole block for the empty sentinel; otherwise just its length and a join.
            new_context_total = no_context_len if n_docs == 1 else context_total - doc_lens[n_docs - 1] - 2
            new_total = total - context_total + new_context_total
            if new_total >= total:
                # A doc shorter than its replacement sentinel would grow the prompt, not shrink it — stop.
                break
            n_docs -= 1
            docs_dropped += 1
            context_total = new_context_total
            total = new_total

        while total > self.prompt_budget_chars and n_hist > 0:
            # Dropping oldest-first means the item about to go is at index len(history) - n_hist, not n_hist - 1.
            dropped_len = msg_lens[len(history) - n_hist]
            new_history_total = no_history_len if n_hist == 1 else history_total - dropped_len - 2
            new_total = total - history_total + new_history_total
            if new_total >= total:
                break
            n_hist -= 1
            history_dropped += 1
            history_total = new_history_total
            total = new_total

        docs = docs[:n_docs]
        history = history[len(history) - n_hist :] if n_hist else []

        if docs_dropped or history_dropped:
            logger.warning(
                "Prompt exceeded budget of %d chars; dropped %d context docs and %d history messages",
                self.prompt_budget_chars,
                docs_dropped,
                history_dropped,
            )

        question = query
        question_truncated_chars = 0
        if total > self.prompt_budget_chars:
            excess = total - self.prompt_budget_chars
            marker = "... [truncated]"
            keep = max(0, len(query) - excess - len(marker))
            truncated_question = query[:keep] + marker
            # The marker alone can be longer than a short query, so only truncate if it actually shrinks things.
            if len(truncated_question) < len(query):
                question = truncated_question
                question_truncated_chars = len(query) - keep
                total -= len(query) - len(truncated_question)
                logger.warning(
                    "Prompt still exceeded budget of %d chars after dropping context and history; "
                    "truncated question by %d chars",
                    self.prompt_budget_chars,
                    question_truncated_chars,
                )

            if total > self.prompt_budget_chars:
                # Nothing left helps; send it over budget anyway (template and question stay intact) but log loudly.
                logger.warning(
                    "Prompt of %d chars still exceeds budget of %d chars after dropping all "
                    "context and history and truncating the question as far as it can go; "
                    "sending it over budget rather than dropping the question or template",
                    total,
                    self.prompt_budget_chars,
                )

        prompt = build(docs, history, question)
        return prompt, docs, docs_dropped, history_dropped, question_truncated_chars
