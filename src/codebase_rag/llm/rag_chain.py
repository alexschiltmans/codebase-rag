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

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        prompt_template: str | None = None,
        top_k: int = 5,
        use_conversation_memory: bool = True,
        max_conversation_history: int = 5,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            retriever: Document retriever component.
            llm: Language model for generation.
            prompt_template: Optional custom prompt template.
            top_k: Number of documents to retrieve.
            use_conversation_memory: Whether to use conversation memory.
            max_conversation_history: Maximum number of conversation turns to keep.
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.use_conversation_memory = use_conversation_memory
        self.conversation_history: list[dict[str, Any]] = []
        self.max_conversation_history = max_conversation_history
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
            retrieval_time = time.time() - start_time
            logger.debug("Retrieved %d documents in %.2f seconds", len(documents), retrieval_time)

            if retrieval_span:
                retrieval_span.end(output={"documents_retrieved": len(documents), "retrieval_time": retrieval_time})

            if not documents:
                return self._empty_retrieval_result(start_time, retrieval_time, trace)

            context = self._create_context(documents)
            prompt = self._create_prompt(query, context)

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
                    "documents_retrieved": len(documents),
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
            retrieval_time = time.time() - start_time
            logger.debug("Retrieved %d documents in %.2f seconds", len(documents), retrieval_time)

            if retrieval_span:
                retrieval_span.end(output={"documents_retrieved": len(documents), "retrieval_time": retrieval_time})

            if not documents:
                self.last_result = self._empty_retrieval_result(start_time, retrieval_time, trace)
                yield str(self.last_result["answer"])
                return

            context = self._create_context(documents)
            prompt = self._create_prompt(query, context)

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
                    "documents_retrieved": len(documents),
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
        """Retrieve and filter relevant documents for a query."""
        try:
            return self._do_retrieve(query, top_k)
        except TypeError:
            return self._do_retrieve(query)

    def _do_retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Execute retrieval, dispatching based on retriever capabilities.

        Relevance filtering happens inside the retriever itself (see
        `VectorRetriever.search`), not here — after RRF fusion, a fused
        score can no longer distinguish relevant from irrelevant results.
        """
        if hasattr(self.retriever, "search"):
            args = (query, top_k) if top_k is not None else (query,)
            documents_and_scores = self.retriever.search(*args)
            return [doc for doc, _ in documents_and_scores]
        return self.retriever.get_relevant_documents(query)  # type: ignore[no-any-return]

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
            content = getattr(doc, "page_content", "") or getattr(doc, "content", "")
            metadata = getattr(doc, "metadata", {}) or {}

            source_info = ""
            if "source" in metadata:
                source_info = f"Source: {metadata['source']}"

            doc_text = f"[Document {i + 1}] {content}\n{source_info}\n"
            context_parts.append(doc_text)

        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the language model.

        Args:
            query: The user query.
            context: The document context.

        Returns:
            Formatted prompt string.
        """
        conversation_history = self._format_conversation_history() if self.use_conversation_memory else ""

        return self.prompt_template.format(question=query, context=context, conversation_history=conversation_history)

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

        user_count = sum(1 for msg in self.conversation_history if msg["role"] == "user")

        if user_count > self.max_conversation_history:
            excess_count = user_count - self.max_conversation_history

            removed_user_count = 0
            i = 0
            while i < len(self.conversation_history) and removed_user_count < excess_count:
                if self.conversation_history[i]["role"] == "user":
                    removed_user_count += 1
                i += 1

            if i > 0:
                self.conversation_history = self.conversation_history[i:]

    def _format_conversation_history(self) -> str:
        """Format the conversation history for inclusion in the prompt.

        Returns:
            Formatted conversation history string
        """
        if not self.use_conversation_memory or not self.conversation_history:
            return "No previous conversation."

        formatted_messages = []
        for message in self.conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            formatted_messages.append(f"{role}: {content}")

        return "Previous conversation:\n" + "\n\n".join(formatted_messages)
