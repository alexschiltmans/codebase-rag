"""Test script to verify conversation memory in RAG chain.

This script creates a simple RAG chain and tests its conversation memory capabilities.
"""

import logging
from typing import Any

import pytest

from codebase_rag.config import Config
from codebase_rag.database.qdrant_store import QdrantStore
from codebase_rag.llm.ollama_client import OllamaClient
from codebase_rag.llm.rag_chain import RAGChain
from codebase_rag.retrieval.vector_search import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_conversation_memory() -> None:
    """Test the conversation memory functionality of the RAG chain."""
    try:
        config = Config.get_instance()

        qdrant_store = QdrantStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
        )

        vector_retriever = VectorRetriever(qdrant_store)

        llm = OllamaClient(model_name=config.llm_model_name, base_url=config.ollama_base_url, temperature=0.7)
        if llm.check_connection().get("status") != "connected":
            pytest.skip("Ollama not reachable")

        rag_chain = RAGChain(
            retriever=vector_retriever, llm=llm, use_conversation_memory=True, max_conversation_history=5
        )

        response1 = rag_chain.run("What does this codebase do?")
        print_response(response1)
        assert "answer" in response1
        assert isinstance(response1["answer"], str)
        assert len(response1["answer"]) > 0

        # Follow-up questions that rely on conversation context
        response2 = rag_chain.run("What are its main features?")
        print_response(response2)
        assert "answer" in response2
        assert len(response2["answer"]) > 0

        response3 = rag_chain.run("How is it used in practice?")
        print_response(response3)
        assert "answer" in response3
        assert len(response3["answer"]) > 0

        assert len(rag_chain.conversation_history) >= 6  # 3 user + 3 assistant
        roles = [msg["role"] for msg in rag_chain.conversation_history]
        assert roles.count("user") >= 3
        assert roles.count("assistant") >= 3

    except Exception as e:
        logger.error("Error during test: %s", e)
        raise


def print_response(response: dict[str, Any]) -> None:
    """Print a formatted response from the RAG chain.

    Args:
        response: RAG chain response dictionary
    """
    for _source in response["sources"]:
        pass

    metrics = response.get("metrics", {})
    if metrics:
        pass


if __name__ == "__main__":
    test_conversation_memory()
