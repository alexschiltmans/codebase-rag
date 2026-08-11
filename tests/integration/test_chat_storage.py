#!/usr/bin/env python
"""Test script to verify chat history persistence in SQLite.

This script tests the chat history storage mechanism by:
1. Creating a new chat history with test messages
2. Saving it to SQLite
3. Retrieving it back from SQLite
4. Verifying the data integrity

Usage:
    uv run python tests/integration/test_chat_storage.py

"""

import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from codebase_rag.database.chat_storage import ChatHistoryManager
except ImportError:
    from codebase_rag.database.chat_storage import ChatHistoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_chat_storage")


def test_chat_history_persistence() -> None:
    """Test chat history persistence functionality."""
    test_chat_id = str(uuid.uuid4())
    logger.info("Testing chat history persistence with chat_id: %s", test_chat_id)

    test_messages = [
        {"role": "user", "content": "Hello, this is a test message 1"},
        {"role": "assistant", "content": "Hello! I'm responding to test message 1"},
        {"role": "user", "content": "This is test message 2"},
        {"role": "assistant", "content": "And this is the response to test message 2"},
    ]

    chat_manager = ChatHistoryManager()

    logger.info("Saving test messages to SQLite...")
    chat_manager.save_chat_history(test_chat_id, test_messages)

    logger.info("Retrieving test messages from SQLite...")
    retrieved_messages = chat_manager.get_chat_history(test_chat_id)

    assert retrieved_messages, "Failed to retrieve messages"
    assert len(retrieved_messages) == len(test_messages), (
        f"Message count mismatch: expected {len(test_messages)}, got {len(retrieved_messages)}"
    )

    for i, (original, retrieved) in enumerate(zip(test_messages, retrieved_messages, strict=False)):
        assert original["role"] == retrieved["role"], (
            f"Message {i} role mismatch: original={original['role']}, retrieved={retrieved['role']}"
        )
        assert original["content"] == retrieved["content"], (
            f"Message {i} content mismatch: original={original['content']}, retrieved={retrieved['content']}"
        )

    logger.info("Chat history persistence test PASSED")

    logger.info("Listing all chat histories...")
    chat_list = chat_manager.list_chat_histories()
    logger.info("Found %d chat histories", len(chat_list))


if __name__ == "__main__":
    try:
        test_chat_history_persistence()
        sys.exit(0)
    except Exception as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)
