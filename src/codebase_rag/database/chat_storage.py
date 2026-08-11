"""Module for storing chat history.

This module provides functionality to save and retrieve chat history using
SQLite for persistent local storage of chat sessions.

Uses the Repository design pattern to provide a layer of abstraction over the storage backend.
"""

import logging
from typing import Any, Protocol

from codebase_rag.database.sqlite_storage import SqliteChatStorage

logger = logging.getLogger(__name__)


class ChatStorageProvider(Protocol):
    """Protocol defining the interface for chat storage providers."""

    def save_chat(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        """Save a chat session.

        Args:
            chat_id: Unique identifier for the chat session
            messages: List of message dictionaries
        """
        ...

    def get_chat(self, chat_id: str) -> list[dict[str, Any]] | None:
        """Retrieve a chat session by ID.

        Args:
            chat_id: Unique identifier for the chat session

        Returns:
            list[dict[str, Any]] | None: The chat messages or None if not found
        """
        ...

    def list_chats(self) -> list[str]:
        """List all available chat IDs.

        Returns:
            list[str]: List of chat IDs
        """
        ...

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session.

        Args:
            chat_id: Unique identifier for the chat session

        Returns:
            bool: True if successful, False otherwise
        """
        ...


class ChatHistoryManager:
    """Manager for saving and retrieving chat history.

    Uses SQLite for persistent local storage.
    Implements the Facade pattern to provide a simple interface over the storage backend.
    """

    def __init__(self) -> None:
        """Initialize chat history manager."""
        try:
            self.storage: SqliteChatStorage | None = SqliteChatStorage()
            logger.info("Initialized SQLite storage for chat history")
        except Exception as e:
            logger.error("Failed to initialize SQLite storage: %s", e)
            self.storage = None

    def save_chat_history(self, chat_id: str, messages: list[dict[str, Any]]) -> bool:
        """Save chat history to storage.

        Args:
            chat_id: Unique identifier for the chat session
            messages: List of message dictionaries

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.storage:
            return False

        try:
            self.storage.save_chat(chat_id, messages)
            return True
        except Exception as e:
            logger.error("Failed to save chat history: %s", e)
            return False

    def get_chat_history(self, chat_id: str) -> list[dict[str, Any]] | None:
        """Retrieve chat history.

        Args:
            chat_id: Unique identifier for the chat session

        Returns:
            list[dict[str, Any]] | None: The chat messages or None if not found
        """
        if not self.storage:
            return None

        try:
            return self.storage.get_chat(chat_id)
        except Exception as e:
            logger.error("Failed to retrieve chat history: %s", e)
            return None

    def list_chat_histories(self) -> list[dict[str, Any]]:
        """List all available chat histories with metadata.

        Returns:
            list[dict[str, Any]]: List of chat metadata dictionaries
        """
        if not self.storage:
            return []

        try:
            return self.storage.list_chats()
        except Exception as e:
            logger.error("Failed to list chat histories: %s", e)
            return []

    def delete_chat_history(self, chat_id: str) -> bool:
        """Delete a chat history from storage.

        Args:
            chat_id: Unique identifier for the chat session

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.storage:
            return False

        try:
            return self.storage.delete_chat(chat_id)
        except Exception as e:
            logger.error("Failed to delete chat history: %s", e)
            return False


_chat_history_manager_instance = None


def get_chat_history_manager() -> ChatHistoryManager:
    """Get the singleton instance of ChatHistoryManager.

    Returns:
        ChatHistoryManager: The chat history manager instance
    """
    global _chat_history_manager_instance

    if _chat_history_manager_instance is None:
        _chat_history_manager_instance = ChatHistoryManager()

    return _chat_history_manager_instance
