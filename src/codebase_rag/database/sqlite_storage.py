"""Module for storing chat history in SQLite.

This module provides functionality to save and retrieve chat history using
SQLite for persistent local storage of chat sessions.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebase_rag.config import Config

logger = logging.getLogger(__name__)


class SqliteChatStorage:
    """SQLite-based storage for chat history.

    Conforms to the ChatStorageProvider protocol.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file. Defaults to config value.
        """
        if db_path is None:
            config = Config.get_instance()
            db_path = config.chat_storage_path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Untitled Chat',
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    start_time TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
            logger.info("Initialized SQLite chat storage at %s", self.db_path)
        finally:
            conn.close()

    def save_chat(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        """Save a chat session.

        Args:
            chat_id: Unique identifier for the chat session.
            messages: List of message dictionaries.
        """
        now = datetime.now(UTC).isoformat()

        # Determine title from first user message
        title = "Untitled Chat"
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                content = msg["content"]
                title = content[:50] + "..." if len(content) > 50 else content
                break

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chats (chat_id, messages, title, start_time, last_updated, message_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    messages = excluded.messages,
                    title = excluded.title,
                    last_updated = excluded.last_updated,
                    message_count = excluded.message_count
                """,
                (chat_id, json.dumps(messages), title, now, now, len(messages)),
            )
            conn.commit()
            logger.info("Saved chat %s to SQLite", chat_id)
        except sqlite3.Error as e:
            logger.error("Error saving chat to SQLite: %s", e)
            raise
        finally:
            conn.close()

    def get_chat(self, chat_id: str) -> list[dict[str, Any]] | None:
        """Retrieve a chat session by ID.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            The chat messages or None if not found.
        """
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT messages FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
            if row is None:
                logger.warning("Chat %s not found in SQLite", chat_id)
                return None
            return json.loads(row["messages"])  # type: ignore[no-any-return]
        except sqlite3.Error as e:
            logger.error("Error retrieving chat from SQLite: %s", e)
            raise
        finally:
            conn.close()

    def list_chats(self) -> list[dict[str, Any]]:
        """List all available chats with their metadata.

        Returns:
            List of chat metadata dictionaries, sorted by last_updated descending.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT chat_id, title, user_id, start_time, last_updated, message_count
                FROM chats ORDER BY last_updated DESC
                """
            ).fetchall()

            return [
                {
                    "chat_id": row["chat_id"],
                    "title": row["title"],
                    "user_id": row["user_id"],
                    "start_time": row["start_time"],
                    "last_updated": row["last_updated"],
                    "message_count": row["message_count"],
                }
                for row in rows
            ]
        except sqlite3.Error as e:
            logger.error("Error listing chats from SQLite: %s", e)
            return []
        finally:
            conn.close()

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat session.

        Args:
            chat_id: Unique identifier for the chat session.

        Returns:
            True if successful, False otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted chat %s from SQLite", chat_id)
            else:
                logger.warning("Chat %s not found for deletion", chat_id)
            return deleted
        except sqlite3.Error as e:
            logger.error("Error deleting chat from SQLite: %s", e)
            return False
        finally:
            conn.close()
