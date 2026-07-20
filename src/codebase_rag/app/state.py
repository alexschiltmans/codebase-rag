"""Typed accessor for per-session Streamlit state.

Cross-session state lives on ``AppRuntime`` (see ``runtime.py``); everything
in this module is scoped to a single browser session via ``st.session_state``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import streamlit as st


class QueryLifecycle(StrEnum):
    """States a chat query moves through, replacing the old
    ``thinking``/``processing_query`` boolean pair."""

    IDLE = "idle"
    PENDING = "pending"
    FAILED = "failed"


_DEFAULTS: dict[str, Any] = {
    "query_state": QueryLifecycle.IDLE,
    "pending_query": None,
    "query_error": None,
    "messages": [],
    "chat_histories": {},
    "current_chat_id": None,
}


@dataclass
class SessionState:
    """Thin wrapper around ``st.session_state`` for the values this app owns.

    Reads and writes go straight through to ``st.session_state`` so widget
    callbacks and reruns see the same values; this class exists to give
    those values names and types instead of scattering string keys.
    """

    _store: Any = field(default_factory=lambda: st.session_state)

    def ensure_defaults(self) -> None:
        for key, default in _DEFAULTS.items():
            if key not in self._store:
                self._store[key] = [] if isinstance(default, list) else {} if isinstance(default, dict) else default

    @property
    def query_state(self) -> QueryLifecycle:
        return QueryLifecycle(self._store.get("query_state", QueryLifecycle.IDLE))

    @query_state.setter
    def query_state(self, value: QueryLifecycle) -> None:
        self._store["query_state"] = value

    @property
    def pending_query(self) -> str | None:
        return self._store.get("pending_query")  # type: ignore[no-any-return]

    @property
    def query_error(self) -> str | None:
        return self._store.get("query_error")  # type: ignore[no-any-return]

    def submit_query(self, query: str) -> None:
        """Move IDLE -> PENDING with the query text attached."""
        self._store["pending_query"] = query
        self._store["query_error"] = None
        self._store["query_state"] = QueryLifecycle.PENDING

    def query_succeeded(self) -> None:
        """Move PENDING -> IDLE after a successful (or already-rendered) answer."""
        self._store["pending_query"] = None
        self._store["query_error"] = None
        self._store["query_state"] = QueryLifecycle.IDLE

    def query_failed(self, error: str) -> None:
        """Move PENDING -> FAILED, keeping the query so Retry can resubmit it."""
        self._store["query_error"] = error
        self._store["query_state"] = QueryLifecycle.FAILED

    def dismiss_failure(self) -> None:
        """Move FAILED -> IDLE without resubmitting."""
        self._store["pending_query"] = None
        self._store["query_error"] = None
        self._store["query_state"] = QueryLifecycle.IDLE

    def retry_failed_query(self) -> None:
        """Move FAILED -> PENDING, resubmitting the same query once."""
        self._store["query_error"] = None
        self._store["query_state"] = QueryLifecycle.PENDING

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._store["messages"]  # type: ignore[no-any-return]

    @property
    def current_chat_id(self) -> str | None:
        return self._store.get("current_chat_id")  # type: ignore[no-any-return]

    def ensure_current_chat(self) -> str:
        """Return the current chat id, lazily minting one if none exists yet.

        Replaces eagerly creating a chat row in ``initialize_chat_history``,
        which produced a phantom "New Chat" entry before the first message.
        """
        chat_id = self._store.get("current_chat_id")
        if chat_id is None:
            chat_id = str(uuid.uuid4())
            self._store["current_chat_id"] = chat_id
            self._store["chat_histories"][chat_id] = []
        return chat_id  # type: ignore[no-any-return]

    def start_new_chat(self) -> None:
        new_chat_id = str(uuid.uuid4())
        self._store["chat_histories"][new_chat_id] = []
        self._store["current_chat_id"] = new_chat_id
        self._store["messages"] = []

    def switch_chat(self, chat_id: str) -> None:
        self._store["current_chat_id"] = chat_id
        self._store["messages"] = self._store["chat_histories"][chat_id].copy()

    def append_message(
        self,
        role: str,
        content: str,
        sources: list[dict[str, str]] | None = None,
        *,
        question_truncated: bool = False,
    ) -> None:
        message: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and sources:
            message["sources"] = sources
        if role == "assistant" and question_truncated:
            message["question_truncated"] = True
        self._store["messages"].append(message)

        chat_id = self.ensure_current_chat()
        self._store["chat_histories"][chat_id].append(message)

    def chat_history_for(self, chat_id: str) -> list[dict[str, Any]]:
        return self._store["chat_histories"][chat_id]  # type: ignore[no-any-return]

    @property
    def chat_histories(self) -> dict[str, list[dict[str, Any]]]:
        return self._store["chat_histories"]  # type: ignore[no-any-return]


def get_session() -> SessionState:
    """Return the (lazily-defaulted) session-state accessor for this session."""
    state = SessionState()
    state.ensure_defaults()
    return state
