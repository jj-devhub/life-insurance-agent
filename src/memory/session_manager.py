# git commit: feat(memory): add session state manager
# Module: memory/session_manager
"""
In-memory session state manager for tracking per-session conversation history.

Works alongside Mem0 (which handles cross-session memory). This module handles
within-session state: message history, session IDs, and session metadata.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages in-memory conversation sessions.

    Each session has a unique ID, a message history, and metadata (user_id,
    creation time, last activity time). Sessions are stored in-memory and
    are not persisted across server restarts.

    Usage:
        sm = SessionManager()
        session_id = sm.create_session("user123")
        sm.add_message(session_id, HumanMessage(content="Hello!"))
        history = sm.get_history(session_id)
    """

    def __init__(self) -> None:
        """Initialize the session store."""
        # session_id -> session data dict
        self._sessions: dict[str, dict] = {}

    def create_session(self, user_id: str = "default_user") -> str:
        """
        Create a new conversation session.

        Args:
            user_id: Identifier for the user owning this session.

        Returns:
            The generated session ID (UUID4 string).
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "turn_count": 0,
        }

        logger.debug("Created session '%s' for user '%s'", session_id, user_id)
        return session_id

    def get_or_create_session(
        self, session_id: str | None = None, user_id: str = "default_user"
    ) -> str:
        """
        Get an existing session or create a new one.

        Args:
            session_id: Optional existing session ID.
            user_id: User identifier (used if creating a new session).

        Returns:
            The session ID (existing or newly created).
        """
        if session_id and session_id in self._sessions:
            return session_id
        return self.create_session(user_id)

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """
        Add a message to the session history.

        Args:
            session_id: The session to add the message to.
            message: A LangChain BaseMessage (HumanMessage, AIMessage, etc.).

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        session["messages"].append(message)
        session["last_activity"] = datetime.now(timezone.utc).isoformat()

        # Increment turn count on human messages
        if isinstance(message, HumanMessage):
            session["turn_count"] += 1

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """
        Get all messages in a session.

        Args:
            session_id: The session ID.

        Returns:
            List of BaseMessage objects in chronological order.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return list(self._sessions[session_id]["messages"])

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Get the session history as a list of simple dicts.

        Args:
            session_id: The session ID.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        messages = self.get_messages(session_id)
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": msg.content})
        return history

    def get_session_info(self, session_id: str) -> dict:
        """
        Get metadata about a session (without full message history).

        Args:
            session_id: The session ID.

        Returns:
            Dict with session metadata.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        return {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "turn_count": session["turn_count"],
            "message_count": len(session["messages"]),
        }

    def clear_session(self, session_id: str) -> bool:
        """
        Delete a session and its history.

        Args:
            session_id: The session to delete.

        Returns:
            True if the session was found and deleted.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Cleared session '%s'", session_id)
            return True
        return False

    def list_sessions(self, user_id: str | None = None) -> list[dict]:
        """
        List all sessions, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by.

        Returns:
            List of session info dicts.
        """
        sessions = []
        for sid, data in self._sessions.items():
            if user_id is None or data["user_id"] == user_id:
                sessions.append(self.get_session_info(sid))
        return sessions
