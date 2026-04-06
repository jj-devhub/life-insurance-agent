# git commit: feat(api): add chat route handlers
# Module: api/routes/chat
"""
Chat API route handlers.

Endpoints:
    POST   /api/v1/chat                    - Send a message and get a response
    GET    /api/v1/chat/history/{session_id} - Get conversation history
    DELETE /api/v1/chat/history/{session_id} - Clear session history
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.agents.graph import chat as agent_chat
from src.api.models import (
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)
from src.memory.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

# Shared session manager instance (created at app startup)
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


@router.post("", response_model=ChatResponse)
async def send_message(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the Life Insurance Support Assistant.

    The message is processed through the LangGraph multi-agent workflow:
    1. Mem0 retrieves relevant past memories
    2. Supervisor classifies intent and routes to specialist agent
    3. Specialist agent generates response using KB context
    4. Interaction is saved to Mem0 for future recall

    Returns the assistant's response with metadata (intent, agent, confidence).
    """
    sm = get_session_manager()

    try:
        # Get or create session
        session_id = sm.get_or_create_session(request.session_id, request.user_id)

        # Get existing conversation history for context
        from langchain_core.messages import AIMessage, HumanMessage

        existing_messages = sm.get_messages(session_id)

        # Add user message to session
        user_msg = HumanMessage(content=request.message)
        sm.add_message(session_id, user_msg)

        # Invoke the agent graph
        result = agent_chat(
            message=request.message,
            user_id=request.user_id,
            session_id=session_id,
            history=existing_messages,
        )

        # Add assistant response to session
        ai_msg = AIMessage(content=result["response"])
        sm.add_message(session_id, ai_msg)

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            agent_used=result.get("agent", "unknown"),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            timestamp=datetime.now(timezone.utc),
        )

    except Exception as e:
        logger.error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    """Get the full conversation history for a session."""
    sm = get_session_manager()

    try:
        history = sm.get_history(session_id)
        info = sm.get_session_info(session_id)

        return ChatHistoryResponse(
            session_id=session_id,
            user_id=info["user_id"],
            messages=[ChatHistoryItem(**msg) for msg in history],
            turn_count=info["turn_count"],
            created_at=info["created_at"],
            last_activity=info["last_activity"],
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> dict:
    """Delete a conversation session and its history."""
    sm = get_session_manager()
    deleted = sm.clear_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return {"message": f"Session {session_id} cleared successfully"}
