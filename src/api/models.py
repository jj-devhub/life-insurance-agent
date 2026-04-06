# git commit: feat(api): add request/response Pydantic models
# Module: api/models
"""
Pydantic request and response models for the FastAPI REST API.

Defines the contract for all API endpoints including chat, knowledge base,
memory, and health check routes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Chat models
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    """Request body for the POST /api/v1/chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's message text.",
    )
    user_id: str = Field(
        default="default_user",
        description="Unique user identifier for memory persistence.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Session ID to continue an existing conversation. Auto-generated if not provided."
        ),
    )


class ChatResponse(BaseModel):
    """Response body for the POST /api/v1/chat endpoint."""

    response: str = Field(..., description="The assistant's response text.")
    session_id: str = Field(..., description="Session identifier.")
    intent: str = Field(..., description="Classified intent (e.g., 'policy_inquiry').")
    agent_used: str = Field(..., description="Name of the agent that handled the query.")
    confidence: float = Field(..., description="Intent classification confidence (0–1).")
    sources: list[str] = Field(default_factory=list, description="KB sources referenced.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp.")


class ChatHistoryItem(BaseModel):
    """A single message in the chat history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message content text.")


class ChatHistoryResponse(BaseModel):
    """Response body for GET /api/v1/chat/history/{session_id}."""

    session_id: str
    user_id: str
    messages: list[ChatHistoryItem]
    turn_count: int
    created_at: str
    last_activity: str


# --------------------------------------------------------------------------- #
# Knowledge Base models
# --------------------------------------------------------------------------- #


class KBEntryResponse(BaseModel):
    """A knowledge base entry in API responses."""

    category: str
    topic: str
    keywords: list[str]
    summary: str
    source_file: str | None = None
    related_topics: list[str] = Field(default_factory=list)


class KBListResponse(BaseModel):
    """Response body for GET /api/v1/knowledge-base."""

    total_entries: int
    categories: list[str]
    entries: list[KBEntryResponse]


class KBReloadResponse(BaseModel):
    """Response body for POST /api/v1/knowledge-base/reload."""

    success: bool
    entries_loaded: int
    chunks_indexed: int
    message: str


# --------------------------------------------------------------------------- #
# Memory models
# --------------------------------------------------------------------------- #


class MemoryItem(BaseModel):
    """A single memory entry from Mem0."""

    id: str | None = None
    memory: str
    created_at: str | None = None


class MemoryResponse(BaseModel):
    """Response body for GET /api/v1/memory/{user_id}."""

    user_id: str
    memories: list[MemoryItem]
    total: int


class MemoryDeleteResponse(BaseModel):
    """Response body for DELETE /api/v1/memory/{user_id}."""

    user_id: str
    success: bool
    message: str


# --------------------------------------------------------------------------- #
# Health check models
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Response body for GET /api/v1/health."""

    status: str = Field(default="ok", description="Overall status: 'ok' or 'degraded'.")
    app_name: str
    version: str = "1.0.0"
    llm_provider: str
    mem0_enabled: bool
    kb_indexed: bool
    kb_entries: int


# --------------------------------------------------------------------------- #
# Error model
# --------------------------------------------------------------------------- #


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str | None = None
    status_code: int
