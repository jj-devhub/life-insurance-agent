# git commit: feat(agents): add AgentState definition
# Module: agents/state
"""
Shared state definition for the LangGraph multi-agent workflow.

AgentState is a TypedDict that flows through all nodes in the graph.
Each node reads from and writes to this shared state, enabling
communication between the supervisor and specialist agents.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Attributes:
        messages: Conversation message history (auto-appended via add_messages).
        user_id: Unique identifier for the current user.
        session_id: Unique identifier for the current session.
        intent: Classified intent from the supervisor (e.g., "policy_inquiry").
        retrieved_context: Knowledge base context retrieved via RAG.
        mem0_context: Past memories retrieved from Mem0.
        current_agent: Name of the agent currently handling the request.
        confidence: Supervisor's confidence in intent classification (0–1).
        needs_escalation: Flag for human escalation.
        response: The final formatted response text.
        sources: List of KB source references used in the response.
        metadata: Additional metadata (agent timing, debug info, etc.).
    """

    # Core conversation state — messages are auto-appended by LangGraph
    messages: Annotated[list[BaseMessage], add_messages]

    # Session identifiers
    user_id: str
    session_id: str

    # Supervisor routing output
    intent: str
    confidence: float

    # Context injection
    retrieved_context: str
    mem0_context: str

    # Processing state
    current_agent: str
    needs_escalation: bool

    # Output
    response: str
    sources: list[str]
    metadata: dict[str, Any]
