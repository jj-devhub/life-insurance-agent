# git commit: feat(agents): assemble main LangGraph workflow
# Module: agents/graph
"""
Main LangGraph StateGraph assembly for the Life Insurance Support Assistant.

Wires together all agent nodes into a directed graph:
    START → retrieve_memory → supervisor → [agent] → save_memory → END

The graph supports:
    - Intent-based conditional routing via the supervisor
    - RAG context injection from the Qdrant knowledge base
    - Mem0 memory retrieval and persistence
    - Full conversation state management
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.claims_agent import claims_agent_node
from src.agents.fallback_agent import fallback_agent_node, greeting_handler_node
from src.agents.general_agent import general_agent_node
from src.agents.policy_agent import policy_agent_node
from src.agents.state import AgentState
from src.agents.supervisor import route_by_intent, supervisor_node
from src.knowledge.retriever import KnowledgeRetriever
from src.memory.mem0_manager import Mem0Manager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared resources (initialized once per graph creation)
# --------------------------------------------------------------------------- #

_retriever: KnowledgeRetriever | None = None
_mem0: Mem0Manager | None = None


def _get_retriever() -> KnowledgeRetriever:
    """Get or create the knowledge base retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def _get_mem0() -> Mem0Manager:
    """Get or create the Mem0 manager singleton."""
    global _mem0
    if _mem0 is None:
        _mem0 = Mem0Manager()
    return _mem0


# --------------------------------------------------------------------------- #
# Pre/Post processing nodes
# --------------------------------------------------------------------------- #


def retrieve_memory_node(state: AgentState) -> dict:
    """
    Entry node: retrieves relevant Mem0 memories for the current user.

    Searches Mem0 for past context relevant to the user's current query.
    Also retrieves KB context via semantic search for downstream agents.

    Args:
        state: Current AgentState.

    Returns:
        Dict with 'mem0_context' and 'retrieved_context' populated.
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default_user")

    # Get the latest user message for search
    latest_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_query = msg.content
            break

    if not latest_query:
        return {"mem0_context": "", "retrieved_context": ""}

    # 1. Search Mem0 for past memories
    mem0 = _get_mem0()
    mem0_context = mem0.search_memories(user_id, latest_query)

    # 2. Search KB for relevant content
    retriever = _get_retriever()
    kb_context = retriever.search_formatted(latest_query, top_k=5)

    logger.debug(
        "Retrieved context: mem0=%d chars, kb=%d chars",
        len(mem0_context),
        len(kb_context),
    )

    return {
        "mem0_context": mem0_context,
        "retrieved_context": kb_context,
    }


def save_memory_node(state: AgentState) -> dict:
    """
    Exit node: saves the current interaction to Mem0 for future recall.

    Persists the latest user message and assistant response so Mem0 can
    extract and store key facts for cross-session memory.

    Args:
        state: Current AgentState.

    Returns:
        Empty dict (no state changes needed).
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default_user")

    if len(messages) < 2:
        return {}

    # Get the last user message and assistant response
    last_messages = []
    for msg in messages[-4:]:  # Look at last 4 messages to find the pair
        if isinstance(msg, HumanMessage):
            last_messages.append({"role": "user", "content": msg.content})
        else:
            last_messages.append({"role": "assistant", "content": msg.content})

    if last_messages:
        mem0 = _get_mem0()
        mem0.save_interaction(user_id, last_messages)

    return {}


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


def create_agent_graph() -> StateGraph:
    """
    Build and return the compiled LangGraph workflow.

    Graph structure:
        START
          → retrieve_memory (fetch Mem0 + KB context)
          → supervisor (classify intent, route)
          → [policy_agent | claims_agent | general_agent | fallback_agent | greeting_handler]
          → save_memory (persist to Mem0)
          → END

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Create the graph with our state schema
    workflow = StateGraph(AgentState)

    # ---- Add nodes ---- #
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("policy_agent", policy_agent_node)
    workflow.add_node("claims_agent", claims_agent_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("fallback_agent", fallback_agent_node)
    workflow.add_node("greeting_handler", greeting_handler_node)
    workflow.add_node("save_memory", save_memory_node)

    # ---- Set entry point ---- #
    workflow.set_entry_point("retrieve_memory")

    # ---- Add edges ---- #

    # retrieve_memory → supervisor
    workflow.add_edge("retrieve_memory", "supervisor")

    # supervisor → conditional routing to specialist agents
    workflow.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "policy_agent": "policy_agent",
            "claims_agent": "claims_agent",
            "general_agent": "general_agent",
            "fallback_agent": "fallback_agent",
            "greeting_handler": "greeting_handler",
        },
    )

    # All specialist agents → save_memory
    workflow.add_edge("policy_agent", "save_memory")
    workflow.add_edge("claims_agent", "save_memory")
    workflow.add_edge("general_agent", "save_memory")
    workflow.add_edge("fallback_agent", "save_memory")
    workflow.add_edge("greeting_handler", "save_memory")

    # save_memory → END
    workflow.add_edge("save_memory", END)

    # ---- Compile ---- #
    compiled = workflow.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# --------------------------------------------------------------------------- #
# Convenience function for invoking the graph
# --------------------------------------------------------------------------- #

# Module-level compiled graph (lazy initialization)
_compiled_graph = None


def get_compiled_graph():
    """Get or create the compiled graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


def chat(
    message: str,
    user_id: str = "default_user",
    session_id: str = "",
    history: list | None = None,
) -> dict:
    """
    High-level convenience function to send a message through the agent pipeline.

    Args:
        message: The user's message text.
        user_id: User identifier for Mem0 memory.
        session_id: Session identifier.
        history: Optional list of previous BaseMessage objects for context.

    Returns:
        Dict with keys: response, intent, agent, confidence, sources.
    """
    graph = get_compiled_graph()

    # Build input state
    messages = list(history) if history else []
    messages.append(HumanMessage(content=message))

    input_state = {
        "messages": messages,
        "user_id": user_id,
        "session_id": session_id,
        "intent": "",
        "confidence": 0.0,
        "retrieved_context": "",
        "mem0_context": "",
        "current_agent": "",
        "needs_escalation": False,
        "response": "",
        "sources": [],
        "metadata": {},
    }

    # Invoke the graph
    result = graph.invoke(input_state)

    # Extract the last AI message as the response
    response_text = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
            response_text = msg.content
            break

    return {
        "response": response_text,
        "intent": result.get("intent", "unknown"),
        "agent": result.get("current_agent", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "sources": result.get("sources", []),
        "messages": result.get("messages", []),
    }
