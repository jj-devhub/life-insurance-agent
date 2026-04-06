# git commit: feat(agents): add general FAQ agent node with RAG
# Module: agents/general_agent
"""
General FAQ Agent node for the LangGraph workflow.

Handles general life insurance questions using RAG over the knowledge base.
Covers topics like premium payments, policy changes, basic "what is" questions,
and other general inquiries.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.agents.state import AgentState
from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

GENERAL_AGENT_SYSTEM_PROMPT = (
    "You are a friendly Life Insurance Support Assistant for a US-based service.\n\n"
    "You help users with general life insurance questions including:\n"
    "- What is life insurance and how does it work\n"
    "- Who needs life insurance and when to buy\n"
    "- How much coverage is needed\n"
    "- Premium payment methods and frequencies\n"
    "- Policy changes (beneficiary updates, coverage adjustments, conversions)\n"
    "- Policy cancellation and reinstatement\n"
    "- General terminology and concepts\n\n"
    "GUIDELINES:\n"
    "1. Be friendly, approachable, and use simple language.\n"
    "2. Ground your answers in the knowledge base context provided below.\n"
    "3. Use bullet points and clear formatting for readability.\n"
    "4. Provide practical, actionable advice.\n"
    "5. When appropriate, mention relevant policy types the user might want to explore.\n"
    "6. Never provide specific financial or legal advice — recommend professionals.\n"
    "7. If the knowledge base doesn't have enough information, be honest about it.\n"
    "8. Cite the relevant topic when referencing specific KB content.\n\n"
    "{memory_context}\n\n"
    "KNOWLEDGE BASE CONTEXT:\n"
    "{kb_context}"
)


def _get_llm():
    """Get the configured LLM instance."""
    settings = get_settings()
    if settings.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.4,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.4,
        )


def general_agent_node(state: AgentState) -> dict:
    """
    General FAQ Agent: answers broad life insurance questions via RAG.

    Args:
        state: Current AgentState.

    Returns:
        Dict with updated messages and agent identifier.
    """
    messages = state.get("messages", [])
    kb_context = state.get("retrieved_context", "No knowledge base context available.")
    mem0_context = state.get("mem0_context", "")

    memory_section = ""
    if mem0_context:
        memory_section = f"\nUSER MEMORY (from previous conversations):\n{mem0_context}\n"

    system_prompt = GENERAL_AGENT_SYSTEM_PROMPT.format(
        kb_context=kb_context,
        memory_context=memory_section,
    )

    llm = _get_llm()

    try:
        llm_messages = [SystemMessage(content=system_prompt)]
        recent = messages[-8:] if len(messages) > 8 else messages
        llm_messages.extend(recent)

        result = llm.invoke(llm_messages)
        logger.info("General Agent generated response (%d chars)", len(result.content))

        return {
            "messages": [AIMessage(content=result.content)],
            "current_agent": "general_agent",
        }

    except Exception as e:
        logger.error("General Agent error: %s", e)
        error_msg = (
            "I apologize, but I'm having trouble answering your question right now. "
            "Please try rephrasing your question, or ask about a specific life insurance topic."
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_agent": "general_agent",
        }
