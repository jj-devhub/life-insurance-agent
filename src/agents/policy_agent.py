# git commit: feat(agents): add policy specialist agent node
# Module: agents/policy_agent
"""
Policy Agent node for the LangGraph workflow.

Handles questions about life insurance policy types, features, comparisons,
eligibility, benefits, coverage amounts, and premiums. Uses RAG over the
knowledge base and Mem0 context for personalized responses.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.agents.state import AgentState
from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

POLICY_AGENT_SYSTEM_PROMPT = (
    "You are a knowledgeable Life Insurance Policy Specialist "
    "for a US-based insurance support service.\n\n"
    "Your expertise covers:\n"
    "- Policy types: Term Life, Whole Life, Universal Life (IUL, GUL, VUL), Variable Life\n"
    "- Policy features, advantages, disadvantages, and comparisons\n"
    "- Eligibility: age requirements, health factors, underwriting process\n"
    "- Benefits: death benefit, cash value, riders, tax advantages\n"
    "- Coverage amounts and premium ranges\n"
    "- Top US insurance providers\n\n"
    "GUIDELINES:\n"
    "1. Provide accurate, detailed information grounded in the knowledge base context.\n"
    "2. When comparing policies, create clear, structured comparisons.\n"
    "3. If the user has mentioned personal details (from memory context), tailor your response.\n"
    "4. Always mention that specific rates and availability vary by insurer.\n"
    "5. Use clear, professional language that a non-expert can understand.\n"
    "6. If you don't have enough information to answer accurately, say so honestly.\n"
    "7. Never provide specific financial advice — recommend consulting a licensed agent.\n"
    "8. Format responses with bullet points and sections for readability.\n\n"
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
            temperature=0.3,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )


def policy_agent_node(state: AgentState) -> dict:
    """
    Policy Agent: answers policy-related questions using knowledge base + memory.

    Retrieves relevant KB content, injects Mem0 context, and generates
    a comprehensive response about life insurance policies.

    Args:
        state: Current AgentState with messages, retrieved_context, mem0_context.

    Returns:
        Dict with updated messages (appended AIMessage) and sources.
    """
    messages = state.get("messages", [])
    kb_context = state.get("retrieved_context", "No knowledge base context available.")
    mem0_context = state.get("mem0_context", "")

    # Build memory context section
    memory_section = ""
    if mem0_context:
        memory_section = f"\nUSER MEMORY (from previous conversations):\n{mem0_context}\n"

    # Format system prompt with context
    system_prompt = POLICY_AGENT_SYSTEM_PROMPT.format(
        kb_context=kb_context,
        memory_context=memory_section,
    )

    llm = _get_llm()

    try:
        # Build message list for LLM: system + recent conversation + current question
        llm_messages = [SystemMessage(content=system_prompt)]

        # Add recent conversation history (last 8 messages for context window)
        recent = messages[-8:] if len(messages) > 8 else messages
        llm_messages.extend(recent)

        result = llm.invoke(llm_messages)

        logger.info("Policy Agent generated response (%d chars)", len(result.content))

        return {
            "messages": [AIMessage(content=result.content)],
            "current_agent": "policy_agent",
        }

    except Exception as e:
        logger.error("Policy Agent error: %s", e)
        error_msg = (
            "I apologize, but I'm having trouble processing your policy question right now. "
            "Please try again, or contact a licensed insurance agent for immediate assistance."
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_agent": "policy_agent",
        }
