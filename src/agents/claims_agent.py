# git commit: feat(agents): add claims specialist agent node
# Module: agents/claims_agent
"""
Claims Agent node for the LangGraph workflow.

Handles questions about filing life insurance claims, required documents,
claim status tracking, timelines, payment options, and denial remedies.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.agents.state import AgentState
from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

CLAIMS_AGENT_SYSTEM_PROMPT = (
    "You are a Life Insurance Claims Specialist "
    "for a US-based insurance support service.\n\n"
    "Your expertise covers:\n"
    "- Step-by-step claims filing process\n"
    "- Required documents (death certificate, claim forms, etc.)\n"
    "- Claim status tracking and typical timelines\n"
    "- Payment options (lump sum, installments, annuity)\n"
    "- Common reasons for claim denial and how to appeal\n"
    "- State regulations on claim processing timelines\n"
    "- Contestability period rules\n\n"
    "GUIDELINES:\n"
    "1. Provide clear, step-by-step guidance for claims processes.\n"
    "2. Be empathetic — users filing claims are often dealing with loss.\n"
    "3. Use the knowledge base context to provide accurate information.\n"
    "4. When discussing timelines, mention that they vary by insurer and state.\n"
    "5. If the user seems to need urgent help, suggest contacting the insurer.\n"
    "6. Recommend ordering multiple certified copies of the death certificate.\n"
    "7. Mention the NAIC Life Insurance Policy Locator Service when relevant.\n"
    "8. If a claim has been denied, outline the appeals process clearly.\n\n"
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


def claims_agent_node(state: AgentState) -> dict:
    """
    Claims Agent: answers claims-related questions with empathy and accuracy.

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

    system_prompt = CLAIMS_AGENT_SYSTEM_PROMPT.format(
        kb_context=kb_context,
        memory_context=memory_section,
    )

    llm = _get_llm()

    try:
        llm_messages = [SystemMessage(content=system_prompt)]
        recent = messages[-8:] if len(messages) > 8 else messages
        llm_messages.extend(recent)

        result = llm.invoke(llm_messages)
        logger.info("Claims Agent generated response (%d chars)", len(result.content))

        return {
            "messages": [AIMessage(content=result.content)],
            "current_agent": "claims_agent",
        }

    except Exception as e:
        logger.error("Claims Agent error: %s", e)
        error_msg = (
            "I apologize, but I'm having trouble processing your claims question right now. "
            "For urgent claims assistance, please contact your insurance company's claims "
            "department directly — their number is on your policy documents."
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_agent": "claims_agent",
        }
