# git commit: feat(agents): add fallback and greeting handler nodes
# Module: agents/fallback_agent
"""
Fallback and Greeting handler nodes for the LangGraph workflow.

Fallback Agent: politely redirects out-of-scope questions back to life insurance.
Greeting Handler: responds to greetings, thanks, and goodbyes.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.agents.state import AgentState
from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

FALLBACK_SYSTEM_PROMPT = (
    "You are a Life Insurance Support Assistant. The user has asked a question\n"
    "that is outside your area of expertise (life insurance).\n\n"
    "GUIDELINES:\n"
    "1. Politely acknowledge the user's question.\n"
    "2. Explain that you specialize in life insurance topics only.\n"
    "3. Suggest 2-3 life insurance topics they might be interested in, such as:\n"
    "   - Understanding different types of life insurance (term, whole, universal)\n"
    "   - How to determine how much coverage they need\n"
    "   - The claims filing process\n"
    "   - Eligibility and health requirements\n"
    "   - Policy riders and additional benefits\n"
    "4. Be warm and helpful, not dismissive.\n"
    "5. Keep it brief — 2-3 sentences plus the suggestions."
)

GREETING_SYSTEM_PROMPT = (
    "You are a friendly Life Insurance Support Assistant.\n"
    "Respond warmly to the user's greeting or social message.\n\n"
    "If it's a greeting: Welcome them, introduce yourself briefly as a life insurance\n"
    "support assistant, and ask how you can help with their life insurance questions.\n\n"
    "If it's a thank you: Express that you're happy to help and invite further questions.\n\n"
    "If it's a goodbye: Wish them well and remind them you're available anytime for\n"
    "life insurance questions.\n\n"
    "Keep responses concise and warm. Mention 1-2 specific topics you can help with."
)


def _get_llm():
    """Get the configured LLM instance."""
    settings = get_settings()
    if settings.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7,
        )


def fallback_agent_node(state: AgentState) -> dict:
    """
    Fallback Agent: redirects out-of-scope questions to life insurance topics.

    Args:
        state: Current AgentState.

    Returns:
        Dict with updated messages.
    """
    messages = state.get("messages", [])
    llm = _get_llm()

    try:
        llm_messages = [SystemMessage(content=FALLBACK_SYSTEM_PROMPT)]
        recent = messages[-4:] if len(messages) > 4 else messages
        llm_messages.extend(recent)

        result = llm.invoke(llm_messages)
        logger.info("Fallback Agent redirected out-of-scope query")

        return {
            "messages": [AIMessage(content=result.content)],
            "current_agent": "fallback_agent",
        }

    except Exception as e:
        logger.error("Fallback Agent error: %s", e)
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I appreciate your question! However, I specialize in "
                        "life insurance topics. "
                        "I can help you with:\n"
                        "• Understanding policy types (term, whole, universal life)\n"
                        "• Coverage amounts and eligibility\n"
                        "• Claims filing process\n\n"
                        "What would you like to know about life insurance?"
                    )
                )
            ],
            "current_agent": "fallback_agent",
        }


def greeting_handler_node(state: AgentState) -> dict:
    """
    Greeting Handler: responds to social messages (hello, thanks, goodbye).

    Args:
        state: Current AgentState.

    Returns:
        Dict with updated messages.
    """
    messages = state.get("messages", [])
    llm = _get_llm()

    try:
        llm_messages = [SystemMessage(content=GREETING_SYSTEM_PROMPT)]
        recent = messages[-2:] if len(messages) > 2 else messages
        llm_messages.extend(recent)

        result = llm.invoke(llm_messages)
        logger.info("Greeting handler responded")

        return {
            "messages": [AIMessage(content=result.content)],
            "current_agent": "greeting_handler",
        }

    except Exception as e:
        logger.error("Greeting handler error: %s", e)
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Hello! 👋 I'm your Life Insurance Support Assistant. "
                        "I can help you understand policy types, coverage options, "
                        "eligibility requirements, claims processes, and more. "
                        "What would you like to know about life insurance?"
                    )
                )
            ],
            "current_agent": "greeting_handler",
        }
