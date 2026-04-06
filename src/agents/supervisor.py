# git commit: feat(agents): add supervisor intent router node
# Module: agents/supervisor
"""
Supervisor node for the LangGraph workflow.

Classifies user intent and routes to the appropriate specialist agent.
Uses structured LLM output (JSON) for deterministic routing decisions.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Supervisor system prompt
# --------------------------------------------------------------------------- #

SUPERVISOR_SYSTEM_PROMPT = (
    "You are an intent classification system for a Life Insurance Support Assistant.\n\n"
    "Your job is to analyze the user's message and classify it into "
    "one of the following intents:\n\n"
    "INTENTS:\n"
    '1. "policy_inquiry" — Questions about life insurance policy types, features, comparisons,\n'
    "   eligibility, benefits (term life, whole life, universal life, variable life, riders,\n"
    "   cash value, death benefit, coverage amounts, premiums).\n\n"
    '2. "claims_inquiry" — Questions about filing claims, required documents, claim status,\n'
    "   claim process, claim denial, beneficiary claims.\n\n"
    '3. "general_faq" — General questions about life insurance (what it is, do I need it,\n'
    "   how much, premium payments, policy changes, cancellation, reinstatement).\n\n"
    '4. "greeting" — Greetings, introductions, thank you messages, goodbyes.\n\n'
    '5. "out_of_scope" — Questions NOT related to life insurance at all (weather, sports,\n'
    "   cooking, other insurance types like auto/home unless comparing with life insurance).\n\n"
    "RULES:\n"
    '- If the question touches on BOTH policy details AND claims, prefer "policy_inquiry"\n'
    "  unless the primary focus is clearly on the claims process itself.\n"
    "- If the user asks about eligibility, health factors, underwriting, age requirements —\n"
    '  classify as "policy_inquiry".\n'
    "- If unsure between two intents, pick the more specific one with lower confidence.\n"
    "- Consider conversation context from previous messages when classifying.\n\n"
    "RESPOND WITH ONLY valid JSON in this exact format:\n"
    "{\n"
    '    "intent": "<intent_name>",\n'
    '    "confidence": <float 0.0-1.0>,\n'
    '    "reasoning": "<brief explanation>"\n'
    "}"
)


# --------------------------------------------------------------------------- #
# Supervisor node function
# --------------------------------------------------------------------------- #


def _get_llm():
    """Get the appropriate LLM based on configuration."""
    settings = get_settings()

    if settings.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )


def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor node: classifies user intent and determines routing.

    Reads the latest user message (plus conversation context), calls the LLM
    with the classification prompt, and returns the intent + confidence.

    Args:
        state: Current AgentState.

    Returns:
        Dict with 'intent', 'confidence', and 'current_agent' updates.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "intent": "greeting",
            "confidence": 1.0,
            "current_agent": "greeting_handler",
        }

    # Build classification prompt with recent conversation context
    recent_messages = messages[-6:]  # Last 3 turns (6 messages) for context
    context_text = ""
    for msg in recent_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        context_text += f"{role}: {msg.content}\n"

    llm = _get_llm()

    try:
        result = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Conversation context:\n{context_text}\n\n"
                        "Classify the LATEST user message."
                    )
                ),
            ]
        )

        # Parse JSON response
        response_text = result.content.strip()

        # Handle potential markdown code blocks in response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        parsed = json.loads(response_text)
        intent = parsed.get("intent", "general_faq")
        confidence = float(parsed.get("confidence", 0.5))

        # Map intent to agent
        intent_to_agent = {
            "policy_inquiry": "policy_agent",
            "claims_inquiry": "claims_agent",
            "general_faq": "general_agent",
            "greeting": "greeting_handler",
            "out_of_scope": "fallback_agent",
        }

        current_agent = intent_to_agent.get(intent, "general_agent")

        logger.info(
            "Supervisor classified intent='%s' (confidence=%.2f) → %s",
            intent,
            confidence,
            current_agent,
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "current_agent": current_agent,
        }

    except json.JSONDecodeError as e:
        logger.warning("Supervisor failed to parse LLM response as JSON: %s", e)
        return {
            "intent": "general_faq",
            "confidence": 0.3,
            "current_agent": "general_agent",
        }
    except Exception as e:
        logger.error("Supervisor node error: %s", e)
        return {
            "intent": "general_faq",
            "confidence": 0.3,
            "current_agent": "general_agent",
        }


def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge function: routes to the appropriate agent node
    based on the classified intent.

    Args:
        state: Current AgentState (must have 'current_agent' set by supervisor).

    Returns:
        Name of the next node to execute.
    """
    return state.get("current_agent", "general_agent")
