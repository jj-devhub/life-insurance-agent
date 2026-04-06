# git commit: feat(agents): initialize LangGraph agents package
# Module: agents
"""
LangGraph multi-agent workflow for life insurance support.

Components:
    - state.py          : AgentState TypedDict shared across all nodes
    - supervisor.py     : Intent classification and routing node
    - policy_agent.py   : Life insurance policy information specialist
    - claims_agent.py   : Claims filing and status specialist
    - general_agent.py  : General FAQ agent with RAG
    - fallback_agent.py : Out-of-scope handling and escalation
    - graph.py          : Main LangGraph StateGraph assembly
"""

from src.agents.graph import create_agent_graph

__all__ = ["create_agent_graph"]
