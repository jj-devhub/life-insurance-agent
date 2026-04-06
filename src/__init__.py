# Life Insurance Support Assistant
# git commit: feat(init): initialize src package
# Module: project-foundation
"""
Life Insurance Support Assistant - AI-powered chat agent for life insurance inquiries.

Architecture:
    - src/agents/    : LangGraph multi-agent workflow (supervisor + specialist agents)
    - src/api/       : FastAPI backend with REST endpoints
    - src/cli/       : Rich + Typer interactive CLI chat interface
    - src/knowledge/ : YAML knowledge base loader, Qdrant indexer, semantic retriever
    - src/memory/    : Mem0 persistent memory + session state management
    - src/config.py  : Centralized configuration via pydantic-settings
"""
