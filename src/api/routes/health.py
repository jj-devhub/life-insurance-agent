# git commit: feat(api): add health check route
# Module: api/routes/health
"""
Health check API route.

Endpoint:
    GET /api/v1/health - Application health check with dependency status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.models import HealthResponse
from src.config import get_settings
from src.knowledge.loader import KnowledgeBaseLoader
from src.knowledge.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Application health check.

    Returns the overall status of the application and its dependencies:
    LLM provider, Mem0 status, and knowledge base indexing status.
    """
    settings = get_settings()

    # Check KB status
    kb_indexed = False
    kb_entries = 0
    try:
        retriever = KnowledgeRetriever()
        kb_indexed = retriever.is_collection_ready()
        loader = KnowledgeBaseLoader(settings.kb_path)
        entries = loader.load_all()
        kb_entries = len(entries)
    except Exception as e:
        logger.warning("Health check KB error: %s", e)

    overall_status = "ok" if kb_indexed else "degraded"

    return HealthResponse(
        status=overall_status,
        app_name=settings.app_name,
        version="1.0.0",
        llm_provider=settings.llm_provider.value,
        mem0_enabled=settings.mem0_enabled,
        kb_indexed=kb_indexed,
        kb_entries=kb_entries,
    )
