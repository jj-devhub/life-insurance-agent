# git commit: feat(api): add knowledge base and memory routes
# Module: api/routes/knowledge
"""
Knowledge Base and Memory API route handlers.

Endpoints:
    GET    /api/v1/knowledge-base          - List all KB entries
    GET    /api/v1/knowledge-base/{category} - Get entries by category
    POST   /api/v1/knowledge-base/reload   - Reload and re-index KB
    GET    /api/v1/memory/{user_id}        - Get user memories
    DELETE /api/v1/memory/{user_id}        - Clear user memories
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.models import (
    KBEntryResponse,
    KBListResponse,
    KBReloadResponse,
    MemoryDeleteResponse,
    MemoryItem,
    MemoryResponse,
)
from src.config import get_settings
from src.knowledge.indexer import KnowledgeBaseIndexer
from src.knowledge.loader import KnowledgeBaseLoader
from src.memory.mem0_manager import Mem0Manager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Knowledge Base routes
# --------------------------------------------------------------------------- #

kb_router = APIRouter(prefix="/api/v1/knowledge-base", tags=["Knowledge Base"])


@kb_router.get("", response_model=KBListResponse)
async def list_knowledge_base() -> KBListResponse:
    """List all entries in the knowledge base with their metadata."""
    settings = get_settings()

    try:
        loader = KnowledgeBaseLoader(settings.kb_path)
        entries = loader.load_all()

        entry_responses = [
            KBEntryResponse(
                category=e.category,
                topic=e.topic,
                keywords=e.keywords,
                summary=e.summary.strip(),
                source_file=e.source_file,
                related_topics=e.related_topics,
            )
            for e in entries
        ]

        categories = sorted(set(e.category for e in entries))

        return KBListResponse(
            total_entries=len(entries),
            categories=categories,
            entries=entry_responses,
        )
    except Exception as e:
        logger.error("Failed to list KB: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.get("/{category}", response_model=KBListResponse)
async def get_kb_by_category(category: str) -> KBListResponse:
    """Get knowledge base entries filtered by category."""
    settings = get_settings()

    try:
        loader = KnowledgeBaseLoader(settings.kb_path)
        entries = loader.load_by_category(category)

        if not entries:
            raise HTTPException(
                status_code=404,
                detail=f"No entries found for category '{category}'. "
                f"Available: {', '.join(loader.get_categories())}",
            )

        entry_responses = [
            KBEntryResponse(
                category=e.category,
                topic=e.topic,
                keywords=e.keywords,
                summary=e.summary.strip(),
                source_file=e.source_file,
                related_topics=e.related_topics,
            )
            for e in entries
        ]

        return KBListResponse(
            total_entries=len(entries),
            categories=[category],
            entries=entry_responses,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get KB category '%s': %s", category, e)
        raise HTTPException(status_code=500, detail=str(e))


@kb_router.post("/reload", response_model=KBReloadResponse)
async def reload_knowledge_base() -> KBReloadResponse:
    """
    Reload and re-index the knowledge base from YAML files.

    Use this after editing YAML files in the knowledge_base/ directory.
    This will re-read all files, validate them, and re-index into Qdrant.
    """
    try:
        # Re-index with force=True to rebuild the collection
        indexer = KnowledgeBaseIndexer()
        chunks_count = indexer.index_all(force=True)

        # Get entry count
        settings = get_settings()
        loader = KnowledgeBaseLoader(settings.kb_path)
        entries = loader.load_all()

        return KBReloadResponse(
            success=True,
            entries_loaded=len(entries),
            chunks_indexed=chunks_count,
            message=(
                f"Successfully reloaded {len(entries)} entries and indexed {chunks_count} chunks."
            ),
        )
    except Exception as e:
        logger.error("KB reload failed: %s", e)
        return KBReloadResponse(
            success=False,
            entries_loaded=0,
            chunks_indexed=0,
            message=f"Reload failed: {str(e)}",
        )


# --------------------------------------------------------------------------- #
# Memory routes
# --------------------------------------------------------------------------- #

memory_router = APIRouter(prefix="/api/v1/memory", tags=["Memory"])


@memory_router.get("/{user_id}", response_model=MemoryResponse)
async def get_user_memories(user_id: str) -> MemoryResponse:
    """Get all stored memories for a user from Mem0."""
    mem0 = Mem0Manager()

    if not mem0.is_available:
        return MemoryResponse(user_id=user_id, memories=[], total=0)

    memories_raw = mem0.get_all_memories(user_id)
    memories = [
        MemoryItem(
            id=m.get("id"),
            memory=m.get("memory", ""),
            created_at=m.get("created_at"),
        )
        for m in memories_raw
    ]

    return MemoryResponse(user_id=user_id, memories=memories, total=len(memories))


@memory_router.delete("/{user_id}", response_model=MemoryDeleteResponse)
async def clear_user_memories(user_id: str) -> MemoryDeleteResponse:
    """Delete all memories for a user (GDPR compliance)."""
    mem0 = Mem0Manager()

    if not mem0.is_available:
        return MemoryDeleteResponse(
            user_id=user_id,
            success=False,
            message="Mem0 is not available or disabled.",
        )

    success = mem0.clear_memories(user_id)
    return MemoryDeleteResponse(
        user_id=user_id,
        success=success,
        message="All memories cleared." if success else "Failed to clear memories.",
    )
