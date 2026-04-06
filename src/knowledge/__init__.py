# git commit: feat(knowledge): initialize knowledge base package
# Module: knowledge
"""
Knowledge base management: YAML loading, Qdrant indexing, semantic retrieval.

Components:
    - loader.py    : Reads and validates YAML files from knowledge_base/
    - schemas.py   : Pydantic models for KB entry validation
    - indexer.py   : Indexes KB entries into Qdrant vector store
    - retriever.py : Semantic search over indexed KB content
"""

from src.knowledge.loader import KnowledgeBaseLoader
from src.knowledge.retriever import KnowledgeRetriever

__all__ = ["KnowledgeBaseLoader", "KnowledgeRetriever"]
