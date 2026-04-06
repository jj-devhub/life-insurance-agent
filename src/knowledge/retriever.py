# git commit: feat(knowledge): add semantic retriever for KB search
# Module: knowledge/retriever
"""
Semantic search retriever for the life insurance knowledge base.

Queries the Qdrant vector store to find the most relevant KB content
for a given user query. Supports category filtering and configurable
result limits.
"""

from __future__ import annotations

import logging

from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge base content using semantic search over Qdrant.

    Usage:
        retriever = KnowledgeRetriever()
        results = retriever.search("What is term life insurance?")
        for r in results:
            print(r["content"], r["score"])
    """

    def __init__(self) -> None:
        """Initialize the retriever with Qdrant client and embedding function."""
        self.settings = get_settings()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Qdrant client and embedding function."""
        from qdrant_client import QdrantClient

        storage_path = self.settings.qdrant_storage_path
        self.client = QdrantClient(path=str(storage_path))
        self.collection_name = self.settings.qdrant_collection_name

        # Initialize embeddings based on provider
        if self.settings.llm_provider == LLMProvider.OLLAMA:
            from langchain_ollama import OllamaEmbeddings

            self.embeddings = OllamaEmbeddings(
                model=self.settings.ollama_embedding_model,
                base_url=self.settings.ollama_base_url,
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                openai_api_key=self.settings.openai_api_key,
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Search the knowledge base for content relevant to the query.

        Args:
            query: The user's question or search text.
            top_k: Maximum number of results to return.
            category: Optional category filter (e.g., "policy_types", "claims").
            score_threshold: Minimum similarity score (0–1) to include results.

        Returns:
            List of dicts with keys: content, score, topic, category, chunk_type, source_file.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            # Generate query embedding
            query_vector = self.embeddings.embed_query(query)

            # Build optional category filter
            query_filter = None
            if category:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category),
                        )
                    ]
                )

            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

            # Format results
            formatted = []
            for result in results:
                payload = result.payload or {}
                formatted.append(
                    {
                        "content": payload.get("content", ""),
                        "score": round(result.score, 4),
                        "topic": payload.get("topic", ""),
                        "category": payload.get("category", ""),
                        "chunk_type": payload.get("chunk_type", ""),
                        "source_file": payload.get("source_file", ""),
                        "keywords": payload.get("keywords", ""),
                    }
                )

            logger.debug(
                "Search for '%s' returned %d results (top score: %s)",
                query[:50],
                len(formatted),
                formatted[0]["score"] if formatted else "N/A",
            )
            return formatted

        except Exception as e:
            logger.error("Knowledge base search failed: %s", e)
            return []

    def search_formatted(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> str:
        """
        Search and return results as a formatted context string for LLM injection.

        Args:
            query: The user's question.
            top_k: Maximum results.
            category: Optional category filter.

        Returns:
            Formatted string with relevant KB content, ready for prompt injection.
        """
        results = self.search(query, top_k=top_k, category=category)

        if not results:
            return "No relevant information found in the knowledge base."

        sections = []
        for i, r in enumerate(results, 1):
            sections.append(
                f"[Source {i}: {r['topic']} ({r['category']}) — relevance: {r['score']}]\n"
                f"{r['content']}"
            )

        return "\n\n---\n\n".join(sections)

    def is_collection_ready(self) -> bool:
        """Check if the Qdrant collection exists and has data."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count > 0
        except Exception:
            return False
