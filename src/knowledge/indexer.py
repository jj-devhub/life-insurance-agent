# git commit: feat(knowledge): add Qdrant vector store indexer
# Module: knowledge/indexer
"""
Indexes knowledge base entries into a Qdrant vector store for semantic search.

Supports both OpenAI and Ollama embedding models. Provides full re-indexing
and a CLI command for manual re-indexing after KB edits.
"""

from __future__ import annotations

import logging
from uuid import NAMESPACE_DNS, uuid5

from src.config import LLMProvider, get_settings
from src.knowledge.loader import KnowledgeBaseLoader

logger = logging.getLogger(__name__)


def _get_embedding_function():
    """
    Return the appropriate embedding function based on the configured LLM provider.

    Returns:
        A LangChain Embeddings instance (OpenAI or Ollama).
    """
    settings = get_settings()

    if settings.llm_provider == LLMProvider.OLLAMA:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )


def _get_qdrant_client():
    """
    Return a Qdrant client configured for local file-based storage.

    Creates the storage directory if it doesn't exist.
    """
    from qdrant_client import QdrantClient

    settings = get_settings()
    storage_path = settings.qdrant_storage_path
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def _generate_chunk_id(topic: str, chunk_type: str) -> str:
    """Generate a deterministic ID for a KB chunk based on topic and chunk type."""
    return str(uuid5(NAMESPACE_DNS, f"{topic}::{chunk_type}"))


class KnowledgeBaseIndexer:
    """
    Indexes KB entries into Qdrant for semantic retrieval.

    Usage:
        indexer = KnowledgeBaseIndexer()
        indexer.index_all()
    """

    def __init__(self) -> None:
        """Initialize indexer with settings, embedding function, and Qdrant client."""
        self.settings = get_settings()
        self.embeddings = _get_embedding_function()
        self.client = _get_qdrant_client()
        self.collection_name = self.settings.qdrant_collection_name

    def index_all(self, force: bool = False) -> int:
        """
        Load all KB entries and index them into Qdrant.

        Args:
            force: If True, recreate the collection from scratch.

        Returns:
            Number of chunks indexed.
        """
        from qdrant_client.models import Distance, VectorParams

        # Load KB entries
        loader = KnowledgeBaseLoader(self.settings.kb_path)
        entries = loader.load_all()

        if not entries:
            logger.warning("No KB entries to index.")
            return 0

        # Prepare chunks
        all_chunks = []
        for entry in entries:
            chunks = entry.to_indexable_chunks()
            all_chunks.extend(chunks)

        logger.info("Prepared %d chunks from %d KB entries", len(all_chunks), len(entries))

        # Recreate collection if forced or doesn't exist
        collections = [c.name for c in self.client.get_collections().collections]

        if force or self.collection_name not in collections:
            if self.collection_name in collections:
                self.client.delete_collection(self.collection_name)
                logger.info("Deleted existing collection: %s", self.collection_name)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.qdrant_embedding_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection: %s", self.collection_name)

        # Generate embeddings and upsert
        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]

        logger.info("Generating embeddings for %d chunks...", len(texts))
        vectors = self.embeddings.embed_documents(texts)

        # Build Qdrant points
        from qdrant_client.models import PointStruct

        points = []
        for i, (text, metadata, vector) in enumerate(zip(texts, metadatas, vectors)):
            point_id = _generate_chunk_id(metadata["topic"], metadata["chunk_type"])
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": text,
                        **metadata,
                    },
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(
                "Upserted batch %d/%d",
                i // batch_size + 1,
                (len(points) + batch_size - 1) // batch_size,
            )

        logger.info(
            "Successfully indexed %d chunks into Qdrant collection '%s'",
            len(points),
            self.collection_name,
        )
        return len(points)

    def get_collection_info(self) -> dict:
        """Return information about the current Qdrant collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "points_count": 0,
                "status": "not_found",
            }

    def delete_collection(self) -> None:
        """Delete the Qdrant collection."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection: %s", self.collection_name)


# --------------------------------------------------------------------------- #
# CLI entry point for: python -m src.knowledge.indexer
# --------------------------------------------------------------------------- #


def cli_index():
    """CLI command to index the knowledge base."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  Life Insurance KB Indexer")
    print("=" * 60)

    force = "--force" in sys.argv

    try:
        indexer = KnowledgeBaseIndexer()
        count = indexer.index_all(force=force)
        info = indexer.get_collection_info()

        print(f"\n✅ Indexed {count} chunks successfully!")
        print(f"   Collection: {info['collection_name']}")
        print(f"   Total points: {info['points_count']}")
        print(f"   Status: {info['status']}")
    except Exception as e:
        print(f"\n❌ Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_index()
