# git commit: feat(memory): add Mem0 persistent memory manager
# Module: memory/mem0_manager
"""
Mem0 persistent memory manager for cross-session user memory.

Provides search, save, retrieve, and clear operations for user memories.
Gracefully degrades if Mem0 is disabled via configuration. Uses local
storage (SQLite + Qdrant) for fully self-contained operation.
"""

from __future__ import annotations

import logging

from src.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


class Mem0Manager:
    """
    Manages persistent user memory using Mem0 (open-source).

    Mem0 automatically extracts and stores key facts from conversations,
    enabling the agent to remember user details (policy numbers, preferences,
    past inquiries) across multiple sessions.

    Usage:
        manager = Mem0Manager()
        manager.save_interaction("user123", [
            {"role": "user", "content": "My policy number is ABC123"},
            {"role": "assistant", "content": "Got it, I'll remember that."}
        ])
        memories = manager.search_memories("user123", "policy number")
    """

    def __init__(self) -> None:
        """Initialize Mem0 with local configuration."""
        self.settings = get_settings()
        self.enabled = self.settings.mem0_enabled
        self._memory = None

        if self.enabled:
            self._initialize_mem0()

    def _initialize_mem0(self) -> None:
        """Set up Mem0 with the appropriate LLM and embedding provider."""
        try:
            from mem0 import Memory

            # Build Mem0 config based on LLM provider
            config = {
                "version": "v1.1",
            }

            # Configure LLM
            if self.settings.llm_provider == LLMProvider.OLLAMA:
                config["llm"] = {
                    "provider": "ollama",
                    "config": {
                        "model": self.settings.ollama_model,
                        "ollama_base_url": self.settings.ollama_base_url,
                    },
                }
                config["embedder"] = {
                    "provider": "ollama",
                    "config": {
                        "model": self.settings.ollama_embedding_model,
                        "ollama_base_url": self.settings.ollama_base_url,
                    },
                }
            else:
                config["llm"] = {
                    "provider": "openai",
                    "config": {
                        "model": self.settings.openai_model,
                        "api_key": self.settings.openai_api_key,
                    },
                }
                config["embedder"] = {
                    "provider": "openai",
                    "config": {
                        "model": self.settings.openai_embedding_model,
                        "api_key": self.settings.openai_api_key,
                    },
                }

            # Use Qdrant for vector storage (separate from KB collection)
            mem0_storage = self.settings.mem0_storage_path
            mem0_storage.mkdir(parents=True, exist_ok=True)

            config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_memories",
                    "path": str(mem0_storage / "qdrant"),
                    "embedding_model_dims": self.settings.qdrant_embedding_size,
                },
            }

            # History database
            config["history_db_path"] = str(mem0_storage / "history.db")

            self._memory = Memory.from_config(config)
            logger.info(
                "Mem0 initialized successfully (provider: %s)", self.settings.llm_provider.value
            )

        except ImportError:
            logger.error("mem0ai package not installed. Run: pip install mem0ai")
            self.enabled = False
        except Exception as e:
            logger.error("Failed to initialize Mem0: %s", e)
            self.enabled = False

    def search_memories(self, user_id: str, query: str) -> str:
        """
        Search for relevant memories for a user.

        Args:
            user_id: Unique user identifier.
            query: The search query (typically the user's current message).

        Returns:
            Formatted string of relevant memories for prompt injection.
            Returns empty string if Mem0 is disabled or no memories found.
        """
        if not self.enabled or self._memory is None:
            return ""

        try:
            results = self._memory.search(
                query=query,
                user_id=user_id,
                limit=self.settings.memory_search_limit,
            )

            if not results or not results.get("results"):
                return ""

            memories = results["results"]
            if not memories:
                return ""

            # Format memories for prompt injection
            memory_lines = []
            for i, mem in enumerate(memories, 1):
                memory_text = mem.get("memory", "")
                if memory_text:
                    memory_lines.append(f"  {i}. {memory_text}")

            if not memory_lines:
                return ""

            formatted = "Relevant information from previous conversations:\n" + "\n".join(
                memory_lines
            )
            logger.debug("Retrieved %d memories for user '%s'", len(memory_lines), user_id)
            return formatted

        except Exception as e:
            logger.warning("Mem0 search failed for user '%s': %s", user_id, e)
            return ""

    def save_interaction(self, user_id: str, messages: list[dict[str, str]]) -> None:
        """
        Save a conversation interaction to Mem0 for future recall.

        Mem0 automatically extracts key facts and preferences from the
        messages and stores them as discrete memory items.

        Args:
            user_id: Unique user identifier.
            messages: List of message dicts with 'role' and 'content' keys.
        """
        if not self.enabled or self._memory is None:
            return

        try:
            self._memory.add(
                messages=messages,
                user_id=user_id,
            )
            logger.debug(
                "Saved interaction to Mem0 for user '%s' (%d messages)", user_id, len(messages)
            )
        except Exception as e:
            logger.warning("Failed to save interaction to Mem0 for user '%s': %s", user_id, e)

    def get_all_memories(self, user_id: str) -> list[dict]:
        """
        Retrieve all stored memories for a user.

        Args:
            user_id: Unique user identifier.

        Returns:
            List of memory dicts with 'id', 'memory', 'created_at' keys.
        """
        if not self.enabled or self._memory is None:
            return []

        try:
            result = self._memory.get_all(user_id=user_id)
            return result.get("results", []) if isinstance(result, dict) else result
        except Exception as e:
            logger.warning("Failed to get memories for user '%s': %s", user_id, e)
            return []

    def clear_memories(self, user_id: str) -> bool:
        """
        Delete all memories for a user (GDPR compliance).

        Args:
            user_id: Unique user identifier.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled or self._memory is None:
            return False

        try:
            self._memory.delete_all(user_id=user_id)
            logger.info("Cleared all memories for user '%s'", user_id)
            return True
        except Exception as e:
            logger.warning("Failed to clear memories for user '%s': %s", user_id, e)
            return False

    @property
    def is_available(self) -> bool:
        """Check if Mem0 is enabled and properly initialized."""
        return self.enabled and self._memory is not None
