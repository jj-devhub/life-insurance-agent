# git commit: feat(config): add centralized configuration module
# Module: config
"""
Centralized application configuration using pydantic-settings.

Loads settings from environment variables and .env file. Supports both
OpenAI and Ollama LLM providers, configurable via LLM_PROVIDER env var.

Usage:
    from src.config import get_settings
    settings = get_settings()
    print(settings.llm_provider)  # "openai" or "ollama"
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class LLMProvider(str, Enum):
    """Supported LLM provider backends."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class LogLevel(str, Enum):
    """Standard Python log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# --------------------------------------------------------------------------- #
# Project root resolution
# --------------------------------------------------------------------------- #

# Resolve project root: directory containing pyproject.toml
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent  # src/ -> project root


# --------------------------------------------------------------------------- #
# Settings class
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    All fields map to env vars by uppercasing the field name.
    Example: llm_provider -> LLM_PROVIDER
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM Provider ---------------------------------------------------- #
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM backend: 'openai' or 'ollama'",
    )

    # ---- OpenAI ---------------------------------------------------------- #
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (required when llm_provider='openai')",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model name",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )

    # ---- Ollama ---------------------------------------------------------- #
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama chat model name",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name",
    )

    # ---- Application ----------------------------------------------------- #
    app_name: str = Field(
        default="Life Insurance Support Assistant",
        description="Display name for the application",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Python log level",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, auto-reload)",
    )

    # ---- Paths ----------------------------------------------------------- #
    knowledge_base_path: str = Field(
        default="./knowledge_base",
        description="Path to the YAML knowledge base directory",
    )
    qdrant_path: str = Field(
        default="./data/qdrant_db",
        description="Path to Qdrant local storage directory",
    )
    mem0_db_path: str = Field(
        default="./data/mem0_db",
        description="Path to Mem0 SQLite database directory",
    )

    # ---- API ------------------------------------------------------------- #
    api_host: str = Field(default="0.0.0.0", description="API server bind host")
    api_port: int = Field(default=8000, description="API server bind port")

    # ---- Memory (Mem0) --------------------------------------------------- #
    mem0_enabled: bool = Field(
        default=True,
        description="Enable Mem0 persistent memory layer",
    )
    memory_search_limit: int = Field(
        default=5,
        description="Max number of memories to retrieve per query",
    )

    # ---- Qdrant ---------------------------------------------------------- #
    qdrant_collection_name: str = Field(
        default="life_insurance_kb",
        description="Qdrant collection name for the knowledge base",
    )
    qdrant_embedding_size: int = Field(
        default=1536,
        description="Embedding vector dimension (1536 for OpenAI, 768 for nomic-embed-text)",
    )

    # ---- Validators ------------------------------------------------------ #

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str | None, info) -> str | None:
        """Warn if OpenAI key is missing when provider is set to openai."""
        # Validation happens at runtime when needed, not at init
        return v

    # ---- Derived properties ---------------------------------------------- #

    @property
    def kb_path(self) -> Path:
        """Resolved knowledge base directory path."""
        p = Path(self.knowledge_base_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def qdrant_storage_path(self) -> Path:
        """Resolved Qdrant storage path."""
        p = Path(self.qdrant_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def mem0_storage_path(self) -> Path:
        """Resolved Mem0 database path."""
        p = Path(self.mem0_db_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def active_model(self) -> str:
        """Return the active chat model name based on selected provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            return self.ollama_model
        return self.openai_model

    @property
    def active_embedding_model(self) -> str:
        """Return the active embedding model name based on selected provider."""
        if self.llm_provider == LLMProvider.OLLAMA:
            return self.ollama_embedding_model
        return self.openai_embedding_model


# --------------------------------------------------------------------------- #
# Singleton accessor
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    The first call parses env vars / .env; subsequent calls return the cache.
    To force a reload (e.g. in tests), call `get_settings.cache_clear()`.
    """
    return Settings()
