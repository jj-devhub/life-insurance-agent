# git commit: feat(knowledge): add KB entry Pydantic schemas
# Module: knowledge/schemas
"""
Pydantic models for validating knowledge base YAML entries.

Each YAML file is validated against the KBEntry model to ensure
consistent structure across the entire knowledge base.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KBEntry(BaseModel):
    """
    A single knowledge base entry loaded from a YAML file.

    This is the core data model that all KB YAML files are validated against.
    The 'details' field is intentionally flexible (dict[str, Any]) to support
    different content structures across categories (policies, claims, etc.).
    """

    category: str = Field(
        ...,
        description="Category of the entry: policy_types, claims, eligibility, benefits, faq",
    )
    topic: str = Field(
        ...,
        description="Human-readable topic title",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for search and retrieval",
    )
    summary: str = Field(
        ...,
        description="Brief summary of the topic (1-3 sentences)",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed information — structure varies by category",
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="List of related topic identifiers",
    )

    # --- Metadata (set by loader, not from YAML) --- #
    source_file: str | None = Field(
        default=None,
        description="Relative path to the source YAML file",
    )

    def to_indexable_chunks(self) -> list[dict[str, str]]:
        """
        Convert this KB entry into chunks suitable for vector store indexing.

        Returns a list of dicts, each with 'content' (text to embed) and
        'metadata' (metadata dict for filtering).

        Strategy:
            - Chunk 1: summary (always present, high-level)
            - Chunk 2+: flattened details sections
        """
        chunks: list[dict[str, str]] = []
        base_metadata = {
            "category": self.category,
            "topic": self.topic,
            "source_file": self.source_file or "",
            "keywords": ", ".join(self.keywords),
        }

        # Chunk 1: Summary
        chunks.append(
            {
                "content": f"Topic: {self.topic}\n\n{self.summary}",
                "metadata": {**base_metadata, "chunk_type": "summary"},
            }
        )

        # Chunk 2+: Flatten details into readable text sections
        details_text = self._flatten_details(self.details)
        if details_text:
            # Split large details into ~1000 char chunks to avoid exceeding token limits
            for i, section in enumerate(self._split_text(details_text, max_chars=1500)):
                chunks.append(
                    {
                        "content": f"Topic: {self.topic}\n\n{section}",
                        "metadata": {**base_metadata, "chunk_type": f"details_{i}"},
                    }
                )

        return chunks

    def _flatten_details(self, obj: Any, prefix: str = "") -> str:
        """Recursively flatten a nested dict/list into readable text."""
        lines: list[str] = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                label = key.replace("_", " ").title()
                if isinstance(value, str):
                    lines.append(f"{prefix}{label}: {value.strip()}")
                elif isinstance(value, list):
                    lines.append(f"{prefix}{label}:")
                    for item in value:
                        if isinstance(item, str):
                            lines.append(f"{prefix}  - {item}")
                        elif isinstance(item, dict):
                            # Named items (e.g., riders, types)
                            item_text = self._flatten_details(item, prefix=prefix + "  ")
                            lines.append(item_text)
                elif isinstance(value, dict):
                    lines.append(f"{prefix}{label}:")
                    lines.append(self._flatten_details(value, prefix=prefix + "  "))
                else:
                    lines.append(f"{prefix}{label}: {value}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    lines.append(f"{prefix}- {item}")
                elif isinstance(item, dict):
                    lines.append(self._flatten_details(item, prefix=prefix + "  "))
        elif isinstance(obj, str):
            lines.append(f"{prefix}{obj.strip()}")
        else:
            lines.append(f"{prefix}{obj}")

        return "\n".join(lines)

    @staticmethod
    def _split_text(text: str, max_chars: int = 1500) -> list[str]:
        """Split text into chunks at paragraph boundaries, respecting max_chars."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]
