# git commit: feat(knowledge): add YAML knowledge base loader
# Module: knowledge/loader
"""
Loads and validates YAML knowledge base files from the knowledge_base/ directory.

Recursively scans all .yaml files, parses them, validates against the KBEntry
schema, and returns a list of validated entries. Provides clear error messages
for invalid files so non-technical editors can fix issues.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.knowledge.schemas import KBEntry

logger = logging.getLogger(__name__)


class KnowledgeBaseLoader:
    """
    Loads YAML knowledge base files and validates them against the KBEntry schema.

    Usage:
        loader = KnowledgeBaseLoader("./knowledge_base")
        entries = loader.load_all()
        for entry in entries:
            print(entry.topic, entry.category)
    """

    def __init__(self, kb_path: str | Path) -> None:
        """
        Initialize the loader with the knowledge base directory path.

        Args:
            kb_path: Path to the root knowledge_base/ directory.
        """
        self.kb_path = Path(kb_path).resolve()
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_path}")
        if not self.kb_path.is_dir():
            raise NotADirectoryError(f"Knowledge base path is not a directory: {self.kb_path}")

    def load_all(self) -> list[KBEntry]:
        """
        Load and validate all YAML files from the knowledge base directory.

        Recursively scans for .yaml and .yml files, skipping README and
        non-KB files. Returns a list of validated KBEntry instances.
        Invalid files are logged as warnings but do not stop processing.

        Returns:
            List of validated KBEntry objects.
        """
        entries: list[KBEntry] = []
        yaml_files = sorted(self.kb_path.rglob("*.yaml")) + sorted(self.kb_path.rglob("*.yml"))

        if not yaml_files:
            logger.warning("No YAML files found in %s", self.kb_path)
            return entries

        logger.info("Found %d YAML files in %s", len(yaml_files), self.kb_path)

        for yaml_file in yaml_files:
            entry = self._load_file(yaml_file)
            if entry is not None:
                entries.append(entry)

        logger.info(
            "Successfully loaded %d/%d knowledge base entries",
            len(entries),
            len(yaml_files),
        )
        return entries

    def load_by_category(self, category: str) -> list[KBEntry]:
        """
        Load entries filtered by category.

        Args:
            category: Category name to filter (e.g., "policy_types", "claims").

        Returns:
            List of KBEntry objects matching the category.
        """
        all_entries = self.load_all()
        return [e for e in all_entries if e.category == category]

    def _load_file(self, filepath: Path) -> KBEntry | None:
        """
        Load and validate a single YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Validated KBEntry if successful, None if the file is invalid.
        """
        relative_path = str(filepath.relative_to(self.kb_path))

        try:
            with open(filepath, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if raw_data is None:
                logger.warning("Empty YAML file: %s", relative_path)
                return None

            if not isinstance(raw_data, dict):
                logger.warning(
                    "Invalid YAML structure in %s: expected a mapping, got %s",
                    relative_path,
                    type(raw_data).__name__,
                )
                return None

            # Skip files that don't have required KB fields (e.g., README)
            if "category" not in raw_data or "topic" not in raw_data:
                logger.debug("Skipping non-KB file: %s", relative_path)
                return None

            # Validate with Pydantic
            entry = KBEntry(**raw_data, source_file=relative_path)
            logger.debug("Loaded: [%s] %s from %s", entry.category, entry.topic, relative_path)
            return entry

        except yaml.YAMLError as e:
            logger.error(
                "YAML parse error in %s: %s\n"
                "  → Check for indentation issues (use 2 spaces, not tabs)",
                relative_path,
                e,
            )
            return None
        except Exception as e:
            logger.error("Error loading %s: %s", relative_path, e)
            return None

    def get_categories(self) -> list[str]:
        """Return a sorted list of unique categories in the knowledge base."""
        entries = self.load_all()
        return sorted(set(e.category for e in entries))

    def get_stats(self) -> dict[str, int]:
        """Return a count of entries per category."""
        entries = self.load_all()
        stats: dict[str, int] = {}
        for entry in entries:
            stats[entry.category] = stats.get(entry.category, 0) + 1
        return stats
