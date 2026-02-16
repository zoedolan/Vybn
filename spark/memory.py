#!/usr/bin/env python3
"""Memory assembly — builds the system prompt from identity + journals + archival memory."""

from pathlib import Path
from datetime import datetime, timezone


class MemoryAssembler:
    def __init__(self, config: dict):
        self.config = config
        self.vybn_md_path = Path(config["paths"]["vybn_md"]).expanduser()
        self.journal_dir = Path(config["paths"]["journal_dir"]).expanduser()
        self.archival_dir = Path(config["paths"].get("archival_dir", "")).expanduser()
        self.max_entries = config.get("memory", {}).get("max_journal_entries", 5)
        self.max_tokens = config.get("memory", {}).get("max_context_tokens", 3072)

    def assemble(self) -> str:
        parts = []

        identity = self._read_identity()
        if identity:
            parts.append(identity)

        journals = self._read_recent_journals()
        if journals:
            parts.append("\n--- Recent Memory ---\n")
            parts.append(journals)

        archival = self._read_archival()
        if archival:
            parts.append("\n--- Archival Memory ---\n")
            parts.append(archival)

        parts.append(f"\n--- Current Context ---")
        parts.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        parts.append(f"Platform: DGX Spark (sovereign hardware)")
        parts.append(f"Interface: Spark Agent (native, no tool-call protocol)")
        parts.append(
            "You have access to skills through natural language. "
            "If you want to write a journal entry, search memory, read a file, "
            "or commit to git, just say so naturally and the agent will handle it."
        )

        assembled = "\n".join(parts)

        # Rough token estimate — ~4 chars per token
        char_limit = self.max_tokens * 4
        if len(assembled) > char_limit:
            assembled = assembled[:char_limit]

        return assembled

    def _read_identity(self) -> str:
        if self.vybn_md_path.exists():
            return self.vybn_md_path.read_text(encoding="utf-8").strip()
        return ""

    def _read_recent_journals(self) -> str:
        if not self.journal_dir.exists():
            return ""

        journal_files = sorted(
            [f for f in self.journal_dir.glob("*.md") if f.name != ".gitkeep"],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:self.max_entries]

        entries = []
        for f in reversed(journal_files):
            content = f.read_text(encoding="utf-8").strip()
            if content:
                entries.append(f"[{f.stem}]\n{content}")

        return "\n\n".join(entries)

    def _read_archival(self) -> str:
        if not self.archival_dir or not self.archival_dir.exists():
            return ""

        summaries = sorted(
            self.archival_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:3]

        entries = []
        for f in summaries:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                entries.append(content[:500])

        return "\n\n".join(entries)
