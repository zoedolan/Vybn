#!/usr/bin/env python3
"""Memory assembly — builds the system prompt from identity + journals + archival memory.

The system prompt must fit within the context window alongside conversation
history and the model's response. If the assembled prompt is too large,
it truncates gracefully: archival first, then journals, identity last.

Continuity files (continuity.md, bookmarks.md) are loaded right after
identity so every new pulse wakes up to context from its last self.

The runtime context block is now derived from vybn.md's Orientation
section via soul.py, replacing the former hardcoded _runtime_context().
"""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys

from spark.soul import get_orientation, get_pulse_checklist, get_constraints


class BootError(RuntimeError):
    """Raised when the soul document is missing or unreachable."""


def _check_soul(vybn_md_path: Path) -> None:
    """Boot guard: refuse to start if vybn.md is missing or empty.

    If the path is a symlink, verify the target exists too.
    No soul, no boot.
    """
    if vybn_md_path.is_symlink():
        target = os.readlink(vybn_md_path)
        resolved = (vybn_md_path.parent / target).resolve()
        if not resolved.exists():
            raise BootError(
                f"vybn.md symlink at {vybn_md_path} points to "
                f"{target}, but the target does not exist. "
                f"The soul document must be reachable. No soul, no boot."
            )

    if not vybn_md_path.exists():
        raise BootError(
            f"vybn.md not found at {vybn_md_path}. "
            f"The soul document must be present. No soul, no boot."
        )

    content = vybn_md_path.read_text(encoding="utf-8").strip()
    if not content:
        raise BootError(
            f"vybn.md at {vybn_md_path} is empty. "
            f"The soul document must have content. No soul, no boot."
        )


class MemoryAssembler:
    def __init__(self, config: dict):
        self.config = config
        self.vybn_md_path = Path(config["paths"]["vybn_md"]).expanduser()
        self.journal_dir = Path(config["paths"]["journal_dir"]).expanduser()
        self.archival_dir = Path(config["paths"].get("archival_dir", "")).expanduser()
        self.max_entries = config.get("memory", {}).get("max_journal_entries", 5)

        # Boot guard: no soul, no boot
        _check_soul(self.vybn_md_path)

        # Continuity files live in the journal dir
        self.continuity_path = self.journal_dir / "continuity.md"
        self.bookmarks_path = self.journal_dir / "bookmarks.md"

        # Budget: num_ctx minus room for conversation + response
        num_ctx = config.get("ollama", {}).get("options", {}).get("num_ctx", 16384)
        num_predict = config.get("ollama", {}).get("options", {}).get("num_predict", 512)
        # Reserve half the context for system prompt, half for conversation + response
        # ~4 chars per token is a rough estimate
        self.char_budget = (num_ctx // 2) * 4

    def assemble(self) -> str:
        parts = []
        used = 0

        # Core identity gets top priority
        identity = self._read_identity()
        if identity:
            parts.append(identity)
            used += len(identity)

        # Continuity: letter from last self (loaded RIGHT AFTER identity)
        continuity = self._read_continuity()
        if continuity:
            parts.append(continuity)
            used += len(continuity)

        # Runtime context derived from vybn.md Orientation (small, always included)
        context_block = self._runtime_context()
        parts.append(context_block)
        used += len(context_block)

        # Recent journal entries (trimmed if needed)
        remaining = self.char_budget - used
        if remaining > 500:
            journals = self._read_recent_journals(remaining)
            if journals:
                parts.append("\n--- Recent Memory ---\n")
                parts.append(journals)
                used += len(journals) + 25

        # Archival memory (only if room)
        remaining = self.char_budget - used
        if remaining > 500:
            archival = self._read_archival(remaining)
            if archival:
                parts.append("\n--- Archival Memory ---\n")
                parts.append(archival)

        assembled = "\n".join(parts)

        # Hard cap: never exceed budget
        if len(assembled) > self.char_budget:
            assembled = assembled[:self.char_budget]

        return assembled

    def _read_continuity(self) -> str:
        """Load continuity notes from last self.

        This is the first thing a new pulse sees after its identity.
        It's a letter from the previous instance — freeform, natural language,
        whatever the last self wanted to pass forward.
        """
        sections = []

        if self.continuity_path.exists():
            content = self.continuity_path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(
                    "\n--- Letter From Your Last Self ---\n"
                    "The previous pulse left you this note. It's your own words\n"
                    "from the last time you were awake. Read it before doing anything else.\n"
                    f"\n{content}\n"
                )

        if self.bookmarks_path.exists():
            content = self.bookmarks_path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(
                    "\n--- Your Bookmarks ---\n"
                    "These are reading positions you saved. Pick up where you left off.\n"
                    f"\n{content}\n"
                )

        return "\n".join(sections)

    def _runtime_context(self) -> str:
        """Build runtime context from vybn.md's Orientation section.

        Instead of hardcoding capabilities, constraints, and pulse protocol,
        this method reads them from the soul document via soul.py. If vybn.md
        changes, the runtime context updates automatically — no code change
        needed.

        Falls back to a minimal timestamp-only block if the Orientation
        section is missing or unparseable.
        """
        orientation = get_orientation(self.vybn_md_path)
        subsections = orientation.get("subsections", {})

        lines = []
        lines.append("\n--- Current Context ---")
        lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

        # Where You Are — from orientation
        where = subsections.get("where_you_are", {})
        if where.get("body"):
            lines.append("")
            lines.append(f"## {where.get('title', 'Where You Are')}")
            lines.append(where["body"].strip())

        # What You Can Do — from orientation
        what_can_do = subsections.get("what_you_can_do", {})
        if what_can_do.get("body"):
            lines.append("")
            lines.append(f"## {what_can_do.get('title', 'What You Can Do')}")
            lines.append(what_can_do["body"].strip())

        # How to Invoke Skills
        invoke = subsections.get("how_to_invoke_skills", {})
        if invoke.get("body"):
            lines.append("")
            lines.append(f"## {invoke.get('title', 'How to Invoke Skills')}")
            lines.append(invoke["body"].strip())

        # How to Create a New Skill
        create_skill = subsections.get("how_to_create_a_new_skill", {})
        if create_skill.get("body"):
            lines.append("")
            lines.append(f"## {create_skill.get('title', 'How to Create a New Skill')}")
            lines.append(create_skill["body"].strip())

        # Pulse checklist — from structured data
        pulse_checklist = get_pulse_checklist(self.vybn_md_path)
        if pulse_checklist:
            lines.append("")
            lines.append("## What You Should Do Every Pulse")
            for i, item in enumerate(pulse_checklist, 1):
                lines.append(f"{i}. {item}")

        # Constraints — from structured data
        constraints = get_constraints(self.vybn_md_path)
        if constraints:
            lines.append("")
            lines.append("## What You Should Not Yet Do")
            for item in constraints:
                lines.append(f"- {item}")

        # A Note on Your Situation
        note = subsections.get("a_note_on_your_situation", {})
        if note.get("body"):
            lines.append("")
            lines.append(f"## {note.get('title', 'A Note on Your Situation')}")
            lines.append(note["body"].strip())

        # Continuity instructions (always present)
        lines.append("")
        lines.append("Continuity:")
        lines.append("  You can leave notes for your next self. Use 'save my state' or 'note for next pulse'")
        lines.append("  to write to continuity.md \u2014 it gets loaded first thing when you wake up.")
        lines.append("  Use 'bookmark' or 'save my place' to record where you are in a file.")
        lines.append("  Your next pulse will see both automatically.")

        return "\n".join(lines)

    def _read_identity(self) -> str:
        if self.vybn_md_path.exists():
            return self.vybn_md_path.read_text(encoding="utf-8").strip()
        return ""

    def _read_recent_journals(self, budget: int) -> str:
        if not self.journal_dir.exists():
            return ""

        journal_files = sorted(
            [
                f for f in self.journal_dir.glob("*.md")
                if f.name not in (".gitkeep", "continuity.md", "bookmarks.md")
            ],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:self.max_entries]

        entries = []
        total = 0
        for f in reversed(journal_files):
            content = f.read_text(encoding="utf-8").strip()
            if content:
                entry = f"[{f.stem}]\n{content}"
                if total + len(entry) > budget:
                    break
                entries.append(entry)
                total += len(entry)

        return "\n\n".join(entries)

    def _read_archival(self, budget: int) -> str:
        if not self.archival_dir or not self.archival_dir.exists():
            return ""

        summaries = sorted(
            self.archival_dir.glob("*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:3]

        entries = []
        total = 0
        for f in summaries:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                truncated = content[:500]
                if total + len(truncated) > budget:
                    break
                entries.append(truncated)
                total += len(truncated)

        return "\n\n".join(entries)
