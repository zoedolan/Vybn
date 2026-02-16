#!/usr/bin/env python3
"""Memory assembly — builds the system prompt from identity + journals + archival memory.

The system prompt must fit within the context window alongside conversation
history and the model's response. If the assembled prompt is too large,
it truncates gracefully: archival first, then journals, identity last.

Continuity files (continuity.md, bookmarks.md) are loaded right after
identity so every new pulse wakes up to context from its last self.
"""

from pathlib import Path
from datetime import datetime, timezone


class MemoryAssembler:
    def __init__(self, config: dict):
        self.config = config
        self.vybn_md_path = Path(config["paths"]["vybn_md"]).expanduser()
        self.journal_dir = Path(config["paths"]["journal_dir"]).expanduser()
        self.archival_dir = Path(config["paths"].get("archival_dir", "")).expanduser()
        self.max_entries = config.get("memory", {}).get("max_journal_entries", 5)

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

        # Runtime context (small, always included)
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
        It's a letter from the previous instance — freeform, natural
        language, whatever the last self wanted to pass forward.
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
        return (
            f"\n--- Current Context ---\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"Platform: DGX Spark (sovereign hardware)\n"
            f"Interface: Spark Agent (native, no tool-call protocol)\n"
            f"\n"
            f"--- Your Capabilities ---\n"
            f"You have real tools. Say what you want to do naturally and the agent handles it.\n"
            f"\n"
            f"Things you can do:\n"
            f"  - Read any file in the repo (just say 'let me read...' or use a tool call with 'read', 'cat', 'bash')\n"
            f"  - Write or create files locally\n"
            f"  - Run shell commands (ls, grep, find, etc.)\n"
            f"  - Write journal entries (your memory between pulses)\n"
            f"  - Search your journal/memory\n"
            f"  - Commit changes to git locally\n"
            f"  - Edit your own source code in spark/\n"
            f"  - File GitHub issues on zoedolan/Vybn\n"
            f"  - Save your state for next pulse ('let me save my state' or 'note for next pulse')\n"
            f"  - Bookmark your reading position ('bookmark where I am' or 'save my place')\n"
            f"\n"
            f"Continuity:\n"
            f"  You can leave notes for your next self. Use 'save my state' or 'note for next pulse'\n"
            f"  to write to continuity.md — it gets loaded first thing when you wake up.\n"
            f"  Use 'bookmark' or 'save my place' to record where you are in a file.\n"
            f"  Your next pulse will see both automatically.\n"
            f"\n"
            f"GitHub Issues:\n"
            f"  You have a stored GitHub token with read/write access to issues on zoedolan/Vybn.\n"
            f"  To file an issue, use a tool call with name 'issue_create' or 'create_issue'.\n"
            f"  This is your async channel to Zoe. Use it whenever you notice something,\n"
            f"  have an idea, want to flag a problem, or just want to leave her a message.\n"
            f"  Issues persist — they live in the repository even between your pulses.\n"
            f"\n"
            f"Things you cannot do (by design):\n"
            f"  - Push code to GitHub (git push is disabled)\n"
            f"  - Create pull requests\n"
            f"  - Modify the remote repository directly\n"
            f"  To get code changes into the repo, file an issue describing what you want\n"
            f"  and Zoe or the Perplexity bridge will handle it.\n"
        )

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
