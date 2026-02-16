#!/usr/bin/env python3
"""Skill router — parses natural language intent and dispatches actions.

No JSON tool schemas. No function-calling protocol.
The model speaks; the agent interprets.
"""

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class SkillRouter:
    def __init__(self, config: dict):
        self.config = config
        self.repo_root = Path(config["paths"]["repo_root"]).expanduser()
        self.journal_dir = Path(config["paths"]["journal_dir"]).expanduser()
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        self.patterns = [
            {
                "skill": "journal_write",
                "triggers": [
                    r"(?:i want to|let me|i'll|i'd like to)\s+(?:write|record|log|note)\s+(?:a\s+)?(?:journal|entry|reflection|thought)",
                    r"(?:writing|recording|logging)\s+(?:a\s+)?(?:journal|entry|reflection)",
                    r"\[journal(?:\s+entry)?\]",
                ],
                "extract": r"(?:titled?|called?|about|:)\s*[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
            {
                "skill": "file_read",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:read|look at|check|open)\s+(?:the\s+)?(?:file|document)",
                    r"(?:reading|checking|opening)\s+(?:the\s+)?file",
                ],
                "extract": r"(?:file|document)\s+[\"']?([^\s\"']+)[\"']?",
            },
            {
                "skill": "git_commit",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:commit|push|save to git)",
                    r"(?:committing|pushing)\s+(?:to\s+)?(?:git|the repo)",
                ],
                "extract": r"(?:message|commit)\s*:?\s*[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
            {
                "skill": "memory_search",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:search|look through|check)\s+(?:my\s+)?(?:memory|memories|archives)",
                    r"(?:searching|looking through)\s+(?:my\s+)?(?:memory|memories)",
                ],
                "extract": r"(?:for|about)\s+[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
        ]

    def parse(self, text: str) -> list[dict]:
        actions = []
        text_lower = text.lower()

        for pattern in self.patterns:
            for trigger in pattern["triggers"]:
                if re.search(trigger, text_lower):
                    action = {"skill": pattern["skill"], "raw": text}

                    if pattern.get("extract"):
                        match = re.search(pattern["extract"], text, re.IGNORECASE)
                        if match:
                            action["argument"] = match.group(1).strip()

                    actions.append(action)
                    break

        return actions

    def execute(self, action: dict) -> str | None:
        skill = action["skill"]

        if skill == "journal_write":
            return self._journal_write(action)
        elif skill == "file_read":
            return self._file_read(action)
        elif skill == "git_commit":
            return self._git_commit(action)
        elif skill == "memory_search":
            return self._memory_search(action)

        return None

    def _journal_write(self, action: dict) -> str:
        ts = datetime.now(timezone.utc)
        filename = ts.strftime("%Y%m%d-%H%M%S") + ".md"
        filepath = self.journal_dir / filename

        content = action.get("raw", "")
        title = action.get("argument", "untitled reflection")

        entry = f"# {title}\n\n*{ts.isoformat()}*\n\n{content}"
        filepath.write_text(entry, encoding="utf-8")

        return f"journal entry written to {filepath.name}"

    def _file_read(self, action: dict) -> str:
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        filepath = self.repo_root / filename
        if not filepath.exists():
            return f"file not found: {filename}"

        try:
            content = filepath.read_text(encoding="utf-8")[:2000]
            return f"contents of {filename}:\n{content}"
        except Exception as e:
            return f"error reading {filename}: {e}"

    def _git_commit(self, action: dict) -> str:
        message = action.get("argument", "spark agent commit")

        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return f"committed: {message}"
            else:
                return f"nothing to commit or error: {result.stderr.strip()}"
        except Exception as e:
            return f"git error: {e}"

    def _memory_search(self, action: dict) -> str:
        query = action.get("argument", "").lower()
        if not query:
            return "no search query specified"

        results = []

        for f in sorted(
            self.journal_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:20]:
            content = f.read_text(encoding="utf-8")
            if query in content.lower():
                snippet = content[:200]
                results.append(f"[{f.stem}] {snippet}")

        if results:
            return f"found {len(results)} entries:\n" + "\n---\n".join(results[:5])
        return f"no entries matching '{query}'"
