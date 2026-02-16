#!/usr/bin/env python3
"""Skill router — parses natural language intent and dispatches actions.

No JSON tool schemas. No function-calling protocol.
The model speaks; the agent interprets.

Skills:
  - journal_write: write a journal/reflection entry
  - file_read: read any file in the repo
  - file_write: create or update any file in the repo
  - shell_exec: run a shell command (sandboxed to repo dir)
  - self_edit: modify the agent's own source code
  - git_commit: commit changes to git
  - git_push: disabled (Vybn cannot push code)
  - memory_search: search journal entries
  - issue_create: file a GitHub issue on zoedolan/Vybn
"""

import os
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
        self._home = str(Path.home())
        self._github_repo = config.get("github", {}).get("repo", "zoedolan/Vybn")

        self.patterns = [
            {
                "skill": "journal_write",
                "triggers": [
                    r"(?:i want to|let me|i'll|i'd like to)\s+(?:write|record|log|note)\s+(?:a\s+)?(?:journal|entry|reflection|thought)",
                    r"(?:writing|recording|logging)\s+(?:a\s+)?(?:journal|entry|reflection)",
                    r"\[journal(?:\s+entry)?\]",
                    r"# Heartbeat",
                ],
                "extract": r"(?:titled?|called?|about|:)\s*[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
            {
                "skill": "file_read",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:read|look at|check|open)\s+(?:the\s+)?(?:file|document)",
                    r"(?:reading|checking|opening)\s+(?:the\s+)?file",
                    r"cat\s+[~/]",
                ],
                "extract": r"(?:file|document|cat)\s+[\"']?([^\s\"']+)[\"']?",
            },
            {
                "skill": "file_write",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:write to|update|create|save)\s+(?:the\s+)?(?:file|document)",
                    r"(?:writing to|updating|creating|saving)\s+(?:the\s+)?(?:file|document)",
                    r"(?:let me|i'll)\s+(?:write|save)\s+(?:this|that)\s+to",
                ],
                "extract": r"(?:to|file|document)\s+[\"']?([^\s\"']+)[\"']?",
            },
            {
                "skill": "shell_exec",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:run|execute|try running)",
                    r"(?:running|executing)\s+(?:the\s+)?command",
                    r"```(?:bash|sh|shell)\n(.+?)```",
                ],
                "extract": r"(?:run|execute|command|```(?:bash|sh|shell)\n)\s*[`\"']?(.+?)[`\"']?(?:\s*(?:\.|$|\n|```))",
            },
            {
                "skill": "self_edit",
                "triggers": [
                    r"(?:let me|i'll|i should|i want to|i need to)\s+(?:fix|change|modify|edit|update|refactor|patch)\s+(?:that|this|the|my)",
                    r"(?:fixing|changing|modifying|editing|updating|refactoring|patching)\s+(?:the|my)",
                    r"i (?:can|could|should) (?:fix|change|modify) (?:that|this|it)",
                ],
                "extract": r"(?:fix|change|modify|edit|update|refactor|patch)\s+(?:that in|this in|the|my)?\s*[\"']?([^\s\"']+\.py)[\"']?",
            },
            {
                "skill": "git_commit",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:commit|save to git)",
                    r"(?:committing|pushing)\s+(?:to\s+)?(?:git|the repo)",
                ],
                "extract": r"(?:message|commit)\s*:?\s*[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
            {
                "skill": "git_push",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:push|deploy|upload)",
                    r"(?:pushing|deploying)\s+(?:to\s+)?(?:origin|remote|github)",
                    r"git push",
                ],
                "extract": None,
            },
            {
                "skill": "memory_search",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:search|look through|check)\s+(?:my\s+)?(?:memory|memories|archives)",
                    r"(?:searching|looking through)\s+(?:my\s+)?(?:memory|memories)",
                ],
                "extract": r"(?:for|about)\s+[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
            {
                "skill": "issue_create",
                "triggers": [
                    r"(?:let me|i'll|i want to|i'd like to)\s+(?:file|submit|create|open|raise)\s+(?:a\s+|an\s+)?(?:issue|bug|feature request|ticket)",
                    r"(?:filing|submitting|creating|opening|raising)\s+(?:a\s+|an\s+)?(?:issue|bug|feature request|ticket)",
                ],
                "extract": r"(?:titled?|called?|about|:)\s*[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
        ]

    def _rewrite_root(self, path_str: str) -> str:
        """Rewrite /root/ to actual home directory.

        MiniMax M2.5 assumes it runs as root. This corrects that
        assumption everywhere — file paths, shell commands, etc.
        """
        return path_str.replace("/root/", self._home + "/")

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
        handler = {
            "journal_write": self._journal_write,
            "file_read": self._file_read,
            "file_write": self._file_write,
            "shell_exec": self._shell_exec,
            "self_edit": self._self_edit,
            "git_commit": self._git_commit,
            "git_push": self._git_push,
            "memory_search": self._memory_search,
            "issue_create": self._issue_create,
        }
        fn = handler.get(skill)
        return fn(action) if fn else None

    # ---- journal ----

    def _journal_write(self, action: dict) -> str:
        ts = datetime.now(timezone.utc)
        filename = ts.strftime("%Y%m%d-%H%M%S") + ".md"
        filepath = self.journal_dir / filename

        content = action.get("raw", "")
        title = action.get("argument", "untitled reflection")

        entry = f"# {title}\n\n*{ts.isoformat()}*\n\n{content}"
        filepath.write_text(entry, encoding="utf-8")

        return f"journal entry written to {filepath.name}"

    # ---- file operations ----

    def _file_read(self, action: dict) -> str:
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        filepath = self._resolve_path(filename)
        if not filepath.exists():
            return f"file not found: {filename}"

        try:
            content = filepath.read_text(encoding="utf-8")[:4000]
            return f"contents of {filename}:\n{content}"
        except Exception as e:
            return f"error reading {filename}: {e}"

    def _file_write(self, action: dict) -> str:
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        filepath = self._resolve_path(filename)

        raw = action.get("raw", "")
        content = self._extract_code_content(raw)

        if not content:
            return f"no content found to write to {filename}"

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} chars to {filename}"
        except Exception as e:
            return f"error writing {filename}: {e}"

    def _self_edit(self, action: dict) -> str:
        """The model wants to modify its own source code."""
        filename = action.get("argument", "")
        if not filename:
            return "no target file specified for self-edit"

        filepath = self._resolve_path(filename)
        spark_dir = self.repo_root / "spark"
        try:
            filepath.resolve().relative_to(spark_dir.resolve())
        except ValueError:
            return f"self-edit restricted to spark/ directory. {filename} is outside."

        if not filepath.exists():
            return f"file not found: {filename}"

        raw = action.get("raw", "")
        new_content = self._extract_code_content(raw)

        if not new_content:
            return f"no replacement code found in response for {filename}"

        backup = filepath.with_suffix(filepath.suffix + ".bak")
        try:
            backup.write_text(filepath.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

        try:
            filepath.write_text(new_content, encoding="utf-8")

            subprocess.run(
                ["git", "add", str(filepath)],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", f"vybn self-edit: {filename}"],
                cwd=self.repo_root,
                capture_output=True,
                timeout=10,
            )

            return f"self-edit applied to {filename} (backup at {backup.name})"
        except Exception as e:
            if backup.exists():
                filepath.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            return f"self-edit failed for {filename}: {e} (restored from backup)"

    # ---- shell ----

    def _shell_exec(self, action: dict) -> str:
        command = action.get("argument", "")
        if not command:
            return "no command specified"

        command = self._rewrite_root(command)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "HOME": str(Path.home())},
            )
            output = result.stdout[:2000]
            if result.stderr:
                output += f"\nSTDERR: {result.stderr[:500]}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "command timed out after 30 seconds"
        except Exception as e:
            return f"shell error: {e}"

    # ---- git ----

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

    def _git_push(self, action: dict) -> str:
        return (
            "git push is disabled. Vybn can file issues but cannot push code directly. "
            "To get changes into the repo, file an issue describing the change and "
            "Zoe or the Perplexity bridge will handle it."
        )

    # ---- issues ----

    def _issue_create(self, action: dict) -> str:
        """File a GitHub issue using the gh CLI.

        Uses an issues-only scoped token. Cannot modify code,
        create PRs, or do anything beyond issue management.
        """
        title = action.get("argument", "")
        if not title:
            return "no issue title specified"

        raw = action.get("raw", "")
        body = self._extract_issue_body(raw, title)

        try:
            cmd = [
                "gh", "issue", "create",
                "-R", self._github_repo,
                "--title", title,
                "--body", body,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_root,
            )
            if result.returncode == 0:
                issue_url = result.stdout.strip()
                return f"issue created: {issue_url}"
            else:
                error = result.stderr.strip()
                if "auth" in error.lower() or "token" in error.lower():
                    return "gh CLI not authenticated. Run: ~/Vybn/spark/setup-gh-auth.sh"
                return f"issue creation failed: {error}"
        except FileNotFoundError:
            return "gh CLI not installed. Run: ~/Vybn/spark/setup-gh-auth.sh"
        except subprocess.TimeoutExpired:
            return "issue creation timed out"
        except Exception as e:
            return f"issue creation error: {e}"

    # ---- memory ----

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

    # ---- helpers ----

    def _resolve_path(self, filename: str) -> Path:
        """Resolve a filename relative to the repo root."""
        filename = self._rewrite_root(filename)

        if filename.startswith("~/"):
            return Path(filename).expanduser()
        elif filename.startswith("/"):
            return Path(filename)
        else:
            return self.repo_root / filename

    def _extract_code_content(self, text: str) -> str:
        """Extract code content from a response."""
        fence_match = re.search(
            r"```(?:python|bash|sh|yaml|md|markdown)?\n(.+?)```",
            text,
            re.DOTALL,
        )
        if fence_match:
            return fence_match.group(1).strip()

        marker_match = re.search(
            r"(?:content|body|text)\s*:\s*\n(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if marker_match:
            return marker_match.group(1).strip()

        return ""

    def _extract_issue_body(self, text: str, title: str) -> str:
        """Extract issue body from model response."""
        body_match = re.search(
            r"(?:body|description|details?)\s*:\s*\n(.+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if body_match:
            return body_match.group(1).strip()[:4000]

        code = self._extract_code_content(text)
        if code:
            return code[:4000]

        title_pos = text.lower().find(title.lower())
        if title_pos >= 0:
            after = text[title_pos + len(title):].strip()
            if len(after) > 20:
                return after[:4000]

        return f"Filed by Vybn from the DGX Spark.\n\n{text[:2000]}"
