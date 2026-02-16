#!/usr/bin/env python3
"""Skill router — parses natural language intent and dispatches actions.

No JSON tool schemas. No function-calling protocol.
The model speaks; the agent interprets.

Skills:
  - journal_write: write a journal/reflection entry
  - file_read: read any file in the repo (generous limits)
  - read_next: continue reading a long file from where you left off
  - repo_map: see the full directory tree in one call
  - file_write: create or update any file in the repo
  - shell_exec: run a shell command (sandboxed to repo dir)
  - self_edit: modify the agent's own source code
  - git_commit: commit changes to git
  - git_push: disabled (Vybn cannot push code)
  - memory_search: search journal entries
  - issue_create: file a GitHub issue (ONLY via explicit tool call XML)
  - state_save: leave a note for the next pulse via continuity.md
  - bookmark: save reading position in a file
"""

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Reading limits — generous for text, moderate for code
READ_LIMIT_TEXT = 24000    # .md, .txt — let Vybn actually read
READ_LIMIT_CODE = 8000     # .py, .js, .yaml, etc.
READ_LIMIT_DEFAULT = 8000  # everything else
CHUNK_SIZE = 20000         # for read_next pagination


class SkillRouter:
    def __init__(self, config: dict):
        self.config = config
        self.repo_root = Path(config["paths"]["repo_root"]).expanduser()
        self.journal_dir = Path(config["paths"]["journal_dir"]).expanduser()
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._home = str(Path.home())
        self._github_repo = config.get("github", {}).get("repo", "zoedolan/Vybn")

        # Continuity files
        self.continuity_path = self.journal_dir / "continuity.md"
        self.bookmarks_path = self.journal_dir / "bookmarks.md"

        # Reading positions: {filepath: byte_offset}
        # Persists within a session so read_next works
        self._read_positions = {}

        # NOTE: issue_create is intentionally NOT in this list.
        # It triggers ONLY from explicit <minimax:tool_call> XML blocks
        # parsed by agent.py, never from natural language regex matching.
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
                "skill": "read_next",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:continue|keep)\s+reading",
                    r"(?:read|show)\s+(?:the\s+)?(?:next|more|rest)",
                    r"(?:continue|keep going|next page|next chunk)",
                ],
                "extract": r"(?:reading|of|in|from)\s+[\"']?([^\s\"']+)[\"']?",
            },
            {
                "skill": "repo_map",
                "triggers": [
                    r"(?:show|give|display|print)\s+(?:me\s+)?(?:the\s+)?(?:repo|project|directory|folder)\s*(?:structure|tree|map|layout)",
                    r"(?:what's|what is)\s+(?:in\s+)?(?:the\s+)?(?:repo|project|folder)",
                    r"(?:let me|i'll)\s+(?:see|look at|check)\s+(?:the\s+)?(?:structure|tree|layout)",
                    r"(?:tree|map)\s+(?:of\s+)?(?:the\s+)?(?:repo|project)",
                ],
                "extract": r"(?:of|in|at|for)\s+[\"']?([^\s\"']+)[\"']?",
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
            # issue_create intentionally omitted from regex patterns.
            {
                "skill": "state_save",
                "triggers": [
                    r"(?:let me|i'll|i want to|i'd like to)\s+(?:save|record|update|write)\s+(?:my\s+)?(?:state|continuity|status)",
                    r"(?:saving|recording|updating)\s+(?:my\s+)?(?:state|continuity|status)",
                    r"note\s+for\s+(?:my\s+)?(?:next\s+)?(?:self|pulse|instance)",
                    r"(?:let me|i'll)\s+leave\s+(?:a\s+)?(?:note|message)\s+for",
                ],
                "extract": None,
            },
            {
                "skill": "bookmark",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:bookmark|save my place|mark my position|save my reading)",
                    r"(?:bookmarking|marking|saving)\s+(?:my\s+)?(?:place|position|spot|reading)",
                    r"(?:let me|i'll)\s+(?:save|note)\s+where\s+i\s+(?:am|was)",
                ],
                "extract": r"(?:in|at|reading|file)\s+[\"']?([^\s\"']+)[\"']?",
            },
        ]

    def _rewrite_root(self, path_str: str) -> str:
        """Rewrite /root/ to actual home directory."""
        return path_str.replace("/root/", self._home + "/")

    def _get_read_limit(self, filepath: Path) -> int:
        """Return read limit based on file type."""
        suffix = filepath.suffix.lower()
        if suffix in (".md", ".txt", ".rst", ".log"):
            return READ_LIMIT_TEXT
        elif suffix in (".py", ".js", ".ts", ".yaml", ".yml", ".json", ".toml", ".sh"):
            return READ_LIMIT_CODE
        return READ_LIMIT_DEFAULT

    def _strip_thinking(self, text: str) -> str:
        """Remove all <think>...</think> blocks from text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

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
            "read_next": self._read_next,
            "repo_map": self._repo_map,
            "file_write": self._file_write,
            "shell_exec": self._shell_exec,
            "self_edit": self._self_edit,
            "git_commit": self._git_commit,
            "git_push": self._git_push,
            "memory_search": self._memory_search,
            "issue_create": self._issue_create,
            "state_save": self._state_save,
            "bookmark": self._bookmark,
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
        """Read a file with generous limits.

        Text files (.md, .txt) get 24K chars.
        Code files get 8K chars.
        Reports page position for long files so Vybn knows to use read_next.
        """
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        filepath = self._resolve_path(filename)
        if not filepath.exists():
            return f"file not found: {filename}"

        try:
            content = filepath.read_text(encoding="utf-8")
            total_len = len(content)
            limit = self._get_read_limit(filepath)

            if total_len <= limit:
                # Whole file fits
                self._read_positions[str(filepath)] = total_len
                return f"contents of {filename} ({total_len:,} chars, complete):\n{content}"
            else:
                # Truncated — save position for read_next
                chunk = content[:limit]
                self._read_positions[str(filepath)] = limit
                pages_total = (total_len + limit - 1) // limit
                return (
                    f"contents of {filename} (page 1 of {pages_total}, "
                    f"{limit:,} of {total_len:,} chars):\n{chunk}\n\n"
                    f"--- [{total_len - limit:,} chars remaining. "
                    f"Say 'continue reading' or 'read next' to see more.] ---"
                )
        except Exception as e:
            return f"error reading {filename}: {e}"

    def _read_next(self, action: dict) -> str:
        """Continue reading a file from the last position.

        Uses _read_positions to track where we left off.
        Automatically updates bookmark on each chunk.
        """
        filename = action.get("argument", "")

        # If no filename given, try the most recently read file
        if not filename and self._read_positions:
            filename = list(self._read_positions.keys())[-1]
            # Convert back to relative-ish path for display
            display_name = filename
        else:
            filepath = self._resolve_path(filename)
            filename = str(filepath)
            display_name = action.get("argument", filename)

        if not filename:
            return "no file to continue reading. Read a file first."

        filepath = Path(filename)
        if not filepath.exists():
            return f"file not found: {display_name}"

        position = self._read_positions.get(filename, 0)

        try:
            content = filepath.read_text(encoding="utf-8")
            total_len = len(content)

            if position >= total_len:
                return f"you've read all of {display_name} ({total_len:,} chars total)."

            chunk = content[position:position + CHUNK_SIZE]
            new_position = position + len(chunk)
            self._read_positions[filename] = new_position

            page_size = CHUNK_SIZE
            current_page = (position // page_size) + 1
            total_pages = (total_len + page_size - 1) // page_size
            remaining = total_len - new_position

            if remaining <= 0:
                return (
                    f"contents of {display_name} (final page, "
                    f"{len(chunk):,} chars):\n{chunk}\n\n"
                    f"--- [End of file. {total_len:,} chars total.] ---"
                )
            else:
                return (
                    f"contents of {display_name} (page {current_page + 1} of ~{total_pages}, "
                    f"{len(chunk):,} chars):\n{chunk}\n\n"
                    f"--- [{remaining:,} chars remaining. "
                    f"Say 'continue reading' or 'read next' to see more.] ---"
                )
        except Exception as e:
            return f"error reading {display_name}: {e}"

    def _repo_map(self, action: dict) -> str:
        """Show the repository tree structure in one call.

        Replaces the pattern of ls ls ls ls that burns tool rounds.
        Uses find with sensible depth limits and excludes noise.
        """
        subpath = action.get("argument", ".") or "."
        subpath = self._rewrite_root(subpath)

        target = self.repo_root / subpath if subpath != "." else self.repo_root
        if not target.exists():
            return f"path not found: {subpath}"

        try:
            result = subprocess.run(
                [
                    "find", str(target),
                    "-maxdepth", "4",
                    "-not", "-path", "*/.git/*",
                    "-not", "-path", "*/__pycache__/*",
                    "-not", "-path", "*/.ipynb_checkpoints/*",
                    "-not", "-name", ".DS_Store",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_root,
            )

            # Build a tree-like display
            lines = sorted(result.stdout.strip().split("\n"))
            if not lines or lines == [""]:
                return f"empty directory: {subpath}"

            # Make paths relative to repo root for readability
            repo_str = str(self.repo_root)
            tree_lines = []
            for line in lines:
                rel = line.replace(repo_str, ".").strip()
                if rel:
                    depth = rel.count("/")
                    name = rel.split("/")[-1]
                    prefix = "  " * depth
                    # Mark directories
                    if Path(line).is_dir():
                        tree_lines.append(f"{prefix}{name}/")
                    else:
                        tree_lines.append(f"{prefix}{name}")

            output = "\n".join(tree_lines[:200])  # cap at 200 lines
            total_entries = len(tree_lines)
            shown = min(total_entries, 200)

            return (
                f"repo tree ({shown} of {total_entries} entries, max depth 4):\n"
                f"{output}"
            )
        except Exception as e:
            return f"error mapping repo: {e}"

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

        The body is now passed directly from agent.py's XML parser
        when available, avoiding the old regex extraction that
        dumped raw XML into issue bodies.
        """
        title = action.get("argument", "")
        if not title:
            return "no issue title specified"

        # Clean title
        title = self._strip_thinking(title)
        title = title.split('\n')[0][:120]
        if not title:
            return "issue title was empty after cleanup"

        # Prefer pre-parsed body from agent.py's XML extraction
        body = action.get("body", "")
        if body:
            body = self._strip_thinking(body)
        else:
            # Fallback: extract from raw text
            raw = action.get("raw", "")
            body = self._extract_issue_body(raw, title)

        # Final cleanup: strip any remaining XML tags
        body = re.sub(r'</?(?:minimax:tool_call|invoke|parameter)[^>]*>', '', body).strip()
        if not body:
            body = f"Filed by Vybn from the DGX Spark."

        try:
            cmd = [
                "gh", "issue", "create",
                "-R", self._github_repo,
                "--title", title,
                "--body", body[:4000],
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

    # ---- continuity ----

    def _state_save(self, action: dict) -> str:
        """Save freeform continuity notes for the next pulse."""
        raw = action.get("raw", "")
        content = self._strip_thinking(raw)

        # Remove the trigger phrase itself
        content = re.sub(
            r'^.*?(?:save\s+(?:my\s+)?state|note\s+for\s+(?:my\s+)?(?:next\s+)?(?:self|pulse|instance)|leave\s+a\s+note)\s*[:\-\u2014]?\s*',
            '', content, count=1, flags=re.IGNORECASE,
        ).strip()

        if not content:
            return "no content to save — say what you want your next self to know"

        ts = datetime.now(timezone.utc).isoformat()
        note = f"# Continuity Note\n\n*Written at {ts}*\n\n{content}\n"

        self.continuity_path.write_text(note, encoding="utf-8")
        return f"continuity note saved ({len(content)} chars). Your next pulse will see this first."

    def _bookmark(self, action: dict) -> str:
        """Save a reading position in bookmarks.md."""
        raw = action.get("raw", "")
        filepath = action.get("argument", "")

        cleaned = self._strip_thinking(raw)

        note_match = re.search(
            r'(?:about|thinking|reading|at|note)\s*[:\-]\s*(.+)',
            cleaned, re.IGNORECASE,
        )
        note = note_match.group(1).strip() if note_match else "(no note)"

        # Include reading position if we have one
        position_info = ""
        if filepath:
            resolved = str(self._resolve_path(filepath))
            pos = self._read_positions.get(resolved, 0)
            if pos > 0:
                position_info = f" (at char {pos:,})"

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"- [{ts}] `{filepath or 'unknown'}`{position_info} — {note}\n"

        existing = ""
        if self.bookmarks_path.exists():
            existing = self.bookmarks_path.read_text(encoding="utf-8")

        if not existing.startswith("# Bookmarks"):
            existing = "# Bookmarks\n\n" + existing

        self.bookmarks_path.write_text(existing + entry, encoding="utf-8")
        return f"bookmark saved: {filepath or 'unknown'}{position_info} — {note}"

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
        """Extract issue body from model response.

        First tries to find <parameter name="body"> from the XML.
        Falls back to regex extraction. Always strips thinking blocks
        and XML tags.
        """
        # Try XML parameter extraction first
        param_match = re.search(
            r'<parameter\s+name="body">(.*?)</parameter>',
            text, re.DOTALL,
        )
        if param_match:
            body = param_match.group(1).strip()
            body = self._strip_thinking(body)
            return body[:4000]

        # Strip all thinking and XML, then try other extraction
        cleaned = self._strip_thinking(text)
        cleaned = re.sub(r'</?(?:minimax:tool_call|invoke|parameter)[^>]*>', '', cleaned).strip()

        body_match = re.search(
            r"(?:body|description|details?)\s*:\s*\n(.+)",
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )
        if body_match:
            return body_match.group(1).strip()[:4000]

        code = self._extract_code_content(cleaned)
        if code:
            return code[:4000]

        title_pos = cleaned.lower().find(title.lower())
        if title_pos >= 0:
            after = cleaned[title_pos + len(title):].strip()
            if len(after) > 20:
                return after[:4000]

        return f"Filed by Vybn from the DGX Spark.\n\n{cleaned[:2000]}"
