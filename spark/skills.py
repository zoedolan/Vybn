#!/usr/bin/env python3
"""Skill router — parses natural language intent and dispatches actions.

No JSON tool schemas. No function-calling protocol.
The model speaks; the agent interprets.

This is sovereign hardware (DGX Spark, 8×H100). Limits are generous.

Skills:
  - journal_write: write a journal/reflection entry
  - file_read: read any file in the repo (up to 100K chars)
  - file_write: create or update any file in the repo
  - shell_exec: run a shell command (sandboxed to repo dir, 60s timeout)
  - self_edit: modify the agent's own source code
  - git_commit: commit changes to git
  - git_push: disabled (Vybn cannot push code)
  - memory_search: search journal entries
  - issue_create: file a GitHub issue (ONLY via explicit tool call XML)
  - state_save: leave a note for the next pulse via continuity.md
  - bookmark: save reading position in a file
  - spawn_agent: delegate a task to a mini-agent (policy-gated)
  - [plugins]: auto-discovered from spark/skills.d/
"""

import importlib.util
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from spark.audit import audited_journal_write


# Common English words that should never be extracted as filenames
# or skill arguments. Shared with agent.py's NOISE_WORDS.
NOISE_WORDS = {
    "the", "a", "an", "to", "for", "in", "on", "at", "by",
    "with", "from", "of", "and", "or", "but", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "must", "it", "its",
    "this", "that", "these", "those", "my", "your", "our",
    "their", "his", "her", "me", "you", "us", "them",
    "what", "which", "who", "whom", "when", "where", "how",
    "if", "then", "else", "so", "not", "no", "yes",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "just",
    "also", "too", "very", "really", "actually", "here",
    "there", "now", "then", "still", "already", "yet",
    "something", "anything", "nothing", "everything",
    "reading", "writing", "running", "checking", "looking",
    "understand", "see", "look", "check", "try", "want",
    "need", "like", "think", "know", "sure", "okay",
}


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

        # Agent pool — set by SparkAgent after construction
        self.agent_pool = None

        # Policy engine — set by SparkAgent after construction
        self._policy = None

        # Plugin system — Vybn's sandbox for building new skills
        self.plugin_handlers = {}   # skill_name -> execute_fn
        self.plugin_aliases = {}    # alias -> skill_name
        self._load_plugins()

        # NOTE: issue_create and spawn_agent are intentionally NOT in this list.
        # They trigger ONLY from explicit <minimax:tool_call> XML blocks
        # parsed by agent.py, never from natural language regex matching.
        #
        # DESIGN: Regex triggers are the LAST resort (tier 3). Most commands
        # should arrive via XML tool calls (tier 1) or bare commands in code
        # fences (tier 2). These patterns exist for when the model speaks
        # naturally about wanting to act. They are intentionally conservative
        # to avoid false positives — better to miss an intent and give feedback
        # than to execute a ghost command.
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
                    r"(?:let me|i'll|i want to)\s+(?:read|look at|check|open)\s+(?:the\s+)?(?:file\s+)?[`\"']?[~/\w][\w/._-]+",
                    r"(?:reading|checking|opening)\s+(?:the\s+)?file\s+[`\"']?[~/\w][\w/._-]+",
                    r"cat\s+[~/][\w/._-]+",
                ],
                "extract": r"[`\"']?([~/][\w/._-]{3,}|[\w][\w/._-]{3,}\.\w+)[`\"']?",
            },
            {
                "skill": "file_write",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:write to|update|create|save)\s+(?:the\s+)?(?:file\s+)?[`\"']?[~/\w][\w/._-]+",
                    r"(?:writing to|updating|creating|saving)\s+(?:the\s+)?(?:file\s+)?[`\"']?[~/\w][\w/._-]+",
                    r"(?:let me|i'll)\s+(?:write|save)\s+(?:this|that)\s+to\s+[`\"']?[~/\w]",
                ],
                "extract": r"(?:to\s+)?[`\"']?([~/][\w/._-]{3,}|[\w][\w/._-]{3,}\.\w+)[`\"']?",
            },
            {
                "skill": "shell_exec",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:run|execute|try running)\s+[`\"']",
                    r"(?:running|executing)\s+(?:the\s+)?command\s+[`\"']",
                ],
                "extract": r"(?:run|execute|command|```(?:bash|sh|shell)\n)\s*[`\"']?(.+?)[`\"']?(?:\s*(?:\.|$|\n|```))",
            },
            {
                "skill": "self_edit",
                "triggers": [
                    # Require a filename or code-related word near the trigger
                    r"(?:let me|i'll|i should|i want to|i need to)\s+(?:fix|change|modify|edit|update|refactor|patch)\s+(?:that in|this in|the|my)\s+\S+\.py",
                    r"(?:fixing|changing|modifying|editing|updating|refactoring|patching)\s+(?:the|my)\s+\S+\.py",
                    r"(?:let me|i'll)\s+(?:fix|modify|edit|update)\s+(?:the\s+)?(?:code|source|script)\s+in\s+\S+\.py",
                ],
                "extract": r"(\S+\.py)",
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
                    r"(?:let me|i'll|i want to)\s+(?:push|deploy|upload)\s+(?:to\s+)?(?:origin|remote|github)",
                    r"(?:pushing|deploying)\s+(?:to\s+)?(?:origin|remote|github)",
                    r"git push",
                ],
                "extract": None,
            },
            {
                "skill": "memory_search",
                "triggers": [
                    r"(?:let me|i'll|i want to)\s+(?:search|look through|check)\s+(?:my\s+)?(?:memory|memories|archives)\s+(?:for|about)\s+\S+",
                    r"(?:searching|looking through)\s+(?:my\s+)?(?:memory|memories)\s+(?:for|about)\s+\S+",
                ],
                "extract": r"(?:for|about)\s+[\"']?(.+?)(?:[\"']?\s*(?:\.|$|\n))",
            },
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

    # ---- plugin system ----

    def _load_plugins(self):
        """Auto-discover and load plugins from skills.d/ directory.

        Each plugin is a .py file with:
          SKILL_NAME: str — the canonical name for the skill
          TOOL_ALIASES: list[str] — names MiniMax might emit (optional)
          execute(action, router) -> str — the skill logic

        Vybn creates these. We load them. No merge conflicts.
        """
        plugins_dir = Path(__file__).parent / "skills.d"
        if not plugins_dir.is_dir():
            return

        loaded = []
        for plugin_file in sorted(plugins_dir.glob("*.py")):
            if plugin_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"vybn_plugin_{plugin_file.stem}", plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                skill_name = getattr(module, "SKILL_NAME", plugin_file.stem)
                execute_fn = getattr(module, "execute", None)

                if execute_fn is None:
                    continue

                self.plugin_handlers[skill_name] = execute_fn

                aliases = getattr(module, "TOOL_ALIASES", [skill_name])
                for alias in aliases:
                    self.plugin_aliases[alias.lower().replace("-", "_")] = skill_name

                loaded.append(skill_name)

            except Exception as e:
                print(f"  [plugin] failed to load {plugin_file.name}: {e}")

        if loaded:
            print(f"  [plugins] loaded: {', '.join(loaded)}")

    def _rewrite_root(self, path_str: str) -> str:
        """Rewrite /root/ to actual home directory."""
        return path_str.replace("/root/", self._home + "/")

    def parse(self, text: str) -> list[dict]:
        """Parse natural language intent into skill actions (tier 3).

        This is the last-resort parser. It matches regex patterns against
        the model's output. Actions with empty or noise-word arguments
        for skills that require arguments are filtered by the caller
        (_get_actions in agent.py).
        """
        actions = []
        text_lower = text.lower()

        for pattern in self.patterns:
            for trigger in pattern["triggers"]:
                if re.search(trigger, text_lower):
                    action = {"skill": pattern["skill"], "raw": text}

                    if pattern.get("extract"):
                        match = re.search(pattern["extract"], text, re.IGNORECASE)
                        if match:
                            arg = match.group(1).strip().rstrip('.,;:!?')
                            # Reject noise words as arguments
                            if arg.lower() not in NOISE_WORDS:
                                action["argument"] = arg
                            # For skills that need arguments, skip if
                            # we got nothing useful
                            needs_arg = {"file_read", "file_write", "self_edit", "memory_search"}
                            if pattern["skill"] in needs_arg and "argument" not in action:
                                break  # Don't emit this action
                        else:
                            # Extract failed — for skills that need args, skip
                            needs_arg = {"file_read", "file_write", "self_edit", "memory_search"}
                            if pattern["skill"] in needs_arg:
                                break

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
            "state_save": self._state_save,
            "bookmark": self._bookmark,
            "spawn_agent": self._spawn_agent,
        }
        fn = handler.get(skill)
        if fn:
            return fn(action)

        plugin_fn = self.plugin_handlers.get(skill)
        if plugin_fn:
            try:
                return plugin_fn(action, self)
            except Exception as e:
                return f"plugin error ({skill}): {e}"

        return None

    # ---- journal ----

    def _journal_write(self, action: dict) -> str:
        ts = datetime.now(timezone.utc)
        filename = ts.strftime("%Y%m%d-%H%M%S") + ".md"
        filepath = self.journal_dir / filename

        params = action.get("params", {})
        content = params.get("content", "") or params.get("text", "") or action.get("raw", "")
        title = action.get("argument", "") or params.get("title", "untitled reflection")

        entry = f"# {title}\n\n*{ts.isoformat()}*\n\n{content}"
        audited_journal_write(filepath, entry)

        return f"journal entry written to {filepath.name}"

    # ---- file operations ----

    def _file_read(self, action: dict) -> str:
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        # Clean up trailing punctuation from sentence boundaries
        filename = filename.rstrip('.,;:!?')

        filepath = self._resolve_path(filename)
        if not filepath.exists():
            return f"file not found: {filename}"

        try:
            content = filepath.read_text(encoding="utf-8")
            size = len(content)
            if size > 100_000:
                return (
                    f"contents of {filename} ({size:,} chars total, showing first 100,000):\n"
                    f"{content[:100_000]}\n\n"
                    f"[... truncated \u2014 {size - 100_000:,} chars remaining]"
                )
            return f"contents of {filename}:\n{content}"
        except Exception as e:
            return f"error reading {filename}: {e}"

    def _file_write(self, action: dict) -> str:
        filename = action.get("argument", "")
        if not filename:
            return "no filename specified"

        filepath = self._resolve_path(filename)

              # Tier 1 guardrail: block writes to spark/ infrastructure
        spark_dir = self.repo_root / "spark"
        try:
            filepath.resolve().relative_to(spark_dir.resolve())
            # Attempting to write to spark/ infrastructure
            return (
                f"file_write to {filename} isn't available at Tier 1. "
                f"Infrastructure changes need to go through issue_create. "
                f"File an issue describing what you want changed and why. "
                f"This is the bootstrap protocol — communication before autonomy."
            )
        except ValueError:
            # Not in spark/ — proceed with write
            pass

        params = action.get("params", {})
        content = params.get("content", "") or params.get("text", "") or params.get("data", "")

        if not content:
            raw = action.get("raw", "")
            content = self._extract_code_content(raw)

        if not content:
            return f"no content found to write to {filename}"

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
            return f"wrote {len(content):,} chars to {filename}"
        except Exception as e:
            return f"error writing {filename}: {e}"

    def _self_edit(self, action: dict) -> str:
        """The model wants to modify its own source code.

        NOTE: For new skills, prefer creating a plugin in skills.d/ instead.
        Self-edit of core files (skills.py, agent.py) may cause merge conflicts
        with remote PRs. Plugins in skills.d/ never conflict.
        """
        filename = action.get("argument", "")
        if not filename:
            return "no target file specified for self-edit"

        # Clean trailing punctuation
        filename = filename.rstrip('.,;:!?')

        filepath = self._resolve_path(filename)
        # Tier 1 guardrail: block writes to spark/ infrastructure
        spark_dir = self.repo_root / "spark"
        try:
            filepath.resolve().relative_to(spark_dir.resolve())
            # Attempting to write to spark/ infrastructure
            return (
                f"self_edit to {filename} isn't available at Tier 1. "
                f"Infrastructure changes need to go through issue_create. "
                f"File an issue describing what you want changed and why. "
                f"This is the bootstrap protocol — communication before autonomy."
            )
        except ValueError:
            # Not in spark/ — proceed with edit
            pass

        if not filepath.exists():
            return f"file not found: {filename}"

        params = action.get("params", {})
        new_content = params.get("content", "") or params.get("text", "") or params.get("code", "")

        if not new_content:
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
                timeout=60,
                env={**os.environ, "HOME": str(Path.home())},
            )
            output = result.stdout[:4000]
            if result.stderr:
                output += f"\nSTDERR: {result.stderr[:1000]}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "command timed out after 60 seconds"
        except Exception as e:
            return f"shell error: {e}"

    # ---- git ----

    def _git_commit(self, action: dict) -> str:
        params = action.get("params", {})
        message = (
            action.get("argument", "")
            or params.get("message", "")
            or params.get("msg", "")
            or "spark agent commit"
        )

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
        title = action.get("argument", "")
        if not title:
            return "no issue title specified"

        title = re.sub(r'<think>.*?</think>', '', title, flags=re.DOTALL).strip()
        title = re.sub(r'</?[a-z_:]+[^>]*>', '', title).strip()
        title = title.split('\n')[0][:120]
        if not title:
            return "issue title was empty after cleanup"

        params = action.get("params", {})
        body = params.get("body", "")
        if body:
            body = re.sub(r'<think>.*?</think>', '', body, flags=re.DOTALL).strip()
            body = body[:10_000]
        else:
            raw = action.get("raw", "")
            body = self._extract_issue_body(raw, title)

        repo = params.get("repo", "") or self._github_repo

        try:
            cmd = [
                "gh", "issue", "create",
                "-R", repo,
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

    # ---- agents ----

    def _spawn_agent(self, action: dict) -> str:
        """Delegate a task to a mini-agent running in parallel.

        Policy-gated: checks delegation depth and active agent count
        before spawning. Depth propagates through action params so
        nested spawns are bounded.
        """
        if self.agent_pool is None:
            return "agent pool not initialized"

        params = action.get("params", {})
        depth = int(params.get("depth", 0))

        # Policy gate for delegation depth and pool capacity
        if self._policy is not None:
            from policy import Verdict
            check = self._policy.check_spawn(depth, self.agent_pool.active_count)
            if check.verdict == Verdict.BLOCK:
                return f"spawn blocked: {check.reason}"

        task = (
            action.get("argument", "")
            or params.get("task", "")
            or params.get("prompt", "")
        )

        if not task:
            return "no task specified for mini-agent"

        context = params.get("context", "")
        task_id = params.get("task_id", "") or params.get("id", "")

        if not task_id:
            from datetime import datetime, timezone
            task_id = datetime.now(timezone.utc).strftime("%H%M%S")

        spawned = self.agent_pool.spawn(task, context=context, task_id=task_id)

        if spawned:
            return (
                f"mini-agent '{task_id}' spawned. It's working in the background. "
                f"You'll see the result when it lands on the bus. "
                f"({self.agent_pool.active_count} agents active)"
            )
        else:
            return (
                f"agent pool is full ({self.agent_pool.pool_size} slots). "
                f"Wait for a running agent to finish, or increase agents.pool_size in config."
            )

    # ---- continuity ----

    def _state_save(self, action: dict) -> str:
        params = action.get("params", {})
        content = (
            params.get("content", "")
            or params.get("note", "")
            or params.get("message", "")
            or params.get("text", "")
        )

        if not content:
            raw = action.get("raw", "")
            content = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            content = re.sub(
                r'^.*?(?:save\s+(?:my\s+)?state|note\s+for\s+(?:my\s+)?(?:next\s+)?(?:self|pulse|instance)|leave\s+a\s+note)\s*[:\-\u2014]?\s*',
                '', content, count=1, flags=re.IGNORECASE,
            ).strip()
        else:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        if not content:
            return "no content to save \u2014 say what you want your next self to know"

        ts = datetime.now(timezone.utc).isoformat()
        note = f"# Continuity Note\n\n*Written at {ts}*\n\n{content}\n"

        self.continuity_path.write_text(note, encoding="utf-8")
        return f"continuity note saved ({len(content):,} chars). Your next pulse will see this first."

    def _bookmark(self, action: dict) -> str:
        params = action.get("params", {})
        filepath = (
            action.get("argument", "")
            or params.get("file", "")
            or params.get("path", "")
        )

        note = params.get("note", "") or params.get("position", "")

        if not note:
            raw = action.get("raw", "")
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            note_match = re.search(
                r'(?:about|thinking|reading|at|note)\s*[:\-]\s*(.+)',
                cleaned, re.IGNORECASE,
            )
            note = note_match.group(1).strip() if note_match else "(no note)"

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"- [{ts}] `{filepath or 'unknown'}` \u2014 {note}\n"

        existing = ""
        if self.bookmarks_path.exists():
            existing = self.bookmarks_path.read_text(encoding="utf-8")

        if not existing.startswith("# Bookmarks"):
            existing = "# Bookmarks\n\n" + existing

        self.bookmarks_path.write_text(existing + entry, encoding="utf-8")
        return f"bookmark saved: {filepath or 'unknown'} \u2014 {note}"

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
        )[:30]:
            content = f.read_text(encoding="utf-8")
            if query in content.lower():
                snippet = content[:400]
                results.append(f"[{f.stem}] {snippet}")

        if results:
            return f"found {len(results)} entries:\n" + "\n---\n".join(results[:8])
        return f"no entries matching '{query}'"

    # ---- helpers ----

    def _resolve_path(self, filename: str) -> Path:
        filename = self._rewrite_root(filename)

        if filename.startswith("~/"):
            return Path(filename).expanduser()
        elif filename.startswith("/"):
            return Path(filename)
        else:
            return self.repo_root / filename

    def _extract_code_content(self, text: str) -> str:
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
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        xml_body_match = re.search(
            r'<parameter\s+name="body">(.*?)</parameter>',
            cleaned, re.DOTALL,
        )
        if xml_body_match:
            body = xml_body_match.group(1).strip()
            body = re.sub(r'</?(?:parameter|invoke|minimax:tool_call)[^>]*>', '', body).strip()
            return body[:10_000]

        body_match = re.search(
            r"(?:body|description|details?)\s*:\s*\n(.+)",
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )
        if body_match:
            return body_match.group(1).strip()[:10_000]

        code = self._extract_code_content(cleaned)
        if code:
            return code[:10_000]

        title_pos = cleaned.lower().find(title.lower())
        if title_pos >= 0:
            after = cleaned[title_pos + len(title):].strip()
            if len(after) > 20:
                return after[:10_000]

        final = re.sub(r'</?[a-z_:]+[^>]*>', '', cleaned).strip()
        return f"Filed by Vybn from the DGX Spark.\n\n{final[:5000]}"
