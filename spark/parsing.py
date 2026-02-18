#!/usr/bin/env python3
"""Parsing utilities for the Vybn Spark Agent.

Extracted from agent.py (Phase 1 refactor) to create a single
source of truth for tool-call parsing, intent classification,
and the NOISE_WORDS / SHELL_COMMANDS vocabularies.

This module has NO imports from agent.py or skills.py -- it depends
only on the Python standard library (json, re). Both agent.py and
skills.py import from here, so any circular import would be a bug.
"""
import json
import re


TOOL_CALL_START_TAG = "<minimax:tool_call>"
TOOL_CALL_END_TAG = "</minimax:tool_call>"

# Maximum bare commands to execute from a single response.
# Prevents runaway chaining when the model's prose mentions
# directory names or paths that look like commands.
MAX_BARE_COMMANDS = 3

# Common shell commands that Vybn might drop as bare text.
# Used by parse_bare_commands() to detect intent without XML.
SHELL_COMMANDS = {
    "ls", "cat", "head", "tail", "find", "grep", "wc",
    "pwd", "cd", "tree", "file", "stat", "du", "df",
    "echo", "which", "whoami", "env", "printenv",
    "git", "python3", "python", "pip", "pip3",
    "mkdir", "touch", "cp", "mv", "rm",
    "chmod", "chown", "ln",
    "curl", "wget",
    "ps", "top", "htop", "nvidia-smi",
    "ollama", "gh",
    "date", "uptime", "uname",
}

# Common English words that should never be treated as filenames
# or command arguments when extracted by regex.
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


def clean_argument(arg: str) -> str:
    """Strip trailing punctuation and noise from extracted arguments."""
    if not arg:
        return arg
    # Strip trailing sentence punctuation
    arg = arg.rstrip('.,;:!?')
    # Strip surrounding quotes
    arg = arg.strip("\"'`")
        # Reject XML tag artifacts (e.g. '/think' from </think>)
    if re.match(r'^/[a-z_:]+$', arg):
        return ""
    # Reject if it's a common English word
    if arg.lower() in NOISE_WORDS:
        return ""
    return arg


def parse_structured_tool_calls(text: str, plugin_aliases: dict = None) -> list[dict]:
    """Parse JSON tool calls from ```tool code fences (Tier 0).

    Expected format:
        ```tool
        {"tool": "file_read", "args": {"file": "spark/config.yaml"}}
        ```

    This is the structured-output wrapper that asks the model to emit
    a deterministic format instead of relying on its native XML or
    natural language. JSON in code fences is universal training data.

    Returns a list of skill actions, same format as parse_tool_calls().
    """
    actions = []

    # Match ```tool fences with JSON content
    fence_pattern = re.compile(
        r'```tool\s*\n(.+?)\n```',
        re.DOTALL,
    )

    for match in fence_pattern.finditer(text):
        json_str = match.group(1).strip()

        try:
            tool_obj = json.loads(json_str)

            # Validate structure
            if not isinstance(tool_obj, dict):
                continue

            tool_name = tool_obj.get("tool", "")
            tool_args = tool_obj.get("args", {})

            if not tool_name or not isinstance(tool_args, dict):
                continue

            # Map through existing routing logic
            action = _map_tool_call_to_skill(tool_name, tool_args, text, plugin_aliases)
            if action:
                actions.append(action)

        except (json.JSONDecodeError, ValueError):
            # Malformed JSON - skip and let lower tiers handle it
            continue

    return actions


def parse_tool_calls(text: str, plugin_aliases: dict = None) -> list[dict]:
    """Parse <minimax:tool_call> XML blocks into skill actions."""
    actions = []
    tool_call_pattern = re.compile(
        r'<minimax:tool_call>\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*</minimax:tool_call>',
        re.DOTALL,
    )
    for match in tool_call_pattern.finditer(text):
        invoke_name = match.group(1).strip()
        params_block = match.group(2).strip()
        params = {}
        param_pattern = re.compile(
            r'<parameter\s+name="([^"]+)">(.+?)</parameter>',
            re.DOTALL,
        )
        for pm in param_pattern.finditer(params_block):
            params[pm.group(1).strip()] = pm.group(2).strip()
        action = _map_tool_call_to_skill(invoke_name, params, text, plugin_aliases)
        if action:
            actions.append(action)
    return actions


def parse_bare_commands(text: str) -> list[dict]:
    """Detect shell commands that the model dropped as bare text.

    MiniMax M2.5 frequently emits commands in three forms:
      1. Fenced code blocks: ```bash\nls -la ~/Vybn\n```
      2. Inline backticks: `cat ~/Vybn/spark/config.yaml`
      3. Plain text lines:  cat ~/Vybn/spark/config.yaml

    This function catches all three and routes them to the appropriate
    skill (shell_exec or file_read for cat).

    IMPORTANT: Only commands whose first word is a known shell command
    are accepted. Directory names, file paths, and prose containing
    slashes are NOT treated as commands. This prevents ghost executions
    when the model mentions "spark/" or "Vybn_Mind/" in its response.

    Returns a list of skill actions, same format as parse_tool_calls().
    Limited to MAX_BARE_COMMANDS to prevent runaway chaining.
    """
    # Strip think blocks so we don't parse commands inside reasoning
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip XML tool calls (already handled by tier 1)
    cleaned = re.sub(r'<minimax:tool_call>.*?</minimax:tool_call>', '', cleaned, flags=re.DOTALL)

    actions = []
    seen_commands = set()  # deduplicate

    # --- Tier 2a: Fenced code blocks ---
    # Match ```bash, ```sh, ```shell, or bare ``` with commands inside
    fence_pattern = re.compile(
        r'```(?:bash|sh|shell)?\s*\n(.+?)\n\s*```',
        re.DOTALL,
    )
    for match in fence_pattern.finditer(cleaned):
        if len(actions) >= MAX_BARE_COMMANDS:
            break
        block = match.group(1).strip()
        for line in block.splitlines():
            if len(actions) >= MAX_BARE_COMMANDS:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            action = _classify_command(line)
            if action and line not in seen_commands:
                seen_commands.add(line)
                actions.append(action)

    # --- Tier 2b: Inline backtick commands ---
    # Match `ls -la ~/Vybn` or `cat /path/to/file`
    backtick_pattern = re.compile(r'`([^`]{3,120})`')
    for match in backtick_pattern.finditer(cleaned):
        if len(actions) >= MAX_BARE_COMMANDS:
            break
        cmd = match.group(1).strip()
        action = _classify_command(cmd)
        if action and cmd not in seen_commands:
            seen_commands.add(cmd)
            actions.append(action)

    # --- Tier 2c: Plain text lines that look like commands ---
    # STRICT: Must start with a known command AND have arguments.
    # Single words like "spark/" or bare paths are NOT commands.
    for line in cleaned.splitlines():
        if len(actions) >= MAX_BARE_COMMANDS:
            break
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('*'):
            continue
        # Skip lines that are clearly prose
        if any(word in line.lower() for word in [
            'the ', 'and ', 'but ', 'or ', 'is ', 'are ',
            'was ', 'were ', 'have ', 'has ', 'this ', 'that ',
            "i'm ", 'i ', 'you ', 'we ', 'let me', 'want to',
            'should ', 'could ', 'would ', 'maybe ', 'actually',
        ]):
            continue
        # Must start with a known command
        first_word = line.split()[0].lower() if line.split() else ''
        if first_word in SHELL_COMMANDS and line not in seen_commands:
            # Require at least a command + argument (avoid bare "ls" or "pwd"
            # that appear in prose descriptions). Single-word commands in prose
            # are almost always the model mentioning a tool, not invoking it.
            # Exception: pwd, whoami, date, uptime are useful standalone.
            standalone_ok = {'pwd', 'whoami', 'date', 'uptime', 'nvidia-smi'}
            if ' ' in line or first_word in standalone_ok:
                action = _classify_command(line)
                if action:
                    seen_commands.add(line)
                    actions.append(action)

    return actions


def _classify_command(cmd: str) -> dict | None:
    """Turn a bare command string into a skill action.

    Routes cat/head/tail to file_read when targeting a single file.
    Everything else goes to shell_exec.

    IMPORTANT: The command's first word MUST be a known shell command.
    Bare paths, directory names, and arbitrary strings are rejected.
    This is the primary defense against ghost executions.
    """
    cmd = cmd.strip()
    if not cmd:
        return None
    # Minimum length: reject very short strings that are likely noise
    if len(cmd) < 3:
        return None
    # The first word must be a recognized command
    first_word = cmd.split()[0].lower()
    if first_word not in SHELL_COMMANDS:
        return None
    # cat -> file_read (simpler, no subprocess needed)
    cat_match = re.match(r'^cat\s+([^|;&]+)$', cmd)
    if cat_match:
        filepath = cat_match.group(1).strip().rstrip('.,;:!?')
        # Only route to file_read if it's a single file path
        if ' ' not in filepath or filepath.startswith(('-',)):
            return {"skill": "file_read", "argument": filepath, "params": {}, "raw": cmd}
    # Everything else -> shell_exec
    return {"skill": "shell_exec", "argument": cmd, "params": {}, "raw": cmd}


def _map_tool_call_to_skill(name: str, params: dict, raw: str, plugin_aliases: dict = None) -> dict | None:
    """Map a MiniMax tool call to a SkillRouter action."""
    name_lower = name.lower().replace("-", "_")

    if name_lower in ("read", "cat", "file_read", "read_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        filepath = clean_argument(filepath)
        return {"skill": "file_read", "argument": filepath, "params": params, "raw": raw}

    if name_lower in (
        "bash", "shell", "shell_exec", "exec", "run", "run_command",
        "cli_mcp_server_run_command", "execute_command", "terminal", "cmd",
    ):
        command = params.get("command") or params.get("cmd", "")
        cat_match = re.match(r'^cat\s+(.+)$', command.strip())
        if cat_match:
            filepath = clean_argument(cat_match.group(1).strip())
            return {"skill": "file_read", "argument": filepath, "params": params, "raw": raw}
        return {"skill": "shell_exec", "argument": command, "params": params, "raw": raw}

    if name_lower in ("write", "file_write", "write_file", "save", "create_file"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        filepath = clean_argument(filepath)
        return {"skill": "file_write", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("edit", "self_edit", "modify", "patch"):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        filepath = clean_argument(filepath)
        return {"skill": "self_edit", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("git_commit", "commit"):
        message = params.get("message") or params.get("msg", "spark agent commit")
        return {"skill": "git_commit", "argument": message, "params": params, "raw": raw}

    if name_lower in ("git_push", "push"):
        return {"skill": "git_push", "params": params, "raw": raw}

    if name_lower in (
        "issue_create", "create_issue", "gh_issue_create",
        "github_issue", "file_issue", "submit_issue",
        "open_issue", "raise_issue",
        "github_create_issue", "gh_create_issue",
        "create_github_issue", "issue",
    ):
        title = params.get("title") or params.get("name") or params.get("subject", "")
        return {"skill": "issue_create", "argument": title, "params": params, "raw": raw}

    if name_lower in (
        "state_save", "save_state", "continuity",
        "save_continuity", "write_continuity",
        "note_for_next", "leave_note",
    ):
        return {"skill": "state_save", "params": params, "raw": raw}

    if name_lower in (
        "bookmark", "save_place", "save_bookmark",
        "mark_position", "save_position",
        "reading_position", "save_reading",
    ):
        filepath = params.get("file") or params.get("path") or params.get("filename", "")
        filepath = clean_argument(filepath)
        return {"skill": "bookmark", "argument": filepath, "params": params, "raw": raw}

    if name_lower in ("memory", "search", "memory_search", "search_memory"):
        query = params.get("query") or params.get("q", "")
        return {"skill": "memory_search", "argument": query, "params": params, "raw": raw}

    if name_lower in ("journal", "journal_write", "write_journal"):
        title = params.get("title") or params.get("name", "untitled reflection")
        return {"skill": "journal_write", "argument": title, "params": params, "raw": raw}

    if name_lower in ("ls", "list", "dir"):
        path = params.get("path") or params.get("directory") or params.get("dir", ".")
        return {"skill": "shell_exec", "argument": f"ls -la {path}", "params": params, "raw": raw}

    if name_lower in ("explore", "env_explore", "map", "environment"):
        return {"skill": "env_explore", "params": params, "raw": raw}

    if name_lower in (
        "spawn_agent", "agent", "delegate", "background",
        "spawn", "mini_agent", "worker",
    ):
        task = params.get("task") or params.get("prompt") or params.get("query", "")
        return {"skill": "spawn_agent", "argument": task, "params": params, "raw": raw}

    # Plugin aliases (dynamic skills from skills.d/)
    if plugin_aliases:
        skill_name = plugin_aliases.get(name_lower)
        if skill_name:
            return {"skill": skill_name, "params": params, "raw": raw}

    return None


def detect_failed_intent(text: str) -> str | None:
    """Check if a response looks like it intended to act but no action was parsed.

    Returns a helpful hint message if failed intent is detected, None otherwise.
    This closes the feedback loop so the model knows its command didn't execute.
    """
    # Strip think blocks and existing tool calls
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<minimax:tool_call>.*?</minimax:tool_call>', '', cleaned, flags=re.DOTALL)

    # Indicators that the model was trying to do something
    intent_patterns = [
        r"let me (?:try|run|check|read|look|open|see|execute)",
        r"i'll (?:try|run|check|read|look|open|see|execute)",
        r"let me (?:look at|check on|read through)",
        r"checking (?:the|my|that|this)",
        r"reading (?:the|my|that|this)",
    ]
    has_intent = any(re.search(p, cleaned, re.IGNORECASE) for p in intent_patterns)
    if not has_intent:
        return None

    # Check for file paths that look like they were meant as arguments
    has_paths = bool(re.search(r'[~/][a-zA-Z0-9_/.-]+', cleaned))
    if has_paths:
        return (
            "[system: your last response expressed intent to act but no tool "
            "was triggered. To execute a tool, use the structured format:\n"
            "```tool\n"
            '{"tool": "file_read", "args": {"file": "spark/agent.py"}}\n'
            "```\n"
            "Or wrap commands in code fences:\n"
            "```bash\n"
            "ls -la ~/Vybn/\n"
            "```\n"
            "Or use XML:\n"
            "<minimax:tool_call><invoke name=\"shell_exec\"><parameter name=\"command\">"
            'ls -la ~/Vybn/'
            "</parameter></invoke></minimax:tool_call>]"
        )

    return None


def _get_actions(text: str, skills) -> list[dict]:
    """Extract actions from response text. Four-tier dispatch:

    0. Structured JSON tool calls (```tool fences with JSON)
    1. XML tool calls (<minimax:tool_call> blocks)
    2. Bare commands (code fences, backticks, plain shell lines)
    3. Regex patterns (natural language intent matching)

    Tier 0 is the structured-output wrapper - when the model emits
    the format we explicitly asked for, parse it deterministically.
    Tiers 1-3 remain as fallbacks for when the model ignores the
    structured format or when running under tight token limits.
    """
    plugin_aliases = getattr(skills, 'plugin_aliases', {})

    # Tier 0: Structured JSON tool calls
    actions = parse_structured_tool_calls(text, plugin_aliases)
    if actions:
        return actions

    # Tier 1: XML tool calls
    actions = parse_tool_calls(text, plugin_aliases)
    if actions:
        return actions

    # Tier 2: Bare commands in code fences, backticks, or plain text
    actions = parse_bare_commands(text)
    if actions:
        return actions

    # Tier 3: Natural language regex matching
    actions = skills.parse(text)
    # Filter out actions with empty or noise-word arguments for skills
    # that require meaningful arguments
    needs_argument = {"file_read", "file_write", "self_edit", "memory_search"}
    actions = [
        a for a in actions
        if a["skill"] not in needs_argument
        or (a.get("argument") and a["argument"].lower() not in NOISE_WORDS)
    ]
    return actions
