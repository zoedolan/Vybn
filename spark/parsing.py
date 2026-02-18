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
