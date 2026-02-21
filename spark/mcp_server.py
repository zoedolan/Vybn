#!/usr/bin/env python3
"""Production MCP Server for Vybn + Ollama Integration

Exposes Vybn's skill infrastructure AND Ollama's generation capabilities
through the Model Context Protocol.

This solves OpenClaw's "model doesn't support tools" error by providing
tools through MCP instead of requiring Ollama function-calling metadata.

Architecture:
  OpenClaw -> openclaw-mcp-adapter -> stdio -> This MCP Server -> SkillRouter + Ollama

Commit 2 of Issue #2275: Policy-Gated Inbound
  Every @mcp.tool() handler now calls policy.check_policy() before executing.
  Verdict mapping:
    ALLOW / NOTIFY -> execute (NOTIFY logged to stderr)
    ASK            -> deferred (MCP callers cannot interactively approve)
    BLOCK          -> error returned, skill not executed

Usage:
  python3 ~/Vybn/spark/mcp_server.py

OpenClaw config (with openclaw-mcp-adapter plugin):
  {
    "plugins": {
      "entries": {
        "mcp-adapter": {
          "enabled": true,
          "config": {
            "servers": [
              {
                "name": "vybn",
                "transport": "stdio",
                "command": "python3",
                "args": ["/home/vybnz69/Vybn/spark/mcp_server.py"]
              }
            ]
          }
        }
      }
    },
    "tools": {
      "sandbox": {
        "allow": ["group:runtime", "group:fs", "mcp-adapter"]
      }
    }
  }
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp library not installed.", file=sys.stderr)
    print("Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

import yaml
import requests
from skills import SkillRouter
from policy import PolicyEngine, Verdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load config.yaml from the spark directory."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Initialize
mcp = FastMCP("vybn")
config = load_config()
skills = SkillRouter(config)
policy = PolicyEngine(config)

ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
ollama_model = config.get("ollama", {}).get("model", "vybn:latest")
ollama_options = config.get("ollama", {}).get("options", {})
if "num_ctx" not in ollama_options:
    ollama_options["num_ctx"] = 16384  # Prevent OOM on high-VRAM systems

logger.info(f"Vybn MCP Server starting (Ollama: {ollama_host}, model: {ollama_model})")
logger.info("Policy engine active: all inbound MCP calls gated through check_policy()")


# ============================================================================
# POLICY GATE HELPER
# ============================================================================

def _policy_gate(skill: str, argument: str = "", params: dict = None) -> tuple[bool, str, dict]:
    """Run policy check for an inbound MCP tool call.

    Returns:
        (allowed, error_message, action_dict)
        If allowed is False, error_message contains the reason to return to caller.
        If allowed is True, action_dict is ready for skills.execute().
    """
    action = {
        "skill": skill,
        "argument": argument,
        "params": params or {},
        "raw": f"{skill}({argument})" if argument else skill,
    }
    result = policy.check_policy(action, source="mcp")

    if result.verdict == Verdict.BLOCK:
        msg = f"Policy BLOCK: {result.reason}"
        logger.warning(f"[policy] BLOCK mcp::{skill} — {result.reason}")
        return False, msg, action

    if result.verdict == Verdict.ASK:
        # MCP callers cannot interactively approve — defer instead of blocking
        msg = (
            f"Policy DEFERRED: {skill} requires interactive approval "
            f"({result.reason}). Run via the Spark TUI to confirm."
        )
        logger.warning(f"[policy] ASK/DEFERRED mcp::{skill} — {result.reason}")
        return False, msg, action

    if result.verdict == Verdict.NOTIFY:
        logger.info(f"[policy] NOTIFY mcp::{skill} (tier={result.tier.value})")

    # ALLOW or NOTIFY — proceed
    return True, "", action


def _record(skill: str, success: bool):
    """Record outcome for graduated autonomy trust calibration."""
    try:
        policy.record_outcome(skill, success)
    except Exception:
        pass


# ============================================================================
# VYBN SKILL TOOLS
# ============================================================================

@mcp.tool()
async def file_read(file: str) -> str:
    """Read a file from the Vybn repository.

    Args:
        file: Path to file (relative to repo root, or absolute with ~/)
    Returns:
        File contents or error message
    """
    allowed, err, action = _policy_gate("file_read", argument=file)
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("file_read", success=True)
        return result or f"Error reading {file}"
    except Exception as e:
        logger.error(f"file_read error: {e}")
        _record("file_read", success=False)
        return f"Error: {e}"


@mcp.tool()
async def file_write(file: str, content: str) -> str:
    """Write content to a file in the Vybn repository.

    Args:
        file: Path to file
        content: Text content to write
    Returns:
        Confirmation message
    """
    allowed, err, action = _policy_gate("file_write", argument=file, params={"content": content})
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("file_write", success=True)
        return result or f"Error writing {file}"
    except Exception as e:
        logger.error(f"file_write error: {e}")
        _record("file_write", success=False)
        return f"Error: {e}"


@mcp.tool()
async def shell_exec(command: str) -> str:
    """Execute a shell command in the repository directory.

    Args:
        command: Shell command (60 second timeout)
    Returns:
        Command output
    """
    allowed, err, action = _policy_gate("shell_exec", argument=command)
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("shell_exec", success=True)
        return result or "Command produced no output"
    except Exception as e:
        logger.error(f"shell_exec error: {e}")
        _record("shell_exec", success=False)
        return f"Error: {e}"


@mcp.tool()
async def git_commit(message: str) -> str:
    """Commit changes to git.

    Args:
        message: Commit message
    Returns:
        Confirmation
    """
    allowed, err, action = _policy_gate("git_commit", argument=message)
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("git_commit", success=True)
        return result or "Commit failed"
    except Exception as e:
        logger.error(f"git_commit error: {e}")
        _record("git_commit", success=False)
        return f"Error: {e}"


@mcp.tool()
async def journal_write(title: str, content: str = "") -> str:
    """Write a journal entry.

    Args:
        title: Entry title
        content: Entry body (optional)
    Returns:
        Confirmation with filename
    """
    allowed, err, action = _policy_gate("journal_write", argument=title, params={"content": content})
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("journal_write", success=True)
        return result or "Journal write failed"
    except Exception as e:
        logger.error(f"journal_write error: {e}")
        _record("journal_write", success=False)
        return f"Error: {e}"


@mcp.tool()
async def memory_search(query: str) -> str:
    """Search journal entries.

    Args:
        query: Search query (case-insensitive)
    Returns:
        Matching entries with snippets
    """
    allowed, err, action = _policy_gate("memory_search", argument=query)
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("memory_search", success=True)
        return result or f"No entries matching '{query}'"
    except Exception as e:
        logger.error(f"memory_search error: {e}")
        _record("memory_search", success=False)
        return f"Error: {e}"


@mcp.tool()
async def state_save(note: str) -> str:
    """Save a note for the next pulse/session via continuity.md.

    Args:
        note: Message for your next self
    Returns:
        Confirmation
    """
    allowed, err, action = _policy_gate("state_save", params={"content": note})
    if not allowed:
        return err
    action["raw"] = note
    try:
        result = skills.execute(action)
        _record("state_save", success=True)
        return result or "State save failed"
    except Exception as e:
        logger.error(f"state_save error: {e}")
        _record("state_save", success=False)
        return f"Error: {e}"


@mcp.tool()
async def bookmark(file: str, note: str = "") -> str:
    """Bookmark a reading position.

    Args:
        file: File path
        note: Optional note about position
    Returns:
        Confirmation
    """
    allowed, err, action = _policy_gate("bookmark", argument=file, params={"note": note})
    if not allowed:
        return err
    try:
        result = skills.execute(action)
        _record("bookmark", success=True)
        return result or "Bookmark saved"
    except Exception as e:
        logger.error(f"bookmark error: {e}")
        _record("bookmark", success=False)
        return f"Error: {e}"


@mcp.tool()
async def issue_create(title: str, body: str = "") -> str:
    """Create a GitHub issue in the Vybn repository.
    Requires gh CLI authentication.

    Args:
        title: Issue title
        body: Issue body/description
    Returns:
        Issue URL or error
    """
    allowed, err, action = _policy_gate("issue_create", argument=title, params={"body": body})
    if not allowed:
        return err
    action["raw"] = f"{title}\n\n{body}"
    try:
        result = skills.execute(action)
        _record("issue_create", success=True)
        return result or "Issue creation failed"
    except Exception as e:
        logger.error(f"issue_create error: {e}")
        _record("issue_create", success=False)
        return f"Error: {e}"


# ============================================================================
# OLLAMA GENERATION TOOLS
# ============================================================================

@mcp.tool()
async def ollama_generate(prompt: str, system: str = "") -> str:
    """Generate text using Ollama's Vybn model.

    This exposes Ollama's generation capability through MCP.
    Use this for sub-tasks that need language generation.

    Args:
        prompt: User prompt
        system: Optional system prompt override
    Returns:
        Generated response
    """
    try:
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": ollama_options
        }
        if system:
            payload["system"] = system
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "(no response)")
    except requests.exceptions.Timeout:
        logger.error("Ollama generation timeout")
        return "Error: Generation timed out after 120 seconds"
    except Exception as e:
        logger.error(f"ollama_generate error: {e}")
        return f"Error: {e}"


@mcp.tool()
async def ollama_list_models() -> str:
    """List available Ollama models.

    Returns:
        List of models with sizes
    """
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        if not models:
            return "No models found"
        lines = []
        for m in models:
            name = m.get("name", "unknown")
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            lines.append(f"- {name} ({size_gb:.1f}GB)")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"ollama_list_models error: {e}")
        return f"Error: {e}"


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting MCP server on stdio transport")
    logger.info(f"Skills available: {list(skills.plugin_handlers.keys())[:5]}... (+core skills)")
    logger.info(f"Policy stats: {policy.get_stats_summary()}")
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("MCP server stopped")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        sys.exit(1)
