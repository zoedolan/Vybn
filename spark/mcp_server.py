#!/usr/bin/env python3
"""MCP Server for Vybn Skills

Exposes the SkillRouter from skills.py as an MCP (Model Context Protocol)
server that OpenClaw or other MCP clients can connect to.

This bridges OpenClaw's tool dispatch to the existing Vybn skill
infrastructure without requiring Ollama to expose function-calling
metadata that MiniMax M2.5 doesn't advertise.

Usage:
    python3 spark/mcp_server.py

OpenClaw config (openclaw.json):
    {
      "mcpServers": {
        "vybn-skills": {
          "command": "python3",
          "args": ["/home/vybnz69/Vybn/spark/mcp_server.py"]
        }
      }
    }

Architecture:
    OpenClaw → stdio → MCP Server → SkillRouter → actual tools
                ↑                        ↓
              This file              skills.py

The MCP server is stateless — it loads config, initializes the
SkillRouter, and dispatches tool calls. All state (sessions,
memory, policy) lives in the skill implementations.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import from spark/
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp library not installed.", file=sys.stderr)
    print("Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

import yaml
from skills import SkillRouter


def load_config() -> dict:
    """Load config.yaml from the spark directory."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Initialize the MCP server
mcp = FastMCP("vybn-skills")

# Load config and initialize the skill router
config = load_config()
skills = SkillRouter(config)


# ---- Tool definitions ----
# Each @mcp.tool() decorator exposes a skill to OpenClaw.
# The skill router handles dispatch, policy checks, and execution.


@mcp.tool()
async def file_read(file: str) -> str:
    """Read a file from the repository.
    
    Args:
        file: Path to the file (relative to repo root, or absolute with ~/)
    
    Returns:
        File contents, or error message if file not found
    """
    action = {
        "skill": "file_read",
        "argument": file,
        "params": {},
        "raw": f"file_read({file})"
    }
    result = skills.execute(action)
    return result or f"Error reading {file}"


@mcp.tool()
async def file_write(file: str, content: str) -> str:
    """Write content to a file in the repository.
    
    Args:
        file: Path to the file (relative to repo root, or absolute with ~/)
        content: Text content to write
    
    Returns:
        Confirmation message or error
    """
    action = {
        "skill": "file_write",
        "argument": file,
        "params": {"content": content},
        "raw": f"file_write({file})"
    }
    result = skills.execute(action)
    return result or f"Error writing {file}"


@mcp.tool()
async def shell_exec(command: str) -> str:
    """Execute a shell command in the repository directory.
    
    Args:
        command: Shell command to execute (60 second timeout)
    
    Returns:
        Command output (stdout + stderr), or error message
    """
    action = {
        "skill": "shell_exec",
        "argument": command,
        "params": {},
        "raw": f"shell_exec({command})"
    }
    result = skills.execute(action)
    return result or "Command produced no output"


@mcp.tool()
async def git_commit(message: str) -> str:
    """Commit changes to git with a message.
    
    Args:
        message: Commit message
    
    Returns:
        Confirmation or error message
    """
    action = {
        "skill": "git_commit",
        "argument": message,
        "params": {},
        "raw": f"git_commit({message})"
    }
    result = skills.execute(action)
    return result or "Commit failed"


@mcp.tool()
async def journal_write(title: str, content: str = "") -> str:
    """Write a journal entry.
    
    Args:
        title: Entry title or subject
        content: Entry body (optional, can be in raw if not provided)
    
    Returns:
        Confirmation with filename
    """
    action = {
        "skill": "journal_write",
        "argument": title,
        "params": {"content": content},
        "raw": content or title
    }
    result = skills.execute(action)
    return result or "Journal write failed"


@mcp.tool()
async def memory_search(query: str) -> str:
    """Search journal entries for a query string.
    
    Args:
        query: Search query (case-insensitive)
    
    Returns:
        Matching entries with snippets, or "no entries found"
    """
    action = {
        "skill": "memory_search",
        "argument": query,
        "params": {},
        "raw": f"memory_search({query})"
    }
    result = skills.execute(action)
    return result or f"No entries matching '{query}'"


@mcp.tool()
async def self_edit(file: str, content: str) -> str:
    """Edit a source file in the spark/ directory.
    
    This is restricted to spark/ to prevent accidental damage to
    the broader repo. For new skills, prefer creating plugins in
    skills.d/ instead of editing core files.
    
    Args:
        file: Filename in spark/ (e.g., "skills.py")
        content: Complete new file content
    
    Returns:
        Confirmation or error (includes backup location)
    """
    action = {
        "skill": "self_edit",
        "argument": file,
        "params": {"content": content},
        "raw": f"self_edit({file})"
    }
    result = skills.execute(action)
    return result or f"Self-edit failed for {file}"


@mcp.tool()
async def state_save(note: str) -> str:
    """Save a note for the next pulse/session via continuity.md.
    
    Args:
        note: Message for your next self
    
    Returns:
        Confirmation with character count
    """
    action = {
        "skill": "state_save",
        "params": {"content": note},
        "raw": note
    }
    result = skills.execute(action)
    return result or "State save failed"


@mcp.tool()
async def bookmark(file: str, note: str = "") -> str:
    """Bookmark a reading position in a file.
    
    Args:
        file: File path
        note: Optional note about position or context
    
    Returns:
        Confirmation
    """
    action = {
        "skill": "bookmark",
        "argument": file,
        "params": {"note": note},
        "raw": f"bookmark({file})"
    }
    result = skills.execute(action)
    return result or "Bookmark saved"


@mcp.tool()
async def issue_create(title: str, body: str = "") -> str:
    """Create a GitHub issue in the Vybn repository.
    
    Requires gh CLI authentication (run setup-gh-auth.sh first).
    
    Args:
        title: Issue title
        body: Issue body/description
    
    Returns:
        Issue URL if successful, error otherwise
    """
    action = {
        "skill": "issue_create",
        "argument": title,
        "params": {"body": body},
        "raw": f"{title}\n\n{body}"
    }
    result = skills.execute(action)
    return result or "Issue creation failed"


# ---- Server lifecycle ----

if __name__ == "__main__":
    # Run the MCP server on stdio transport (standard for OpenClaw)
    mcp.run(transport="stdio")
