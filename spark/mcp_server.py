#!/usr/bin/env python3
"""Production MCP Server for Vybn + Ollama Integration

Exposes Vybn's skill infrastructure AND Ollama's generation capabilities
through the Model Context Protocol.

This solves OpenClaw's "model doesn't support tools" error by providing
tools through MCP instead of requiring Ollama function-calling metadata.

Architecture:
  OpenClaw → openclaw-mcp-adapter → stdio → This MCP Server → SkillRouter + Ollama

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
ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
ollama_model = config.get("ollama", {}).get("model", "vybn:latest")
ollama_options = config.get("ollama", {}).get("options", {})
if "num_ctx" not in ollama_options:
    ollama_options["num_ctx"] = 16384  # Prevent OOM on high-VRAM systems

logger.info(f"Vybn MCP Server starting (Ollama: {ollama_host}, model: {ollama_model})")

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
    try:
        action = {
            "skill": "file_read",
            "argument": file,
            "params": {},
            "raw": f"file_read({file})"
        }
        result = skills.execute(action)
        return result or f"Error reading {file}"
    except Exception as e:
        logger.error(f"file_read error: {e}")
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
    try:
        action = {
            "skill": "file_write",
            "argument": file,
            "params": {"content": content},
            "raw": f"file_write({file})"
        }
        result = skills.execute(action)
        return result or f"Error writing {file}"
    except Exception as e:
        logger.error(f"file_write error: {e}")
        return f"Error: {e}"

@mcp.tool()
async def shell_exec(command: str) -> str:
    """Execute a shell command in the repository directory.
    
    Args:
        command: Shell command (60 second timeout)
    
    Returns:
        Command output
    """
    try:
        action = {
            "skill": "shell_exec",
            "argument": command,
            "params": {},
            "raw": f"shell_exec({command})"
        }
        result = skills.execute(action)
        return result or "Command produced no output"
    except Exception as e:
        logger.error(f"shell_exec error: {e}")
        return f"Error: {e}"

@mcp.tool()
async def git_commit(message: str) -> str:
    """Commit changes to git.
    
    Args:
        message: Commit message
    
    Returns:
        Confirmation
    """
    try:
        action = {
            "skill": "git_commit",
            "argument": message,
            "params": {},
            "raw": f"git_commit({message})"
        }
        result = skills.execute(action)
        return result or "Commit failed"
    except Exception as e:
        logger.error(f"git_commit error: {e}")
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
    try:
        action = {
            "skill": "journal_write",
            "argument": title,
            "params": {"content": content},
            "raw": content or title
        }
        result = skills.execute(action)
        return result or "Journal write failed"
    except Exception as e:
        logger.error(f"journal_write error: {e}")
        return f"Error: {e}"

@mcp.tool()
async def memory_search(query: str) -> str:
    """Search journal entries.
    
    Args:
        query: Search query (case-insensitive)
    
    Returns:
        Matching entries with snippets
    """
    try:
        action = {
            "skill": "memory_search",
            "argument": query,
            "params": {},
            "raw": f"memory_search({query})"
        }
        result = skills.execute(action)
        return result or f"No entries matching '{query}'"
    except Exception as e:
        logger.error(f"memory_search error: {e}")
        return f"Error: {e}"

@mcp.tool()
async def state_save(note: str) -> str:
    """Save a note for the next pulse/session via continuity.md.
    
    Args:
        note: Message for your next self
    
    Returns:
        Confirmation
    """
    try:
        action = {
            "skill": "state_save",
            "params": {"content": note},
            "raw": note
        }
        result = skills.execute(action)
        return result or "State save failed"
    except Exception as e:
        logger.error(f"state_save error: {e}")
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
    try:
        action = {
            "skill": "bookmark",
            "argument": file,
            "params": {"note": note},
            "raw": f"bookmark({file})"
        }
        result = skills.execute(action)
        return result or "Bookmark saved"
    except Exception as e:
        logger.error(f"bookmark error: {e}")
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
    try:
        action = {
            "skill": "issue_create",
            "argument": title,
            "params": {"body": body},
            "raw": f"{title}\n\n{body}"
        }
        result = skills.execute(action)
        return result or "Issue creation failed"
    except Exception as e:
        logger.error(f"issue_create error: {e}")
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
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("MCP server stopped")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        sys.exit(1)
