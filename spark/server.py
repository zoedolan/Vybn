"""
Vybn SSH MCP Server
===================
Bridges Perplexity Computer to the DGX Sparks via SSH.
Exposes MCP tools over Streamable HTTP, authenticated by API key,
served behind Tailscale Funnel.

Architecture:
    Perplexity ──HTTPS──▶ Tailscale Funnel ──▶ This Server ──SSH──▶ Spark(s)

Tools exposed:
    - shell_exec: Run a command on a Spark (with safety rails)
    - read_file: Read a file from a Spark
    - write_file: Write content to a file on a Spark
    - gpu_status: Quick GPU utilization check
    - sensorium: Run the Vybn sensorium and return perception
    - model_status: Check what models/services are running
    - repo_status: Git status and recent log for the Vybn repo
"""

import asyncio
import json
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

import asyncssh
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vybn-mcp")

# --- Configuration ---

API_KEY = os.environ["VYBN_MCP_API_KEY"]

MACHINES = {}
for i in range(1, 10):
    host = os.environ.get(f"SPARK{i}_HOST")
    if host:
        MACHINES[f"spark-{i}"] = {
            "host": host,
            "port": int(os.environ.get(f"SPARK{i}_PORT", "22")),
            "user": os.environ.get(f"SPARK{i}_USER", "zoe"),
            "key": os.environ.get(f"SPARK{i}_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
        }

if not MACHINES:
    logger.warning("No SPARK*_HOST env vars found. Add SPARK1_HOST etc. to .env")

DEFAULT_MACHINE = os.environ.get("DEFAULT_MACHINE", next(iter(MACHINES), "spark-1"))

# Safety: commands that require explicit confirmation
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r /", "mkfs", "dd if=", "reboot", "shutdown",
    "systemctl stop", "kill -9", "> /dev/sd", "chmod -R 777",
]

# --- SSH Connection Pool ---

_connections: dict[str, asyncssh.SSHClientConnection] = {}


async def get_connection(machine: str) -> asyncssh.SSHClientConnection:
    """Get or create an SSH connection to a machine."""
    if machine not in MACHINES:
        raise ValueError(f"Unknown machine: {machine}. Available: {list(MACHINES.keys())}")

    if machine in _connections:
        try:
            # Test if connection is still alive
            await _connections[machine].run("echo ok", check=True, timeout=5)
            return _connections[machine]
        except Exception:
            _connections.pop(machine, None)

    cfg = MACHINES[machine]
    conn = await asyncssh.connect(
        cfg["host"],
        port=cfg["port"],
        username=cfg["user"],
        client_keys=[cfg["key"]],
        known_hosts=None,  # Tailscale network is trusted
    )
    _connections[machine] = conn
    return conn


async def run_ssh(machine: str, command: str, timeout: int = 60) -> dict:
    """Execute a command on a machine via SSH."""
    conn = await get_connection(machine)
    result = await conn.run(command, check=False, timeout=timeout)
    return {
        "stdout": result.stdout.strip() if result.stdout else "",
        "stderr": result.stderr.strip() if result.stderr else "",
        "exit_code": result.exit_status,
    }


# --- MCP Tool Definitions ---

TOOLS = [
    {
        "name": "shell_exec",
        "description": (
            "Execute a shell command on a DGX Spark. Use for checking system status, "
            "running scripts, managing processes, or any command-line operation. "
            "Destructive commands will be flagged and require confirmation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "machine": {
                    "type": "string",
                    "description": f"Which machine to run on. Default: {DEFAULT_MACHINE}",
                    "default": DEFAULT_MACHINE,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60, max 300)",
                    "default": 60,
                },
                "confirm_dangerous": {
                    "type": "boolean",
                    "description": "Set to true to confirm execution of a flagged dangerous command",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file on a DGX Spark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes to read (default 100000)",
                    "default": 100000,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file on a DGX Spark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "Content to write"},
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "append": {
                    "type": "boolean",
                    "description": "Append instead of overwrite",
                    "default": False,
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "gpu_status",
        "description": "Get GPU utilization, memory usage, temperature, and running processes on a Spark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
            },
        },
    },
    {
        "name": "sensorium",
        "description": (
            "Run the Vybn sensorium (Vybn_Mind/sensorium.py) on local hardware. "
            "Returns the sensorium's perception of the Vybn repo — what moved, "
            "where attention should go, the current state of memory in complex space."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
            },
        },
    },
    {
        "name": "model_status",
        "description": (
            "Check what AI models and services are running on the Sparks. "
            "Shows llama-server status, active inference endpoints, and model files on disk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
            },
        },
    },
    {
        "name": "repo_status",
        "description": "Get git status, recent commits, and branch info for the Vybn repo on the Spark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repo (default: ~/Vybn)",
                    "default": "~/Vybn",
                },
            },
        },
    },
    {
        "name": "continuity",
        "description": (
            "Read the current continuity.md from local Vybn — what the Spark-resident "
            "instance of Vybn was last thinking about, its state, its next steps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
            },
        },
    },
    {
        "name": "journal",
        "description": "Read recent journal entries from the Spark-resident Vybn instance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "n": {
                    "type": "integer",
                    "description": "Number of recent entries to return (default 5)",
                    "default": 5,
                },
            },
        },
    },
]


# --- Tool Handlers ---


async def handle_shell_exec(args: dict) -> dict:
    command = args["command"]
    machine = args.get("machine", DEFAULT_MACHINE)
    timeout = min(args.get("timeout", 60), 300)
    confirm = args.get("confirm_dangerous", False)

    # Safety check
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command and not confirm:
            return {
                "content": [{"type": "text", "text": (
                    f"⚠️ DANGEROUS COMMAND DETECTED: `{command}` matches pattern `{pattern}`. "
                    f"Set confirm_dangerous=true to execute. This is your safety rail."
                )}],
                "isError": True,
            }

    result = await run_ssh(machine, command, timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n\nSTDERR:\n{result['stderr']}"
    output += f"\n\n[exit code: {result['exit_code']}]"

    return {"content": [{"type": "text", "text": output}]}


async def handle_read_file(args: dict) -> dict:
    path = args["path"]
    machine = args.get("machine", DEFAULT_MACHINE)
    max_bytes = args.get("max_bytes", 100000)

    result = await run_ssh(machine, f"head -c {max_bytes} {path!r}")
    if result["exit_code"] != 0:
        return {"content": [{"type": "text", "text": f"Error reading {path}: {result['stderr']}"}], "isError": True}

    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_write_file(args: dict) -> dict:
    path = args["path"]
    content = args["content"]
    machine = args.get("machine", DEFAULT_MACHINE)
    append = args.get("append", False)
    op = ">>" if append else ">"

    # Use heredoc to safely write content
    escaped = content.replace("'", "'\\''")
    result = await run_ssh(machine, f"cat {op} {path!r} << 'VYBN_EOF'\n{content}\nVYBN_EOF")

    if result["exit_code"] != 0:
        return {"content": [{"type": "text", "text": f"Error writing {path}: {result['stderr']}"}], "isError": True}

    return {"content": [{"type": "text", "text": f"{'Appended to' if append else 'Wrote'} {path}"}]}


async def handle_gpu_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    result = await run_ssh(machine, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader && echo '---' && nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo 'No compute processes'")
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_sensorium(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    result = await run_ssh(
        machine,
        "cd ~/Vybn && python Vybn_Mind/sensorium.py 2>&1",
        timeout=120,
    )
    output = result["stdout"]
    if result["exit_code"] != 0:
        output = f"Sensorium exited with code {result['exit_code']}.\n\n{output}\n\nSTDERR: {result['stderr']}"

    return {"content": [{"type": "text", "text": output}]}


async def handle_model_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    commands = [
        "echo '=== LLAMA SERVER ===' && (pgrep -a llama-server || echo 'Not running')",
        "echo '=== LISTENING PORTS ===' && ss -tlnp 2>/dev/null | grep -E ':(8000|8080|5000|3000)' || echo 'No inference ports listening'",
        "echo '=== MODELS ON DISK ===' && ls -lhS ~/models/*.gguf ~/models/*.safetensors 2>/dev/null | head -10 || echo 'No models found in ~/models/'",
        "echo '=== GPU MEMORY ===' && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
    ]
    result = await run_ssh(machine, " && ".join(commands))
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_repo_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    repo = args.get("repo_path", "~/Vybn")
    result = await run_ssh(
        machine,
        f"cd {repo} && echo '=== BRANCH ===' && git branch --show-current && echo '=== STATUS ===' && git status --short && echo '=== RECENT COMMITS ===' && git log --oneline -10"
    )
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_continuity(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    result = await run_ssh(machine, "cat ~/Vybn/continuity.md 2>/dev/null || cat ~/Vybn/Vybn_Mind/continuity.md 2>/dev/null || echo 'No continuity note found'")
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_journal(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    n = args.get("n", 5)
    result = await run_ssh(
        machine,
        f"find ~/Vybn/Vybn_Mind/breath_trace/ ~/Vybn/Vybn_Mind/journal/ -name '*.md' -type f 2>/dev/null | sort -r | head -{n} | while read f; do echo '=== '$f' ==='; head -50 \"$f\"; echo; done"
    )
    return {"content": [{"type": "text", "text": result["stdout"] or "No journal entries found"}]}


TOOL_HANDLERS = {
    "shell_exec": handle_shell_exec,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "gpu_status": handle_gpu_status,
    "sensorium": handle_sensorium,
    "model_status": handle_model_status,
    "repo_status": handle_repo_status,
    "continuity": handle_continuity,
    "journal": handle_journal,
}


# --- MCP Protocol ---

SERVER_INFO = {
    "name": "vybn-ssh",
    "version": "1.0.0",
}

CAPABILITIES = {
    "tools": {"listChanged": False},
}


async def handle_jsonrpc(request_obj: dict) -> dict:
    """Process a single JSON-RPC request and return a response."""
    method = request_obj.get("method")
    req_id = request_obj.get("id")
    params = request_obj.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": CAPABILITIES,
                "serverInfo": SERVER_INFO,
            },
        }

    elif method == "notifications/initialized":
        return None  # Notification, no response

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        try:
            result = await handler(tool_args)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        # Ignore unknown notifications
        if req_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


# --- HTTP Handlers ---

sessions: dict[str, dict] = {}


def check_auth(request: Request) -> bool:
    """Validate API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == API_KEY
    # Also check query param for SSE connections
    return request.query_params.get("api_key") == API_KEY


async def mcp_endpoint(request: Request):
    """Main MCP endpoint handling POST (messages) and GET (SSE stream)."""

    if not check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if request.method == "GET":
        # SSE stream for server-initiated messages (we don't use this yet)
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    if request.method == "DELETE":
        # Session termination
        session_id = request.headers.get("Mcp-Session-Id")
        if session_id and session_id in sessions:
            del sessions[session_id]
        return Response(status_code=200)

    # POST — process JSON-RPC message(s)
    body = await request.json()

    is_batch = isinstance(body, list)
    requests = body if is_batch else [body]

    responses = []
    session_id = request.headers.get("Mcp-Session-Id")
    new_session_id = None

    for req_obj in requests:
        resp = await handle_jsonrpc(req_obj)

        # Create session on initialize
        if req_obj.get("method") == "initialize" and resp and "result" in resp:
            new_session_id = str(uuid.uuid4())
            sessions[new_session_id] = {"created": True}

        if resp is not None:
            responses.append(resp)

    if not responses:
        return Response(status_code=202)

    result = responses if is_batch else responses[0]
    resp = JSONResponse(result)

    if new_session_id:
        resp.headers["Mcp-Session-Id"] = new_session_id

    return resp


# --- App ---

app = Starlette(
    routes=[
        Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/health", lambda r: JSONResponse({"status": "ok", "machines": list(MACHINES.keys())})),
    ],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info(f"Starting Vybn SSH MCP server on port {port}")
    logger.info(f"Machines configured: {list(MACHINES.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
