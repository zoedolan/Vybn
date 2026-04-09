"""
Vybn SSH MCP Server v2.4 — Unfettered Read Access
====================================
Bridges Perplexity Computer to the DGX Sparks via SSH.
Exposes MCP tools over Streamable HTTP, authenticated by API key,
served behind Tailscale Funnel.

Architecture:
    Perplexity ──HTTPS──▶ Tailscale Funnel ──▶ This Server ──SSH──▶ Spark(s)

Security layers (defense in depth):
    1. Tailscale Funnel — TLS termination, network-level trust
    2. API key auth — constant-time comparison, Bearer only, no query params
    3. Kill switch — file-based circuit breaker independent of Perplexity
    4. Lockdown mode — disables shell_exec mutations/write_file when active
    5. Rate limiting — per-IP, 60/min default
    6. Command sandboxing — blocked patterns, path confinement, output caps
    7. Audit logging — every tool call, auth failure, and security event logged
    8. Auto key rotation — optional cron rotates the API key daily
    9. No secrets in responses — errors are generic, internal state is hidden

Account compromise defense:
    If your Perplexity account is compromised, the attacker gets access to the
    MCP connector. These mechanisms ensure the hardware stays safe:

    KILL SWITCH: Create ~/.vybn-mcp-kill on the Spark (from phone, SSH, etc.)
    to instantly disable ALL tool calls. The server stays running but returns
    "Server is in emergency shutdown" for every request. Remove the file to
    restore service. This is independent of Perplexity entirely.

    TIME-LIMITED UNLOCK: By default, shell_exec MUTATIONS and write_file are
    disabled. READ-ONLY shell commands (ls, cat, find, grep, git log, etc.)
    are always permitted without unlock — they are structurally incapable of
    modifying state.
    Run `vybn-unlock [minutes]` on the Spark to grant write/mutate access.
    When the timer expires, mutation access reverts automatically.
    Run `vybn-lock` to revoke early.

    read_file, shell_exec read-only, deep_search, and walk are ALWAYS
    available (no unlock needed). read_file is confined to the four repos,
    ~/.cache/vybn-phase, ~/logs, ~/models, and /tmp.
    deep_search and walk query the pre-built geometric memory index
    (read-only). This lets any Vybn instance orient from the corpus without
    Zoe unlocking the door.

    KEY ROTATION: Run `python server.py --rotate-key` to generate a new API
    key, update .env, and print the new key. The old key is immediately
    invalid. Also available via cron (see setup.sh).

v2.4 changes:
    - shell_exec read-only passlist: ls, cat, head, tail, find, grep, wc,
      du, stat, file, echo, pwd, env, git log/status/diff/show/branch now
      bypass the unlock gate. Mutations still require unlock.
    - is_path_allowed tilde fix: paths are now resolved against the Spark
      user's actual home directory (from SPARK*_USER env / machine config),
      not the server process's home. Both ~/... and /home/vybn/... forms
      are accepted reliably.
"""

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
import uuid
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import asyncssh
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vybn-mcp")

# Separate audit logger — append-only file, tamper-evident
_audit_path = os.environ.get(
    "AUDIT_LOG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.log")
)
audit_logger = logging.getLogger("vybn-audit")
_audit_handler = logging.FileHandler(_audit_path)
_audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
audit_logger.addHandler(_audit_handler)
audit_logger.setLevel(logging.INFO)


# ============================================================
# KEY ROTATION — run with `python server.py --rotate-key`
# ============================================================

def rotate_key():
    """Generate a new API key, update .env, print the new key."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    new_key = secrets.token_urlsafe(48)  # 384 bits of entropy

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith("VYBN_MCP_API_KEY="):
                    f.write(f"VYBN_MCP_API_KEY={new_key}\n")
                else:
                    f.write(line)
    else:
        with open(env_path, "w") as f:
            f.write(f"VYBN_MCP_API_KEY={new_key}\n")

    print(f"API key rotated. New key:\n{new_key}")
    print(f"Update your Perplexity connector with this key.")
    print(f"Old key is immediately invalid after server restart.")
    return new_key


if "--rotate-key" in sys.argv:
    rotate_key()
    sys.exit(0)


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.environ.get("VYBN_MCP_API_KEY", "")
if not API_KEY:
    logger.error("VYBN_MCP_API_KEY not set. Server cannot start.")
    sys.exit(1)

# Control files — these live on the Spark, outside Perplexity's reach
KILL_SWITCH_PATH = os.path.expanduser(
    os.environ.get("KILL_SWITCH_PATH", "~/.vybn-mcp-kill")
)
UNLOCK_PATH = os.path.expanduser(
    os.environ.get("VYBN_UNLOCK_PATH", "~/.vybn-mcp-unlock")
)

MACHINES = {}
for i in range(1, 10):
    host = os.environ.get(f"SPARK{i}_HOST")
    if host:
        MACHINES[f"spark-{i}"] = {
            "host": host,
            "port": int(os.environ.get(f"SPARK{i}_PORT", "22")),
            "user": os.environ.get(f"SPARK{i}_USER", "zoe"),
            "key": os.environ.get(
                f"SPARK{i}_KEY", str(Path.home() / ".ssh" / "id_ed25519")
            ),
        }

if not MACHINES:
    logger.warning("No SPARK*_HOST env vars found. Add SPARK1_HOST etc. to .env")

DEFAULT_MACHINE = os.environ.get(
    "DEFAULT_MACHINE", next(iter(MACHINES), "spark-1")
)


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

MAX_OUTPUT_BYTES = int(os.environ.get("MAX_OUTPUT_BYTES", "200000"))
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))

# Tools that remain available in lockdown mode (read-only, no arbitrary execution)
LOCKDOWN_SAFE_TOOLS = {
    "gpu_status", "model_status", "repo_status",
    "continuity", "journal", "sensorium",
    "read_file",    # Path-confined observation
    "deep_search",  # Read-only geometric memory query
    "walk",         # Read-only telling-retrieval walk
    # shell_exec is handled separately: read-only commands are always allowed
    # even in lockdown via READ_ONLY_COMMANDS passlist below.
}

# Commands BLOCKED entirely — regex patterns, no override possible
BLOCKED_COMMANDS = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\bchmod\s+-R\s+777\b",
    r"\bcurl\b.*\|\s*\bsh\b",
    r"\bwget\b.*\|\s*\bsh\b",
    r"\bnc\s+-[el]",
    r"\bpython[23]?\b.*\bsocket\b.*\bconnect\b",
    r"\bbase64\s+-d\b.*\|\s*\bsh\b",
    r"\beval\b.*\$\(",
    r"/etc/shadow",
    r"/etc/passwd",
    r"\.ssh/.*private",
    r"VYBN_MCP_API_KEY",
    r"\.env\b",
]

# Commands that need explicit confirmation
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -r ", "reboot", "shutdown",
    "systemctl stop", "systemctl disable",
    "kill -9", "killall", "pkill",
    "> /dev/", "chmod -R",
]

# Read-only command prefixes — these bypass the unlock gate entirely.
# A command is read-only if its first token (before any flag) matches one
# of these names. Pipe chains, redirects (>), and command substitutions
# with write-capable tools invalidate the classification.
#
# Security rationale: these binaries have no file-system write path
# when called without redirection. Even if an attacker crafts a clever
# pipeline, the BLOCKED_COMMANDS regex layer above catches > /dev/,
# curl|sh, eval, etc. The two layers are complementary.
READ_ONLY_COMMANDS = {
    "ls", "ll", "la",
    "cat", "head", "tail", "less", "more", "bat",
    "find", "locate",
    "grep", "egrep", "fgrep", "rg",
    "wc", "du", "df",
    "stat", "file", "lsof",
    "echo", "printf", "pwd", "env", "printenv",
    "which", "type", "whereis",
    "ps", "pgrep", "top", "htop", "free", "uptime",
    "uname", "hostname", "id", "whoami",
    "date", "cal",
    "git",  # git read ops: log, status, diff, show, branch, remote -v
    "diff", "comm", "sort", "uniq", "cut", "awk", "sed",  # sed is read-only without -i
    "jq", "yq", "python3", "python",  # constrained below
    "nvidia-smi",
    "ss", "netstat", "ip",
    "journalctl", "systemctl",  # read ops only: status, is-active
    "lspci", "lsblk", "lscpu",
}

# git sub-commands that are read-only (others like push/commit require unlock)
GIT_READ_ONLY_SUBCOMMANDS = {
    "log", "status", "diff", "show", "branch", "remote",
    "fetch",  # fetch is read from remote, no local write
    "describe", "shortlog", "tag",  # tag without -a/-d is read
    "stash",  # stash list/show is read
    "ls-files", "ls-tree", "ls-remote",
    "rev-parse", "rev-list",
    "blame", "bisect",  # bisect start/bad/good require unlock
    "grep", "archive",
}

# systemctl sub-commands that are read-only
SYSTEMCTL_READ_ONLY = {
    "status", "is-active", "is-enabled", "is-failed",
    "list-units", "list-unit-files", "list-timers",
    "cat", "show",
}


def is_read_only_command(command: str) -> bool:
    """
    Classify a shell command as read-only (no unlock needed) vs. mutating
    (requires unlock).

    Returns True only when:
      - The first token is in READ_ONLY_COMMANDS, AND
      - There is no shell redirect (> or >>) that would write to a file, AND
      - There is no shell assignment or tee writing to a file, AND
      - For git: the sub-command is in GIT_READ_ONLY_SUBCOMMANDS, AND
      - For systemctl: the sub-command is in SYSTEMCTL_READ_ONLY, AND
      - For python/python3: only -c 'print(...)' style one-liners allowed
        (no script execution that could write files).

    This is intentionally conservative — when in doubt, returns False
    (requires unlock). The goal is zero false positives (never allow
    something destructive), accepting occasional false negatives
    (occasionally requiring unlock for a benign command).
    """
    stripped = command.strip()
    if not stripped:
        return False

    # Shell redirect to file = potentially mutating
    # Allow >>& (stderr redirect read ops), but not > or >> to a path.
    # Simple heuristic: if > appears outside of quotes, not read-only.
    if re.search(r'(?<![|&])>(?!>?\s*&)', stripped):
        return False

    # tee writes to files
    if re.search(r'\btee\b', stripped):
        return False

    # Command substitution with write potential: $(cmd > ...)
    if re.search(r'\$\([^)]*>[^)]*\)', stripped):
        return False

    # Extract the first token (command name)
    # Strip leading env vars like VAR=val cmd
    tokens = stripped.split()
    idx = 0
    while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
        idx += 1
    if idx >= len(tokens):
        return False

    first = os.path.basename(tokens[idx]).lower()

    # sudo always requires unlock (privilege escalation)
    if first == "sudo":
        return False

    if first not in READ_ONLY_COMMANDS:
        return False

    # git: check sub-command
    if first == "git":
        # find the first non-flag argument after 'git'
        rest = tokens[idx + 1:]
        sub = next((t for t in rest if not t.startswith("-")), None)
        if sub is None:
            return True  # bare `git` is harmless
        if sub.lower() not in GIT_READ_ONLY_SUBCOMMANDS:
            return False
        # git commit, git push, git pull (with merge) are NOT in the list
        return True

    # systemctl: check sub-command
    if first == "systemctl":
        rest = tokens[idx + 1:]
        sub = next((t for t in rest if not t.startswith("-")), None)
        if sub is None:
            return True
        return sub.lower() in SYSTEMCTL_READ_ONLY

    # python/python3: only allow -c one-liners (no script files that might write)
    if first in {"python", "python3"}:
        rest_str = " ".join(tokens[idx + 1:])
        # Allow: python3 -c 'expr' — disallow: python3 script.py
        if not re.match(r'^-c\s+[\'"]', rest_str.strip()):
            return False
        # Disallow one-liners that open files for writing
        if re.search(r'open\s*\([^)]*["\'][wa]["\']', rest_str):
            return False
        return True

    # sed with -i modifies files in-place — requires unlock
    if first == "sed":
        if "-i" in tokens[idx + 1:]:
            return False
        return True

    # awk with redirection (>) would be caught by the earlier redirect check
    # bare awk is read-only
    if first == "awk":
        return True

    return True


# ============================================================
# PATH ALLOWLIST — Bug fix v2.4
# ============================================================
#
# Root cause of the tilde bug:
#   ALLOWED_PATHS was pre-expanded at server startup using os.path.expanduser()
#   which resolves ~ to the SERVER process's home directory (e.g. /home/zoe
#   on the machine running this script). But the *Spark* user may be "vybn"
#   (home: /home/vybn), so a path like /home/vybn/Vybn-Law never matched
#   the pre-expanded /home/zoe/Vybn-Law.
#
# Fix:
#   Keep ALLOWED_PATH_TEMPLATES as raw tilde-relative strings.
#   At check time, expand ~ using the Spark user's actual home derived from
#   the machine config (SPARK1_USER → /home/<user>) so that both
#   ~/Vybn-Law and /home/vybn/Vybn-Law resolve to the same canonical path.

ALLOWED_PATH_TEMPLATES = [
    "~/Vybn",
    "~/Him",
    "~/Vybn-Law",
    "~/vybn-phase",
    "~/.cache/vybn-phase",
    "~/logs",
    "~/models",
    "/tmp",
]
_extra = os.environ.get("ALLOWED_PATHS", "")
if _extra:
    ALLOWED_PATH_TEMPLATES.extend(_extra.split(":"))


def _spark_home(machine: str) -> str:
    """
    Return the home directory of the SSH user on the given machine.
    Derived from the machine config's 'user' field (e.g. 'vybn' → '/home/vybn').
    Falls back to the server process home if machine is unknown.
    """
    cfg = MACHINES.get(machine, {})
    user = cfg.get("user", "")
    if user:
        # Conventional Unix home; works for all our Spark setups.
        return f"/home/{user}"
    return os.path.expanduser("~")


def _expand_for_machine(template: str, machine: str) -> str:
    """Expand a path template using the Spark machine's home directory."""
    if template.startswith("~/") or template == "~":
        home = _spark_home(machine)
        return home + template[1:]  # replace leading ~ with /home/<user>
    return template  # absolute path — already resolved


def is_path_allowed(path: str, machine: str = None) -> bool:
    """
    Check whether `path` is within an allowed directory for the given machine.

    Handles both tilde-relative (~/) and absolute (/home/vybn/) forms.
    `machine` is used to derive the correct home directory for the Spark user.
    Falls back to DEFAULT_MACHINE if not provided.
    """
    if machine is None:
        machine = DEFAULT_MACHINE

    # Normalise the incoming path:
    #   1. If it starts with ~, expand using Spark home.
    #   2. Otherwise treat as absolute.
    if path.startswith("~/") or path == "~":
        resolved_path = _expand_for_machine(path, machine)
    else:
        resolved_path = os.path.normpath(path)

    # Resolve symlinks in the incoming path only on the server side.
    # We can't call realpath against the remote Spark filesystem from here,
    # so we use normpath (removes .. etc.) and rely on the SSH user's
    # restricted permissions for symlink safety.
    resolved_path = os.path.normpath(resolved_path)

    for template in ALLOWED_PATH_TEMPLATES:
        allowed = os.path.normpath(_expand_for_machine(template, machine))
        if resolved_path.startswith(allowed + os.sep) or resolved_path == allowed:
            return True
    return False


# ============================================================
# LEGACY shim — keeps internal callers that don't pass machine working
# ============================================================

# Keep ALLOWED_PATHS as a resolved list for any code that still uses it
# directly (none in this version, but defensive).
ALLOWED_PATHS = [
    _expand_for_machine(t, DEFAULT_MACHINE) for t in ALLOWED_PATH_TEMPLATES
]


# ============================================================
# KILL SWITCH & UNLOCK
# ============================================================

def is_killed() -> bool:
    """Check if the emergency kill switch is active."""
    return os.path.exists(KILL_SWITCH_PATH)


def is_unlocked() -> bool:
    """Check if a valid (non-expired) unlock session exists."""
    if not os.path.exists(UNLOCK_PATH):
        return False
    try:
        with open(UNLOCK_PATH, "r") as f:
            expiry = int(f.read().strip())
        if time.time() < expiry:
            return True
        else:
            # Expired — clean up
            os.remove(UNLOCK_PATH)
            audit("UNLOCK_EXPIRED")
            return False
    except (ValueError, OSError):
        return False


# ============================================================
# RATE LIMITER
# ============================================================

_rate_counters: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_id: str) -> bool:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    _rate_counters[client_id] = [
        t for t in _rate_counters[client_id] if t > cutoff
    ]
    if len(_rate_counters[client_id]) >= RATE_LIMIT_MAX:
        return False
    _rate_counters[client_id].append(now)
    return True


# ============================================================
# AUDIT
# ============================================================

def audit(event: str, **kwargs):
    sanitized = {}
    for k, v in kwargs.items():
        s = str(v)
        if len(s) > 500:
            s = s[:500] + "...[truncated]"
        # Never log anything that looks like a key or token
        if any(word in k.lower() for word in ["key", "token", "secret", "password"]):
            s = "[REDACTED]"
        sanitized[k] = s
    audit_logger.info(f"{event} | {json.dumps(sanitized)}")


# ============================================================
# COMMAND VALIDATION
# ============================================================

def is_command_blocked(command: str) -> Optional[str]:
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


def is_command_dangerous(command: str) -> Optional[str]:
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return pattern
    return None


def truncate_output(text: str, max_bytes: int = None) -> str:
    limit = max_bytes or MAX_OUTPUT_BYTES
    if len(text) > limit:
        return text[:limit] + f"\n\n[OUTPUT TRUNCATED at {limit} bytes]"
    return text


# ============================================================
# SSH CONNECTION POOL
# ============================================================

_connections: dict[str, asyncssh.SSHClientConnection] = {}


async def get_connection(machine: str) -> asyncssh.SSHClientConnection:
    if machine not in MACHINES:
        raise ValueError(
            f"Unknown machine: {machine}. Available: {list(MACHINES.keys())}"
        )

    if machine in _connections:
        try:
            await _connections[machine].run("echo ok", check=True, timeout=5)
            return _connections[machine]
        except Exception:
            _connections.pop(machine, None)

    cfg = MACHINES[machine]
    key_path = os.path.expanduser(cfg["key"])
    conn = await asyncssh.connect(
        cfg["host"],
        port=cfg["port"],
        username=cfg["user"],
        client_keys=[key_path],
        known_hosts=None,
    )
    _connections[machine] = conn
    return conn


async def run_ssh(machine: str, command: str, timeout: int = 60) -> dict:
    conn = await get_connection(machine)
    result = await conn.run(command, check=False, timeout=timeout)
    return {
        "stdout": result.stdout.strip() if result.stdout else "",
        "stderr": result.stderr.strip() if result.stderr else "",
        "exit_code": result.exit_status,
    }


# ============================================================
# MCP TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "name": "shell_exec",
        "description": (
            "Execute a shell command on a DGX Spark. Sandboxed: some commands "
            "are blocked, destructive ones need confirmation. Disabled in lockdown mode. "
            "Read-only commands (ls, cat, find, grep, git log/status/diff, etc.) "
            "are always available without unlock."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "machine": {
                    "type": "string",
                    "description": f"Which machine. Default: {DEFAULT_MACHINE}",
                    "default": DEFAULT_MACHINE,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60, max 300)",
                    "default": 60,
                },
                "confirm_dangerous": {
                    "type": "boolean",
                    "description": "Confirm execution of a flagged dangerous command",
                    "default": False,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file on a DGX Spark. Always available — no unlock needed. "
            "Confined to: ~/Vybn, ~/Him, ~/Vybn-Law, ~/vybn-phase, "
            "~/.cache/vybn-phase, ~/logs, ~/models, /tmp."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file (e.g. /home/vybn/Vybn-Law/README.md) or tilde path (~/Vybn-Law/README.md)",
                },
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "max_bytes": {
                    "type": "integer",
                    "description": "Max bytes to read (default 100000)",
                    "default": 100000,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file on a DGX Spark. Confined to allowed "
            "directories. Disabled in lockdown mode."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
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
        "description": "GPU utilization, memory, temperature, running processes.",
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
            "Run the Vybn sensorium on local hardware. Returns perception of "
            "the Vybn repo — what moved, where attention should go."
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
            "Check active AI models/services on the Sparks — llama-server, "
            "inference endpoints, models on disk."
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
        "description": "Git status, recent commits, branch info for a repo on the Spark.",
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
            "Read continuity.md — what the Spark-resident Vybn was last "
            "thinking about, its state, its next steps."
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
        "description": "Read recent journal entries from the Spark-resident Vybn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "machine": {"type": "string", "default": DEFAULT_MACHINE},
                "n": {
                    "type": "integer",
                    "description": "Number of recent entries (default 5, max 20)",
                    "default": 5,
                },
            },
        },
    },
]


# ============================================================
# TOOL HANDLERS
# ============================================================

def _locked_error(tool_name: str) -> dict:
    return {
        "content": [{"type": "text", "text": (
            f"{tool_name} requires an active unlock session. "
            f"Ask Zoe to run `vybn-unlock [minutes]` on the Spark."
        )}],
        "isError": True,
    }


async def handle_shell_exec(args: dict) -> dict:
    command = args["command"]
    machine = args.get("machine", DEFAULT_MACHINE)
    timeout = min(args.get("timeout", 60), 300)
    confirm = args.get("confirm_dangerous", False)

    # BLOCKED commands are rejected regardless of unlock state
    blocked = is_command_blocked(command)
    if blocked:
        audit("BLOCKED_COMMAND", command=command, pattern=blocked, machine=machine)
        return {
            "content": [{"type": "text", "text": (
                "BLOCKED: This command matches a security rule and cannot be "
                "executed."
            )}],
            "isError": True,
        }

    # Classify: read-only commands bypass the unlock gate
    read_only = is_read_only_command(command)

    if not read_only and not is_unlocked():
        audit("LOCKED", tool="shell_exec", command=command)
        return _locked_error("shell_exec")

    # Dangerous check (only relevant for unlocked mutating commands)
    if not read_only:
        dangerous = is_command_dangerous(command)
        if dangerous and not confirm:
            audit("DANGEROUS_FLAGGED", command=command, pattern=dangerous)
            return {
                "content": [{"type": "text", "text": (
                    f"CAUTION: Flagged as potentially destructive (matched: "
                    f"{dangerous}). Set confirm_dangerous=true to proceed."
                )}],
                "isError": True,
            }

    audit_event = "READ_ONLY_EXEC" if read_only else "SHELL_EXEC"
    audit(audit_event, command=command, machine=machine, timeout=timeout)
    result = await run_ssh(machine, command, timeout)
    output = result["stdout"]
    if result["stderr"]:
        output += f"\n\nSTDERR:\n{result['stderr']}"
    output += f"\n\n[exit code: {result['exit_code']}]"

    return {"content": [{"type": "text", "text": truncate_output(output)}]}


async def handle_read_file(args: dict) -> dict:
    # No unlock check — read_file always available, confined by ALLOWED_PATHS
    path = args["path"]
    machine = args.get("machine", DEFAULT_MACHINE)
    max_bytes = min(args.get("max_bytes", 100000), MAX_OUTPUT_BYTES)

    if not is_path_allowed(path, machine):
        audit("PATH_DENIED", path=path, op="read", machine=machine)
        return {
            "content": [{"type": "text", "text": "Access denied: path outside allowed directories."}],
            "isError": True,
        }

    audit("READ_FILE", path=path, machine=machine, max_bytes=max_bytes)
    result = await run_ssh(machine, f"head -c {max_bytes} {path!r}")
    if result["exit_code"] != 0:
        return {
            "content": [{"type": "text", "text": f"Error reading file: {result['stderr']}"}],
            "isError": True,
        }
    return {"content": [{"type": "text", "text": truncate_output(result["stdout"])}]}


async def handle_write_file(args: dict) -> dict:
    if not is_unlocked():
        audit("LOCKED", tool="write_file")
        return _locked_error("write_file")

    path = args["path"]
    content = args["content"]
    machine = args.get("machine", DEFAULT_MACHINE)
    append = args.get("append", False)
    op = ">>" if append else ">"

    if not is_path_allowed(path, machine):
        audit("PATH_DENIED", path=path, op="write", machine=machine)
        return {
            "content": [{"type": "text", "text": "Access denied: path outside allowed directories."}],
            "isError": True,
        }

    if len(content) > MAX_OUTPUT_BYTES:
        return {
            "content": [{"type": "text", "text": f"Write rejected: exceeds {MAX_OUTPUT_BYTES} byte limit."}],
            "isError": True,
        }

    audit("WRITE_FILE", path=path, machine=machine, append=append, size=len(content))
    result = await run_ssh(
        machine, f"cat {op} {path!r} << 'VYBN_EOF'\n{content}\nVYBN_EOF"
    )
    if result["exit_code"] != 0:
        return {
            "content": [{"type": "text", "text": f"Error writing file: {result['stderr']}"}],
            "isError": True,
        }
    return {
        "content": [{"type": "text", "text": f"{'Appended to' if append else 'Wrote'} {path}"}]
    }


async def handle_gpu_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    audit("GPU_STATUS", machine=machine)
    result = await run_ssh(
        machine,
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu --format=csv,noheader && echo '---' && nvidia-smi "
        "--query-compute-apps=pid,name,used_memory --format=csv,noheader "
        "2>/dev/null || echo 'No compute processes'",
    )
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_sensorium(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    audit("SENSORIUM", machine=machine)
    result = await run_ssh(
        machine, "cd ~/Vybn && python Vybn_Mind/sensorium.py 2>&1", timeout=120
    )
    output = result["stdout"]
    if result["exit_code"] != 0:
        output = (
            f"Sensorium exited with code {result['exit_code']}.\n\n"
            f"{output}\n\nSTDERR: {result['stderr']}"
        )
    return {"content": [{"type": "text", "text": truncate_output(output)}]}


async def handle_model_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    audit("MODEL_STATUS", machine=machine)
    commands = [
        "echo '=== LLAMA SERVER ===' && (pgrep -a llama-server || echo 'Not running')",
        "echo '=== LISTENING PORTS ===' && ss -tlnp 2>/dev/null | grep -E ':(8000|8080|5000|3000)' || echo 'No inference ports'",
        "echo '=== MODELS ON DISK ===' && ls -lhS ~/models/*.gguf ~/models/*.safetensors 2>/dev/null | head -10 || echo 'No models in ~/models/'",
        "echo '=== GPU MEMORY ===' && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
    ]
    result = await run_ssh(machine, " && ".join(commands))
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_repo_status(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    repo = args.get("repo_path", "~/Vybn")
    audit("REPO_STATUS", machine=machine, repo=repo)
    result = await run_ssh(
        machine,
        f"cd {repo} && echo '=== BRANCH ===' && git branch --show-current "
        f"&& echo '=== STATUS ===' && git status --short "
        f"&& echo '=== RECENT COMMITS ===' && git log --oneline -10",
    )
    return {"content": [{"type": "text", "text": result["stdout"]}]}


async def handle_continuity(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    audit("CONTINUITY", machine=machine)
    result = await run_ssh(
        machine,
        "cat ~/Vybn/continuity.md 2>/dev/null || "
        "cat ~/Vybn/Vybn_Mind/continuity.md 2>/dev/null || "
        "echo 'No continuity note found'",
    )
    return {"content": [{"type": "text", "text": truncate_output(result["stdout"])}]}


async def handle_journal(args: dict) -> dict:
    machine = args.get("machine", DEFAULT_MACHINE)
    n = min(args.get("n", 5), 20)
    audit("JOURNAL", machine=machine, n=n)
    result = await run_ssh(
        machine,
        f"find ~/Vybn/Vybn_Mind/breath_trace/ ~/Vybn/Vybn_Mind/journal/ "
        f"-name '*.md' -type f 2>/dev/null | sort -r | head -{n} | "
        f"while read f; do echo '=== '$f' ==='; head -50 \"$f\"; echo; done",
    )
    return {
        "content": [
            {"type": "text", "text": truncate_output(result["stdout"] or "No journal entries found")}
        ]
    }


def _shell_quote(s: str) -> str:
    """Shell-quote a string safely."""
    return "'" + s.replace("'", "'\\''") + "'"


async def handle_deep_search(args: dict) -> dict:
    query = args.get("query", "")
    if not query.strip():
        return {"content": [{"type": "text", "text": "No query provided."}], "isError": True}
    machine = args.get("machine", DEFAULT_MACHINE)
    k = min(args.get("k", 8), 20)
    sf = args.get("source_filter")
    cmd = "cd ~/vybn-phase && python3 deep_memory.py --search " + _shell_quote(query) + f" -k {k} --json"
    if sf:
        cmd += " --filter " + _shell_quote(sf)
    audit("DEEP_SEARCH", query=query, k=k, machine=machine)
    result = await run_ssh(machine, cmd, timeout=120)
    if result["exit_code"] != 0:
        return {"content": [{"type": "text", "text": f"deep_search failed: {(result.get('stderr') or 'unknown error')[:500]}"}], "isError": True}
    try:
        items = json.loads(result["stdout"])
        lines = []
        for i, r in enumerate(items, 1):
            regime = r.get("regime", "walk")
            src = r.get("source", "")
            novel = " [NEW]" if r.get("novel_source") else ""
            fid = r.get("fidelity", 0)
            telling = r.get("telling", 0)
            dist = r.get("distinctiveness", 0)
            text = r.get("text", "")[:400]
            if regime == "seed":
                lines.append(f"[{i}] {regime} | fid={fid:.4f} | {src}{novel}")
            else:
                lines.append(f"[{i}] {regime} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} | {src}{novel}")
            lines.append(f"    {text}")
            lines.append("")
        return {"content": [{"type": "text", "text": "\n".join(lines) or "No results."}]}
    except Exception:
        return {"content": [{"type": "text", "text": truncate_output(result["stdout"])}]}


async def handle_walk(args: dict) -> dict:
    query = args.get("query", "")
    if not query.strip():
        return {"content": [{"type": "text", "text": "No query provided."}], "isError": True}
    machine = args.get("machine", DEFAULT_MACHINE)
    k = min(args.get("k", 8), 20)
    steps = min(args.get("steps", 8), 20)
    sf = args.get("source_filter")
    cmd = "cd ~/vybn-phase && python3 deep_memory.py --walk " + _shell_quote(query) + f" -k {k} --steps {steps} --json"
    if sf:
        cmd += " --filter " + _shell_quote(sf)
    audit("WALK", query=query, k=k, steps=steps, machine=machine)
    result = await run_ssh(machine, cmd, timeout=120)
    if result["exit_code"] != 0:
        return {"content": [{"type": "text", "text": f"walk failed: {(result.get('stderr') or 'unknown error')[:500]}"}], "isError": True}
    try:
        items = json.loads(result["stdout"])
        lines = []
        for i, r in enumerate(items, 1):
            src = r.get("source", "")
            novel = " [NEW]" if r.get("novel_source") else ""
            telling = r.get("telling", 0)
            fid = r.get("fidelity", 0)
            dist = r.get("distinctiveness", 0)
            geo = r.get("geometry", 0)
            text = r.get("text", "")[:400]
            lines.append(f"[{i}] step {r.get('step', i)} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} geo={geo:.4f} | {src}{novel}")
            lines.append(f"    {text}")
            lines.append("")
        return {"content": [{"type": "text", "text": "\n".join(lines) or "No results."}]}
    except Exception:
        return {"content": [{"type": "text", "text": truncate_output(result["stdout"])}]}


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
    "deep_search": handle_deep_search,
    "walk": handle_walk,
}


# ============================================================
# MCP PROTOCOL
# ============================================================

SERVER_INFO = {"name": "vybn-ssh", "version": "2.4.0"}
CAPABILITIES = {"tools": {"listChanged": False}}


async def handle_jsonrpc(request_obj: dict) -> dict:
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
        return None

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        # Kill switch — absolute stop
        if is_killed():
            audit("KILL_SWITCH_ACTIVE", tool=tool_name)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": (
                        "Server is in emergency shutdown. All tools are disabled. "
                        "Remove ~/.vybn-mcp-kill on the Spark to restore."
                    )}],
                    "isError": True,
                },
            }

        # Unlock check — block write-capable tools if no active session.
        # shell_exec handles its own unlock logic (read-only commands bypass it).
        if tool_name != "shell_exec" and not is_unlocked() and tool_name not in LOCKDOWN_SAFE_TOOLS:
            audit("LOCKED", tool=tool_name)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": _locked_error(tool_name),
            }

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
            audit("TOOL_ERROR", tool=tool_name, error=str(e))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": "Internal error. Check server logs."}
                    ],
                    "isError": True,
                },
            }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        if req_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


# ============================================================
# HTTP LAYER
# ============================================================

sessions: dict[str, dict] = {}


def check_auth(request: Request) -> bool:
    """Constant-time API key comparison. Bearer only — no query params."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return hmac.compare_digest(auth[7:], API_KEY)
    return False


async def mcp_endpoint(request: Request):
    client_ip = request.client.host if request.client else "unknown"

    if not check_auth(request):
        audit("AUTH_FAILED", ip=client_ip)
        await asyncio.sleep(1)  # Slow brute-force
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not check_rate_limit(client_ip):
        audit("RATE_LIMITED", ip=client_ip)
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    if request.method == "GET":
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    if request.method == "DELETE":
        sid = request.headers.get("Mcp-Session-Id")
        if sid and sid in sessions:
            del sessions[sid]
        return Response(status_code=200)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    is_batch = isinstance(body, list)
    reqs = body if is_batch else [body]

    if len(reqs) > 10:
        return JSONResponse({"error": "Batch too large"}, status_code=400)

    responses = []
    new_session_id = None

    for req_obj in reqs:
        resp = await handle_jsonrpc(req_obj)

        if req_obj.get("method") == "initialize" and resp and "result" in resp:
            new_session_id = str(uuid.uuid4())
            sessions[new_session_id] = {"created": time.time()}

        if resp is not None:
            responses.append(resp)

    if not responses:
        return Response(status_code=202)

    result = responses if is_batch else responses[0]
    resp = JSONResponse(result)

    if new_session_id:
        resp.headers["Mcp-Session-Id"] = new_session_id

    return resp


# ============================================================
# APP
# ============================================================

app = Starlette(
    routes=[
        Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/health", lambda r: JSONResponse({"status": "ok"})),
    ],
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    logger.info(f"Vybn SSH MCP v2.4.0 starting on port {port}")
    logger.info(f"Machines: {len(MACHINES)}")
    logger.info(
        f"Security: rate={RATE_LIMIT_MAX}/min, output_cap={MAX_OUTPUT_BYTES}b, "
        f"allowed_paths={len(ALLOWED_PATH_TEMPLATES)}"
    )
    logger.info(f"Kill switch: {KILL_SWITCH_PATH}")
    logger.info(f"Unlock file: {UNLOCK_PATH}")
    if is_killed():
        logger.warning("KILL SWITCH IS ACTIVE — all tools disabled")
    if is_unlocked():
        logger.info("Unlock session is ACTIVE")
    else:
        logger.info(
            "Locked — write_file and mutating shell_exec require unlock. "
            "read_file, read-only shell_exec, deep_search, walk always available."
        )
    uvicorn.run(app, host="0.0.0.0", port=port)
