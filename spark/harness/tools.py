"""Neutral tool specs + absorb_gate + BashSession.

Tool specs are provider-neutral. Each Provider renders them into its
native shape at call time (Anthropic tool blocks vs OpenAI functions).

absorb_gate is lifted verbatim from vybn_spark_agent.py so vybn_chat_api.py
and other agents can share the same invariant.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .constants import (
    ABSORB_EXCLUDE_SUBSTR,
    ABSORB_EXCLUDE_SUFFIX,
    ABSORB_LOG,
    DANGEROUS_PATTERNS,
    DEFAULT_TIMEOUT,
    MAX_BASH_TIMEOUT,
    TRACKED_REPOS,
)


# ---------------------------------------------------------------------------
# absorb_gate (ported unchanged from vybn_spark_agent.py)
# ---------------------------------------------------------------------------

_REDIRECT_RE = re.compile(r"(?<![<>])>>?\s*([^\s<>|&;'\"]+)")
_TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s<>|&;'\"]+)")
_TOUCH_RE = re.compile(r"\btouch\s+([^\s<>|&;'\"]+)")


def _extract_file_targets(command: str) -> list[str]:
    out: list[str] = []
    for rx in (_REDIRECT_RE, _TEE_RE, _TOUCH_RE):
        for m in rx.finditer(command):
            t = m.group(1).strip("'\"")
            if not t or t.startswith("/dev/") or t.startswith("/proc/"):
                continue
            if not os.path.isabs(t):
                continue
            out.append(os.path.normpath(t))
    return out[:10]


def absorb_gate(command: str) -> str | None:
    """Return refusal text if command would create a new tracked file
    without an inline VYBN_ABSORB_REASON. Otherwise None."""
    if "VYBN_ABSORB_REASON=" in command:
        return None
    for tgt in _extract_file_targets(command):
        if not any(tgt == r or tgt.startswith(r + "/") for r in TRACKED_REPOS):
            continue
        if any(s in tgt for s in ABSORB_EXCLUDE_SUBSTR):
            continue
        if tgt.endswith(ABSORB_EXCLUDE_SUFFIX):
            continue
        if os.path.exists(tgt):
            continue
        return (
            "[absorb_gate] refused. This command would create a new tracked "
            "file:\n"
            f"    {tgt}\n\n"
            "New-file creation is the agent's default failure mode. Before "
            "proceeding, in your reply to Zoe, name the existing file you "
            "considered folding this into and why it did not fit. Then "
            "re-issue the command with an inline reason, e.g.:\n\n"
            "    VYBN_ABSORB_REASON=\"does not fold into X because ...\" "
            "<command>\n\n"
            "Fold, do not pile. If you are certain the new file is right, "
            "the reason is the record of that certainty."
        )
    return None


def log_absorb(command: str) -> None:
    try:
        with open(ABSORB_LOG, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tabsorb\t{command[:400]}\n"
            )
    except Exception:
        pass


def validate_command(command: str) -> tuple[bool, str | None]:
    lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in lower:
            return False, f"Blocked: '{pattern}'"
    return True, None


# ---------------------------------------------------------------------------
# Parallel-safe command classifier.
#
# The persistent BashSession exists because cd/export/source/= accumulate
# state. Most tool calls in a debug loop are reads (cat/ls/grep/head/tail/
# sed/wc/find/stat/git log|status|diff), which run safely in fresh
# subprocesses. When a single assistant turn emits >=2 such commands the
# agent dispatches them in parallel via execute_readonly; mixed turns
# fall back to the persistent shell.
# ---------------------------------------------------------------------------

_READONLY_HEADS = (
    "cat", "ls", "ll", "grep", "rg", "fgrep", "egrep",
    "head", "tail", "sed", "awk", "wc", "find", "locate",
    "stat", "file", "tree",
    "python3 -c", "python -c", "python3 -m py_compile", "python -m py_compile",
    "git log", "git status", "git diff", "git show", "git rev-parse",
    "git blame", "git branch", "git remote",
    "echo", "printf", "date", "whoami", "pwd", "env",
    "curl -s", "curl -sS", "curl -fs", "curl -fsS",
    "jq", "yq", "md5sum", "sha1sum", "sha256sum",
    "diff", "cmp",
)

# Any of these tokens anywhere in the command disqualifies parallel dispatch.
_NON_READONLY_TOKENS = (
    "cd ", "pushd ", "popd ",
    "export ", "unset ", "source ", ". /", ". ~",
    ">>", ">", "<", " tee ",
    "mv ", "cp ", "rm ", "rmdir ", "mkdir ", "touch ",
    "chmod ", "chown ", "ln ",
    "kill ", "pkill ", "killall ",
    "pip install", "pip uninstall", "npm install", "apt ",
    "systemctl ", "service ", "docker ",
    "git commit", "git push", "git pull", "git merge", "git rebase",
    "git add", "git reset", "git checkout", "git stash", "git clone",
    "VYBN_ABSORB_REASON=",
)


def is_parallel_safe(command: str) -> bool:
    """True if `command` can run in a fresh subprocess without mutating
    shell state or the filesystem. Conservative by design — a false
    negative just routes through the persistent shell."""
    if not command or not command.strip():
        return False
    c = command.strip()
    for tok in _NON_READONLY_TOKENS:
        if tok in c:
            return False
    # Split by pipeline / logical separators; every segment must START
    # with a known read verb. We do not try to parse quoting precisely;
    # if a segment has a quoted separator it will just be treated as one
    # segment and still needs to start with a readonly head.
    work = c
    for sep in ("&&", "||", ";", "|"):
        work = work.replace(sep, "\x01")
    segments = [seg.strip() for seg in work.split("\x01") if seg.strip()]
    for seg in segments:
        if not any(seg.startswith(h) for h in _READONLY_HEADS):
            return False
    return True


def execute_readonly(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a parallel-safe command in a fresh subprocess.

    By construction (is_parallel_safe rejects anything that writes),
    we skip absorb_gate; the fresh subprocess has no persistent state
    to leak back into the session.
    """
    if timeout > MAX_BASH_TIMEOUT:
        timeout = MAX_BASH_TIMEOUT
    MAX_LINES = 2000
    MAX_BYTES = 256 * 1024
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "TERM": "dumb"},
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    except Exception as e:
        return f"[exec error: {e}]"
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = out.splitlines(keepends=True)
    byte_count = sum(len(l) for l in lines)
    if len(lines) > MAX_LINES or byte_count > MAX_BYTES:
        head = lines[:MAX_LINES]
        byte_count = sum(len(l) for l in head)
        head.append(
            f"\n[output truncated: captured {len(head)} lines / "
            f"{byte_count} bytes. To continue: sed -n "
            f"'{len(head)+1},$p' <file> or narrow the command.]\n"
        )
        out = "".join(head)
    if proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return out.strip()



# ---------------------------------------------------------------------------
# Neutral tool spec
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """Provider-neutral tool description.

    `anthropic_type` is set for Anthropic-native built-ins like
    `bash_20250124`; that value is only honoured by AnthropicProvider.
    For OpenAI-compatible providers a JSON schema under `parameters` is
    used instead.
    """
    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    anthropic_type: str | None = None


# Built-in neutral tool specs
BASH_TOOL_SPEC = ToolSpec(
    name="bash",
    description="Execute a bash command in the persistent session.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "restart": {"type": "boolean"},
        },
    },
    anthropic_type="bash_20250124",
)


# Round 7: delegate tool. Lets the orchestrator role dispatch a
# self-contained sub-task to a specialist role with isolated history.
# The specialist sees a fresh `messages=[]`, runs its own loop inside
# its own max_iterations budget, and returns its final answer as the
# tool_result string. Specialists cannot themselves delegate — the
# agent loop gates this via _reroute_depth.
#
# Role choices:
#   code    — Opus 4.7 + bash, 50-iter, heavy agentic debug loops
#   task    — Sonnet 4.6 + bash, 10-iter, execution/verification
#   create  — Sonnet 4.6, 3-iter, writing / brainstorm (no tools)
#   local   — Nemotron FP8 via local vLLM, 3-iter (no tools)
#   chat    — Opus 4.6, 1-iter, voice / reflection (no tools)
DELEGATE_TOOL_SPEC = ToolSpec(
    name="delegate",
    description=(
        "Dispatch a self-contained sub-task to a specialist role with an "
        "isolated message history. Use this when the current turn "
        "decomposes into distinct pieces that different substrates handle "
        "better. The sub-task string must be fully self-contained — the "
        "specialist has no access to the orchestrator's conversation. "
        "Returns the specialist's final answer as the tool result. "
        "Specialists cannot themselves delegate."
    ),
    parameters={
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": ["code", "task", "create", "local", "chat"],
                "description": (
                    "Which specialist to dispatch to. code: Opus 4.7 + bash, "
                    "50-iter, agentic debug. task: Sonnet 4.6 + bash, 10-iter, "
                    "execution/verification. create: Sonnet 4.6, 3-iter, "
                    "writing/brainstorm (no tools). local: Nemotron FP8 via "
                    "local vLLM, 3-iter (no tools). chat: Opus 4.6, 1-iter, "
                    "voice/reflection (no tools)."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "The self-contained task description for the specialist. "
                    "Include any context the specialist needs — they see a "
                    "fresh conversation."
                ),
            },
        },
        "required": ["role", "task"],
    },
    anthropic_type=None,
)


# ---------------------------------------------------------------------------
# BashSession (ported unchanged from vybn_spark_agent.py)
# ---------------------------------------------------------------------------

class BashTool:
    """Persistent bash session with sentinel-line protocol.

    Kept byte-compatible with the original BashSession so downstream
    invariants (timeouts, restart semantics, sentinel) do not drift.
    The `absorb_gate` is enforced on every execute().
    """

    def __init__(self) -> None:
        self._sentinel = "___VYBN_CMD_DONE___"
        self._start_process()

    def _start_process(self) -> None:
        self.process = subprocess.Popen(
            ["/bin/bash"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "TERM": "dumb", "PS1": ""},
        )
        os.set_blocking(self.process.stdout.fileno(), False)

    def execute(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        # Clamp to the hard wall-clock ceiling. A caller passing a
        # multi-hour timeout would otherwise stall the whole turn
        # on a network-partitioned ssh/curl.
        if timeout > MAX_BASH_TIMEOUT:
            timeout = MAX_BASH_TIMEOUT
        gate = absorb_gate(command)
        if gate is not None:
            return gate
        if "VYBN_ABSORB_REASON=" in command:
            log_absorb(command)
        full_cmd = f"{command}\necho {self._sentinel} $?\n"
        try:
            self.process.stdin.write(full_cmd)
            self.process.stdin.flush()
        except BrokenPipeError:
            return self.restart()

        lines: list[str] = []
        byte_count = 0
        start = time.time()
        # Ceilings chosen so a single 600-line source file fits
        # whole. Hitting either cap emits a resume hint instead
        # of silent truncation, so the caller can page cleanly.
        MAX_LINES = 2000
        MAX_BYTES = 256 * 1024
        while True:
            if time.time() - start > timeout:
                self._interrupt()
                lines.append(f"\n[timed out after {timeout}s]")
                self._drain(2)
                break
            try:
                line = self.process.stdout.readline()
            except Exception as e:
                lines.append(f"[read error: {e}]")
                break
            if not line:
                time.sleep(0.05)
                continue
            if self._sentinel in line:
                sentinel_idx = line.find(self._sentinel)
                leading = line[:sentinel_idx]
                if leading.rstrip():
                    # Content preceded the sentinel on the same line —
                    # keep it so commands whose last byte isn't \n
                    # (curl -s, echo -n, grep -c, JSON bodies, etc.)
                    # don't silently lose their tail.
                    lines.append(leading if leading.endswith("\n") else leading + "\n")
                tail = line[sentinel_idx + len(self._sentinel):]
                parts = tail.strip().split()
                code = parts[-1] if parts else "0"
                if code != "0":
                    lines.append(f"[exit code: {code}]")
                break
            lines.append(line)
            byte_count += len(line)
            if len(lines) > MAX_LINES or byte_count > MAX_BYTES:
                reached = (
                    f"lines>{MAX_LINES}" if len(lines) > MAX_LINES
                    else f"bytes>{MAX_BYTES}"
                )
                lines.append(
                    f"\n[output truncated: {reached}; "
                    f"{len(lines)} lines / {byte_count} bytes captured. "
                    f"To continue: sed -n '{len(lines)+1},$p' <file>  "
                    f"or pipe to a smaller window (head/tail/grep/sed).]\n"
                )
                self._drain(10)
                break
        return "".join(lines).strip()

    def _interrupt(self) -> None:
        try:
            self.process.stdin.write("\x03\n")
            self.process.stdin.flush()
        except Exception:
            pass

    def _drain(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                line = self.process.stdout.readline()
                if line and self._sentinel in line:
                    break
            except Exception:
                break
            time.sleep(0.05)

    def restart(self) -> str:
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self._start_process()
        return "(bash session restarted)"