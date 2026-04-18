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
        start = time.time()
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
                parts = line.strip().split()
                code = parts[-1] if len(parts) > 1 else "0"
                if code != "0":
                    lines.append(f"[exit code: {code}]")
                break
            lines.append(line)
            if len(lines) > 500:
                lines.append("\n... [truncated at 500 lines] ...\n")
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
