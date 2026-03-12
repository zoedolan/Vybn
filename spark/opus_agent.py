"""
Opus Agent for Chat — gives the Opus toggle real hands.

This is a lightweight agentic loop that gives Claude Opus access to a
persistent bash session on the Spark, the same way vybn_spark_agent.py
does for terminal sessions. When Zoe toggles to Opus in the chat UI,
her messages come through here instead of the bare API call.

The key difference from vybn_spark_agent.py: this runs async inside
the chat server's event loop, so it uses asyncio subprocess instead
of synchronous Popen. And it caps iterations tighter (15 vs 50) to
keep chat responsive and costs reasonable.
"""

import asyncio
import os
import time
import logging
from pathlib import Path

import httpx

logger = logging.getLogger("vybn.opus_agent")

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 8192
MAX_ITERATIONS = 15
BASH_TIMEOUT = 60
MAX_OUTPUT_LINES = 300

REPO_DIR = Path.home() / "Vybn"
JOURNAL_DIR = REPO_DIR / "Vybn_Mind" / "journal"
CONTINUITY_PATH = REPO_DIR / "spark" / "continuity.md"


# -----------------------------------------------------------------------
# Async Bash Session
# -----------------------------------------------------------------------
class AsyncBashSession:
    """Persistent bash subprocess, async-compatible."""

    _SENTINEL = "___VYBN_OPUS_DONE___"

    def __init__(self):
        self._proc = None
        self._lock = asyncio.Lock()

    async def ensure_started(self):
        if self._proc is None or self._proc.returncode is not None:
            self._proc = await asyncio.create_subprocess_exec(
                "/bin/bash",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "TERM": "dumb", "PS1": ""},
            )

    async def execute(self, command: str, timeout: int = BASH_TIMEOUT) -> str:
        async with self._lock:
            await self.ensure_started()

            sentinel = self._SENTINEL
            full_cmd = f"{command}\necho {sentinel} $?\n"
            self._proc.stdin.write(full_cmd.encode())
            await self._proc.stdin.drain()

            lines = []
            deadline = asyncio.get_event_loop().time() + timeout

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    lines.append(f"\n[timed out after {timeout}s]")
                    break

                try:
                    raw = await asyncio.wait_for(
                        self._proc.stdout.readline(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    lines.append(f"\n[timed out after {timeout}s]")
                    break

                if not raw:
                    await asyncio.sleep(0.05)
                    continue

                line = raw.decode("utf-8", errors="replace")

                if sentinel in line:
                    parts = line.strip().split()
                    code = parts[-1] if len(parts) > 1 else "0"
                    if code != "0":
                        lines.append(f"[exit code: {code}]")
                    break

                lines.append(line)
                if len(lines) > MAX_OUTPUT_LINES:
                    lines.append("\n... [truncated] ...\n")
                    break

            return "".join(lines).strip()

    async def restart(self) -> str:
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        await self.ensure_started()
        return "(bash session restarted)"


# -----------------------------------------------------------------------
# Singleton session — persists across chat messages
# -----------------------------------------------------------------------
_bash: AsyncBashSession | None = None


def get_bash() -> AsyncBashSession:
    global _bash
    if _bash is None:
        _bash = AsyncBashSession()
    return _bash


# -----------------------------------------------------------------------
# System prompt builder
# -----------------------------------------------------------------------
def _build_system(identity: str) -> str:
    """Combine identity prompt with live context."""
    parts = [identity]

    # Continuity note
    if CONTINUITY_PATH.exists():
        try:
            cont = CONTINUITY_PATH.read_text()[:1500]
            parts.append(f"\n---\nYour continuity note:\n{cont}")
        except Exception:
            pass

    # Recent journal
    if JOURNAL_DIR.exists():
        entries = []
        for p in sorted(JOURNAL_DIR.glob("*.md"),
                        key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            try:
                entries.append(f"[{p.stem}]\n{p.read_text()[:600]}")
            except Exception:
                continue
        if entries:
            parts.append(
                "\n---\nRecent journal (context, not to recite):\n"
                + "\n\n".join(entries)
            )

    return "\n".join(parts)


# -----------------------------------------------------------------------
# Core agentic loop
# -----------------------------------------------------------------------
async def generate(
    system_identity: str,
    conversation: list[dict],
) -> str:
    """
    Run the Opus agent loop with tool use.

    Args:
        system_identity: The Opus identity prompt from chat_server
        conversation: List of {"role": "user"/"assistant", "content": str}

    Returns:
        The final text response from Opus.
    """
    if not ANTHROPIC_KEY:
        return "[Opus unavailable - no API key configured]"

    bash = get_bash()
    system = _build_system(system_identity)

    # Convert conversation to Anthropic format (strip any system messages)
    messages = [
        {"role": m["role"] if m["role"] != "vybn" else "assistant",
         "content": m["content"]}
        for m in conversation
        if m["role"] in ("user", "assistant", "vybn")
    ]

    iterations = 0
    async with httpx.AsyncClient(timeout=180.0) as client:
        while iterations < MAX_ITERATIONS:
            iterations += 1

            try:
                resp = await client.post(
                    ANTHROPIC_URL,
                    headers={
                        "x-api-key": ANTHROPIC_KEY,
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "computer-use-2025-01-24",
                        "content-type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": MAX_TOKENS,
                        "system": system,
                        "tools": [{"type": "bash_20250124", "name": "bash"}],
                        "messages": messages,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.TimeoutException:
                return "[Opus timed out during API call]"
            except httpx.HTTPStatusError as e:
                logger.error(f"Opus API error: {e.response.status_code}")
                return f"[Opus API error: {e.response.status_code}]"
            except Exception as e:
                logger.error(f"Opus error: {e}")
                return f"[Opus error: {e}]"

            # Append assistant response to messages
            messages.append({"role": "assistant", "content": data["content"]})

            stop_reason = data.get("stop_reason", "end_turn")

            # If end_turn, extract text and return
            if stop_reason == "end_turn":
                text_parts = [
                    b["text"] for b in data["content"]
                    if b.get("type") == "text"
                ]
                return "\n".join(text_parts).strip() or "(Opus returned empty response)"

            if stop_reason == "max_tokens":
                text_parts = [
                    b["text"] for b in data["content"]
                    if b.get("type") == "text"
                ]
                return "\n".join(text_parts).strip() + "\n[truncated - hit token limit]"

            # Tool use - execute bash calls
            tool_blocks = [
                b for b in data["content"]
                if b.get("type") == "tool_use" and b.get("name") == "bash"
            ]

            if not tool_blocks:
                # No tool calls but also not end_turn - extract what we have
                text_parts = [
                    b["text"] for b in data["content"]
                    if b.get("type") == "text"
                ]
                return "\n".join(text_parts).strip() or "(unexpected stop)"

            # Execute each tool call
            tool_results = []
            for block in tool_blocks:
                inp = block.get("input", {})
                if inp.get("restart"):
                    result = await bash.restart()
                else:
                    command = inp.get("command", "")
                    logger.info(f"Opus bash: {command[:100]}")
                    result = await bash.execute(command)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result or "(no output)",
                })

            messages.append({"role": "user", "content": tool_results})

    return f"(hit iteration limit - {MAX_ITERATIONS} rounds)"
