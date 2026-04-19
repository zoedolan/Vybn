"""Providers — how the model speaks to the world.

Two concerns, one concept: provider classes (Anthropic, OpenAI-compatible)
and the tool surface they expose. Previously these lived in two files
(providers.py, tools.py); the split was incidental. A tool spec only
matters because a provider renders it; a provider only matters because
it can be handed tool specs. Fold, do not pile.

This file contains:

    ToolSpec — provider-neutral tool description.
    BASH_TOOL_SPEC / DELEGATE_TOOL_SPEC / INTROSPECT_TOOL_SPEC — the
        three built-in tools. BASH uses the Anthropic bash_20250124
        native shape; DELEGATE and INTROSPECT are OpenAI-style function
        schemas that each provider translates.

    absorb_gate + log_absorb + validate_command — safety wrappers BashTool
        enforces on every execute(). These are policy-rule enforcement;
        the rules themselves (DANGEROUS_PATTERNS, TRACKED_REPOS) live in
        harness.policy and are imported here.

    is_parallel_safe + execute_readonly — the classifier and runner for
        read-only commands that can fan out across fresh subprocesses
        without touching the persistent shell.

    BashTool — the persistent bash session with sentinel protocol.

    Provider / AnthropicProvider / OpenAIProvider — the narrow stream()
        interface + their ProviderRegistry.

Mixed-provider sessions are supported: AnthropicProvider._normalize_messages_
for_anthropic and OpenAIProvider._messages_for_openai translate each
other's native shapes, so a code turn that started on Opus and fell back
to Sonnet or GPT still round-trips cleanly.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from .policy import (
    ABSORB_EXCLUDE_SUBSTR,
    ABSORB_EXCLUDE_SUFFIX,
    ABSORB_LOG,
    DANGEROUS_PATTERNS,
    DEFAULT_TIMEOUT,
    MAX_BASH_TIMEOUT,
    RoleConfig,
    TRACKED_REPOS,
)
from .substrate import LayeredPrompt


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


# Round 9: introspect tool. Orchestrate-only. Returns a compact live-system
# snapshot so the orchestrator can plan against reality rather than assumptions.
INTROSPECT_TOOL_SPEC = ToolSpec(
    name="introspect",
    description=(
        "Return a compact snapshot of the live Vybn system state: last 5 "
        "routing decisions, recent audit entries, current walk alpha and step, "
        "and service health. Orchestrate-only. No arguments required."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    anthropic_type=None,
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


# ---------------------------------------------------------------------------
# Neutral response / tool shapes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class NormalizedResponse:
    """Provider-neutral shape returned by stream().

    `raw_assistant_content` is the provider's native representation of
    the assistant turn; we pass it straight back into the next request
    so tool-use IDs stay aligned with tool_results.
    """
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "error"
    in_tokens: int = 0
    out_tokens: int = 0
    # Cache telemetry (Anthropic prompt caching). Anthropic dropped
    # ephemeral TTL 1h -> 5min on 2026-03-06; without visibility we
    # can't tell if the LayeredPrompt cache_control markers hit.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    raw_assistant_content: Any = None
    provider: str = ""
    model: str = ""


@dataclass
class StreamHandle:
    """Handle returned by provider.stream(); iterating yields text chunks
    and thinking indicators, and final() returns a NormalizedResponse."""
    iterator: Iterator[tuple[str, str]]  # (kind, chunk) where kind in {"text","thinking"}
    finalize: Any  # callable returning NormalizedResponse

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self.iterator

    def final(self) -> NormalizedResponse:
        return self.finalize()


class Provider(Protocol):
    name: str

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle: ...

    def build_tool_result(self, tool_call_id: str, content: str) -> dict: ...


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self.client = client
        else:
            import anthropic  # type: ignore
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )

    @staticmethod
    def _normalize_messages_for_anthropic(messages: list[dict]) -> list[dict]:
        """Rewrite messages so every entry is Anthropic-valid.

        Mixed-provider sessions can leave OpenAI-native shapes in the
        rolling history: {"role":"assistant","content":<openai_dict>} or
        {"role":"tool","tool_call_id":...,"content":...}. Anthropic
        rejects both with 400 ("messages.X.content: Input should be a
        valid list"). We translate them to Anthropic content-block form.
        Pure-Anthropic turns pass through unchanged.
        """
        out: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results() -> None:
            if pending_tool_results:
                out.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # OpenAI-shaped tool response: collapse into an Anthropic
            # tool_result block on a user message.
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else str(content or ""),
                })
                continue

            _flush_tool_results()

            if role == "assistant":
                # Assistant content must be a string or a list of
                # content blocks for Anthropic. The agent loop stores
                # raw_assistant_content straight in `content` — for
                # OpenAI turns that's a dict with its own role/content/
                # tool_calls keys. Re-emit in block form.
                if isinstance(content, dict) and "role" in content:
                    text = content.get("content") or ""
                    blocks: list[dict] = []
                    if isinstance(text, str) and text:
                        blocks.append({"type": "text", "text": text})
                    for tc in content.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        raw_args = fn.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                args = json.loads(raw_args or "{}")
                            except Exception:
                                args = {}
                        else:
                            args = raw_args or {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        })
                    if not blocks:
                        blocks.append({"type": "text", "text": ""})
                    out.append({"role": "assistant", "content": blocks})
                    continue
                # Pure-Anthropic assistant content (list of block
                # objects) or plain string — leave it alone.
                out.append(msg)
                continue

            if role == "user":
                # User content can be a string or a list of blocks. If
                # it's a non-block dict (shouldn't normally happen)
                # coerce to string to avoid a 400.
                if isinstance(content, dict):
                    out.append({"role": "user", "content": str(content)})
                else:
                    out.append(msg)
                continue

            # Unknown roles (system would normally be stripped upstream)
            # are passed through; Anthropic will surface errors clearly.
            out.append(msg)

        _flush_tool_results()
        return out

    def _translate_tools(self, tools: list[ToolSpec]) -> list[dict]:
        out: list[dict] = []
        for t in tools:
            if t.anthropic_type:
                out.append({"type": t.anthropic_type, "name": t.name})
            else:
                out.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                })
        return out

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle:
        kwargs: dict[str, Any] = {
            "model": role.model,
            "max_tokens": role.max_tokens,
            "system": system.anthropic_blocks() or system.flat(),
            "messages": self._normalize_messages_for_anthropic(messages),
        }
        if tools:
            kwargs["tools"] = self._translate_tools(tools)
        if role.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["extra_body"] = {"context_management": {"edits": [
                {"type": "clear_thinking_20251015"},
                {"type": "clear_tool_uses_20250919",
                 "trigger": {"type": "input_tokens", "value": 160000},
                 "keep": {"type": "tool_uses", "value": 6}},
            ]}}
            kwargs["extra_headers"] = {
                "anthropic-beta": "context-management-2025-06-27"
            }

        stream_cm = self.client.messages.stream(**kwargs)
        stream = stream_cm.__enter__()
        closed = {"v": False}

        def _close() -> None:
            # Idempotent: __exit__ is called from whichever of _iter or
            # _final runs to completion or raises first. Without this,
            # a KeyboardInterrupt during streaming leaks the SDK
            # context (open HTTP connection, unreleased locks) because
            # _final() is never invoked.
            if closed["v"]:
                return
            closed["v"] = True
            try:
                stream_cm.__exit__(None, None, None)
            except Exception:
                pass

        def _iter() -> Iterator[tuple[str, str]]:
            try:
                for event in stream:
                    kind = getattr(event, "type", "")
                    if kind == "thinking":
                        yield ("thinking", "")
                    elif kind == "text":
                        yield ("text", getattr(event, "text", ""))
            except BaseException:
                _close()
                raise

        def _final() -> NormalizedResponse:
            try:
                msg = stream.get_final_message()
            finally:
                _close()
            calls: list[ToolCall] = []
            text_parts: list[str] = []
            for block in msg.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif btype == "tool_use":
                    calls.append(ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}) or {},
                    ))
            usage = getattr(msg, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            return NormalizedResponse(
                text="\n".join(text_parts),
                tool_calls=calls,
                stop_reason=getattr(msg, "stop_reason", "") or "end_turn",
                in_tokens=in_tok,
                out_tokens=out_tok,
                cache_creation_tokens=cache_create,
                cache_read_tokens=cache_read,
                raw_assistant_content=msg.content,
                provider=self.name,
                model=role.model,
            )

        return StreamHandle(iterator=_iter(), finalize=_final)

    def build_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content or "(no output)",
        }


# ---------------------------------------------------------------------------
# OpenAIProvider — also used for local OpenAI-compatible vLLM / Nemotron
# ---------------------------------------------------------------------------

# Round 5 hotfix: Nemotron (and other reasoning-style vLLM models) emit
# chain-of-thought inline in `content` wrapped in <think>…</think> tags.
# These must NEVER reach Zoe — they are internal scratchpad, not output.
# If a closing </think> is present, everything up to and including it is
# dropped. If the opening <think> appears without a close (truncation or
# malformed stream), we drop from that point on and let the remaining
# reply stand on whatever came before. If neither tag appears, the text
# flows through unchanged.
_THINK_BLOCK = re.compile(r"^\s*(?:<think>)?\s*.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_ONLY = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    if not text:
        return text
    # Fast path: no think marker at all.
    if "think>" not in text.lower():
        return text
    cleaned = _THINK_BLOCK.sub("", text, count=1)
    # Leftover unclosed <think> (rare: truncation / model error).
    if "<think>" in cleaned.lower():
        cleaned = _THINK_OPEN_ONLY.sub("", cleaned)
    return cleaned.strip()


class OpenAIProvider:
    """OpenAI-compatible provider.

    Works for:
      - OpenAI cloud (GPT family): base_url=None, OPENAI_API_KEY
      - Local vLLM / Nemotron (OpenAI-shaped API): base_url set in role.
    """

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.base_url = base_url

    def _translate_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {
                        "type": "object", "properties": {}
                    },
                },
            }
            for t in tools
        ]

    def _messages_for_openai(
        self, system: LayeredPrompt, messages: list[dict]
    ) -> list[dict]:
        """Flatten the layered prompt and normalize Anthropic-shaped
        assistant / tool_result messages into OpenAI shape.

        This is the translation boundary. Tool calls coming in are
        already in neutral `ToolCall` shape (they were produced by a
        provider); tool_result messages arrive as Anthropic-native
        dicts because that's what the agent loop emits today. We map:

            {"role":"user","content":[{"type":"tool_result",...}]} →
                {"role":"tool","tool_call_id":...,"content":...}
        """
        out: list[dict] = [{"role": "system", "content": system.flat()}]
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user" and isinstance(content, list) and content and \
               isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                for item in content:
                    out.append({
                        "role": "tool",
                        "tool_call_id": item.get("tool_use_id", ""),
                        "content": item.get("content", ""),
                    })
            elif role == "assistant" and not isinstance(content, str):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                # Anthropic "thinking" / "redacted_thinking" blocks
                # have no OpenAI equivalent. If the code role (Opus
                # with adaptive thinking) runs, fails, and falls back
                # to Sonnet or GPT, the cloud endpoint would reject
                # the thinking blocks. Silently drop unknown types.
                for block in content or []:
                    btype = getattr(block, "type", None) or (
                        block.get("type") if isinstance(block, dict) else None
                    )
                    if btype == "text":
                        text_parts.append(
                            getattr(block, "text", None)
                            or (block.get("text") if isinstance(block, dict) else "")
                        )
                    elif btype == "tool_use":
                        tc_id = getattr(block, "id", None) or (
                            block.get("id") if isinstance(block, dict) else ""
                        )
                        tc_name = getattr(block, "name", None) or (
                            block.get("name") if isinstance(block, dict) else ""
                        )
                        tc_args = getattr(block, "input", None) or (
                            block.get("input") if isinstance(block, dict) else {}
                        )
                        tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args or {}),
                            },
                        })
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            else:
                out.append({"role": role, "content": content})
        return out

    def _call(
        self, role: RoleConfig, openai_messages: list[dict], tools: list[ToolSpec]
    ) -> dict:
        base = role.base_url or self.base_url
        # For vLLM/Nemotron deployments that are served at host:port without
        # the `/v1` suffix, the chat-completions URL would otherwise miss
        # `/v1`. We normalise here so role configs can specify either form.
        if base and not base.rstrip("/").endswith("/v1"):
            base = base.rstrip("/") + "/v1"

        # Cloud OpenAI (no base_url) requires max_completion_tokens for
        # GPT-5.x and o-series models; passing the legacy max_tokens key
        # returns HTTP 400. Local vLLM / Nemotron (base_url set) still
        # speaks the legacy key. Branch on transport rather than model
        # name so new OpenAI-compatible models don't need a code change.
        max_key = "max_tokens" if base else "max_completion_tokens"
        payload: dict[str, Any] = {
            "model": role.model,
            "messages": openai_messages,
            max_key: role.max_tokens,
            "temperature": role.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = self._translate_tools(tools)

        # Prefer the official SDK when available (handles auth, retries).
        # Only swallow ImportError — real API failures must propagate with
        # context rather than getting masked by the raw-HTTP fallback.
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            OpenAI = None  # type: ignore

        if OpenAI is not None:
            try:
                client = OpenAI(
                    api_key=self.api_key, base_url=base, timeout=300.0,
                )
                resp = client.chat.completions.create(**payload)
                return (
                    resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                )
            except Exception as exc:
                # Connection / transport problems to a local vLLM that has
                # gone away get retried via plain HTTP below. Any other
                # error (auth, bad-request from cloud OpenAI) propagates.
                msg = str(exc).lower()
                transport_signals = (
                    "connection", "refused", "timed out",
                    "connect", "name or service", "temporar",
                )
                if not any(sig in msg for sig in transport_signals):
                    raise

        # Fallback: plain HTTP — works for local vLLM without openai SDK
        # or when the SDK hit a transport issue against a local server.
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIProvider needs either the `openai` SDK or `requests`"
            ) from exc

        url = (base.rstrip("/") if base else "https://api.openai.com/v1") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(url, json=payload, headers=headers, timeout=300)
        if r.status_code >= 400:
            body = r.text[:500] if r.text else ""
            raise RuntimeError(
                f"OpenAI-compatible call failed: HTTP {r.status_code} "
                f"from {url}: {body}"
            )
        return r.json()

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle:
        """For the OpenAI path we use non-streaming request-response for
        simplicity and then surface the result through the StreamHandle
        iterator as a single text chunk. This keeps the agent loop code
        identical across providers without committing to SSE parsing
        that differs subtly between vLLM and OpenAI cloud.
        """
        openai_messages = self._messages_for_openai(system, messages)
        data = self._call(role, openai_messages, tools)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = _strip_reasoning(msg.get("content") or "").strip()
        tool_calls_raw = msg.get("tool_calls") or []
        calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except Exception as json_exc:
                # Surface malformed tool-call JSON via sentinel keys so
                # the agent loop can hand a real error back to the
                # model instead of silently running an empty command.
                args = {
                    "__parse_error__": str(json_exc),
                    "__raw_arguments__": raw_args[:400],
                }
            calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))
        stop_reason = choice.get("finish_reason") or "end_turn"
        if stop_reason == "stop":
            stop_reason = "end_turn"
        elif stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "length":
            stop_reason = "max_tokens"

        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)

        def _iter() -> Iterator[tuple[str, str]]:
            if text:
                yield ("text", text)

        finalized = NormalizedResponse(
            text=text,
            tool_calls=calls,
            stop_reason=stop_reason,
            in_tokens=in_tok,
            out_tokens=out_tok,
            raw_assistant_content=msg,
            provider=self.name,
            model=role.model,
        )

        return StreamHandle(iterator=_iter(), finalize=lambda: finalized)

    def build_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content or "(no output)",
        }


# ---------------------------------------------------------------------------
# Registry — constructs providers lazily so a missing SDK for one
# provider doesn't break the other.
# ---------------------------------------------------------------------------

class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def get(self, role: RoleConfig) -> Provider:
        # Local OpenAI-compatible paths get their own instance so the
        # base_url is captured at construction.
        key = role.provider
        if role.provider == "openai" and role.base_url:
            key = f"openai::{role.base_url}"
        if key in self._providers:
            return self._providers[key]
        if role.provider == "anthropic":
            self._providers[key] = AnthropicProvider()
        elif role.provider == "openai":
            self._providers[key] = OpenAIProvider(base_url=role.base_url)
        else:
            raise ValueError(f"unknown provider: {role.provider}")
        return self._providers[key]
