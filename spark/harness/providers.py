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
        harness.substrate and are imported here.

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

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .substrate import (
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
# Provider credential environment loading
# ---------------------------------------------------------------------------

# Only these keys are eligible to be injected. Whitelisting keeps an
# accidentally-committed llm.env from quietly enabling unrelated env
# vars on the running service.
_ALLOWED_KEYS: frozenset[str] = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "TOGETHER_API_KEY",
    "MISTRAL_API_KEY",
})

# KEY=value, with optional leading `export ` and optional matched
# single- or double-quotes around the value. Values stop at EOL or
# unquoted `#`. We deliberately do not expand $VAR or $(…).
_LINE = re.compile(
    r"""^\s*(?:export\s+)?
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        (?:
          "(?P<dq>(?:[^"\\]|\\.)*)" |
          '(?P<sq>[^']*)' |
          (?P<bare>[^#\n\r]*?)
        )
        \s*(?:\#.*)?\s*$
    """,
    re.VERBOSE,
)


def _safe_path(p: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(p).expanduser()
    except (RuntimeError, OSError):
        return None
    if not path.is_file():
        return None
    try:
        # Readable in this process without elevation? os.access handles
        # the current euid without raising on unreadable files.
        if not os.access(path, os.R_OK):
            return None
    except OSError:
        return None
    return path


def _parse(path: Path) -> dict[str, str]:
    """Return KEY→value for whitelisted keys found in file. No logging."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key = m.group("key")
        if key not in _ALLOWED_KEYS:
            continue
        val = m.group("dq")
        if val is not None:
            # decode the trivial \\n / \\" escapes we allow inside "..."
            val = val.encode("utf-8").decode("unicode_escape", errors="replace")
        else:
            val = m.group("sq")
            if val is None:
                val = (m.group("bare") or "").strip()
        if not val:
            continue
        out[key] = val
    return out


def load_env_files(
    paths: Iterable[str | os.PathLike[str]] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Merge provider credentials from ~/.config/vybn/llm.env (and
    optionally /etc/environment) into os.environ.

    Returns a dict of {KEY: source_path} — ONLY the keys we actually
    set — suitable for a non-sensitive status line. Values are never
    returned and never logged.

    Precedence: earlier paths win over later paths. Existing os.environ
    values always win unless overwrite=True.
    """
    if paths is None:
        paths = (
            "~/.config/vybn/llm.env",
            "/etc/environment",
        )

    applied: dict[str, str] = {}
    seen: dict[str, str] = {}  # key -> first source that provided it

    for p in paths:
        sp = _safe_path(p)
        if sp is None:
            continue
        for key, val in _parse(sp).items():
            if key in seen:
                continue
            seen[key] = str(sp)
            if not overwrite and os.environ.get(key):
                # Respect the environment the process was launched with.
                continue
            os.environ[key] = val
            applied[key] = str(sp)
    return applied


def describe(applied: dict[str, str]) -> str:
    """Return a non-sensitive, printable summary. No values."""
    if not applied:
        return "no provider keys loaded from disk"
    keys = ", ".join(sorted(applied.keys()))
    return f"loaded {len(applied)} provider key(s) from disk: {keys}"


__all__ = ["load_env_files", "describe"]

# ---------------------------------------------------------------------------
# Provider-agnostic tool-call execution
# ---------------------------------------------------------------------------

@dataclass
class IntrospectionSnapshot:
    """Typed payload returned by the introspect tool."""

    recent_routes: list[dict] = field(default_factory=list)
    services: dict[str, dict] = field(default_factory=dict)
    verification_gaps: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


Printer = Callable[[str], None]



def default_introspect(spark_dir: str) -> str:
    """Live route/walk/deep-memory snapshot for the introspect tool.

    Returns typed JSON rather than prose so callers can assert contracts and
    future changes do not have to parse vibes.
    """
    import urllib.request
    from pathlib import Path

    snapshot = IntrospectionSnapshot()
    events_path = Path(spark_dir) / "agent_events.jsonl"
    try:
        events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
        routes = [e for e in events if e.get("event") == "route_decision"][-5:]
        snapshot.recent_routes = [
            {
                "turn": r.get("turn"),
                "role": r.get("role"),
                "provider": r.get("provider"),
                "model": r.get("model"),
                "reason": r.get("reason"),
            }
            for r in routes
        ]
    except Exception as e:  # noqa: BLE001
        snapshot.verification_gaps.append(f"events unavailable: {e}")

    for name, url in (
        ("walk", "http://127.0.0.1:8101/health"),
        ("deep_memory", "http://127.0.0.1:8100/health"),
    ):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                health = json.loads(resp.read())
            snapshot.services[name] = {
                "reachable": True,
                "status": health.get("status"),
                "chunks": health.get("chunks"),
                "walk_step": health.get("walk_step") or health.get("step"),
                "walk_alpha": health.get("walk_alpha"),
            }
        except Exception as e:  # noqa: BLE001
            snapshot.services[name] = {"reachable": False}
            snapshot.verification_gaps.append(f"{name} unavailable: {e}")
    return snapshot.to_json()


def execute_tool_calls(
    response: Any,
    bash: Any,
    provider: Any,
    *,
    delegate_cb: Callable[[str, str], str] | None = None,
    dim: Printer = lambda text: None,
    warn: Printer = lambda text: None,
    preview: Printer = lambda text: None,
    introspect: Callable[[], str] | None = None,
) -> tuple[list, bool]:
    """Run provider-neutral ToolCall objects and return native tool results."""
    results: list[dict] = []
    interrupted = False

    bash_calls = [c for c in response.tool_calls if c.name == "bash"]
    parallel_candidates: list[tuple[Any, str]] = []
    if len(bash_calls) >= 2:
        ok = True
        for call in bash_calls:
            args = call.arguments or {}
            if args.get("restart") or "__parse_error__" in args:
                ok = False
                break
            cmd = args.get("command", "") or ""
            valid, _ = validate_command(cmd)
            if not valid or not is_parallel_safe(cmd):
                ok = False
                break
            parallel_candidates.append((call, cmd))
        if ok and parallel_candidates:
            dim(f"[parallel: {len(parallel_candidates)} read-only bash calls]")
            out_by_id: dict[str, str] = {}
            with ThreadPoolExecutor(max_workers=min(8, len(parallel_candidates))) as ex:
                future_to_call = {
                    ex.submit(execute_readonly, cmd): call
                    for call, cmd in parallel_candidates
                }
                for fut in future_to_call:
                    c = future_to_call[fut]
                    try:
                        out_by_id[c.id] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        out_by_id[c.id] = f"(parallel exec error: {e})"
            first_cmd = parallel_candidates[0][1]
            dim(f"$ {first_cmd[:200]}{'...' if len(first_cmd) > 200 else ''}")
            preview(out_by_id[parallel_candidates[0][0].id])
            for call in response.tool_calls:
                if call.id in out_by_id:
                    results.append(provider.build_tool_result(call.id, out_by_id[call.id]))
                elif call.name != "bash":
                    results.append(provider.build_tool_result(
                        call.id, f"(unsupported tool: {call.name})"
                    ))
            return results, False

    for call in response.tool_calls:
        if call.name == "introspect":
            out = introspect() if introspect is not None else "(introspect unavailable)"
            results.append(provider.build_tool_result(call.id, out))
            continue

        if call.name == "delegate":
            if delegate_cb is None:
                results.append(provider.build_tool_result(
                    call.id,
                    "(delegate unavailable: specialists cannot themselves "
                    "delegate; only the orchestrator role may dispatch)",
                ))
                continue
            if interrupted:
                results.append(provider.build_tool_result(call.id, "(skipped — interrupted)"))
                continue
            try:
                args = call.arguments or {}
                if "__parse_error__" in args:
                    err = args["__parse_error__"]
                    raw = args.get("__raw_arguments__", "")
                    out = f"(delegate error: malformed JSON arguments — {err}; raw={raw!r})"
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                sub_role = (args.get("role") or "").strip()
                sub_task = (args.get("task") or "").strip()
                if not sub_role or not sub_task:
                    out = "(delegate error: both `role` and `task` are required)"
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                if sub_role not in ("code", "task", "create", "local", "chat"):
                    out = (
                        f"(delegate error: unknown role {sub_role!r}; must be "
                        "one of code/task/create/local/chat)"
                    )
                    warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                dim(f"[delegate -> {sub_role}] {sub_task[:160]}{'...' if len(sub_task) > 160 else ''}")
                try:
                    sub_out = delegate_cb(sub_role, sub_task)
                except KeyboardInterrupt:
                    interrupted = True
                    results.append(provider.build_tool_result(call.id, "(delegate interrupted by user)"))
                    continue
                except Exception as e:  # noqa: BLE001
                    sub_out = f"(delegate error: {e})"
                    warn(sub_out)
                results.append(provider.build_tool_result(call.id, sub_out or "(delegate returned no text)"))
            except KeyboardInterrupt:
                interrupted = True
                results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            continue

        if call.name != "bash":
            results.append(provider.build_tool_result(call.id, f"(unsupported tool: {call.name})"))
            continue
        if interrupted:
            results.append(provider.build_tool_result(call.id, "(skipped — interrupted)"))
            continue

        try:
            args = call.arguments or {}
            if "__parse_error__" in args:
                err = args["__parse_error__"]
                raw = args.get("__raw_arguments__", "")
                out = f"(tool-call error: malformed JSON arguments — {err}; raw={raw!r})"
                warn(out)
                results.append(provider.build_tool_result(call.id, out))
                continue
            if args.get("restart"):
                out = bash.restart()
                dim("[bash session restarted]")
            else:
                command = args.get("command", "") or ""
                ok, reason = validate_command(command)
                if ok:
                    dim(f"$ {command[:200]}{'...' if len(command) > 200 else ''}")
                    out = bash.execute(command)
                    preview(out)
                else:
                    out = reason or "(blocked)"
                    warn(out)
            results.append(provider.build_tool_result(call.id, out))
        except KeyboardInterrupt:
            interrupted = True
            results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            warn("interrupted")

    return results, interrupted

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
#   code    — Opus 4.6 + bash, 50-iter, heavy agentic debug loops
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
                    "Which specialist to dispatch to. code: Opus 4.6 + bash, "
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

_REDIRECT_RE = re.compile(r"(?<![<>\-])>>?\s*([^\s<>|&;'\"]+)")
_TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s<>|&;'\"]+)")
_TOUCH_RE = re.compile(r"\btouch\s+([^\s<>|&;'\"]+)")
_SQ_RE = re.compile(r"'(?:[^'\\]|\\.)*'")
_DQ_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_HEREDOC_PAT = re.compile(r"<<-?\s*'?(\w+)'?[\s\S]*?^\1\s*$", re.MULTILINE)


def _strip_opaque(command: str) -> str:
    """Replace quoted strings and heredoc bodies with safe placeholders.
    A > inside a string literal is data, not a shell redirect."""
    s = _HEREDOC_PAT.sub(" __HEREDOC__ ", command)
    s = _DQ_RE.sub(" __DQ__ ", s)
    s = _SQ_RE.sub(" __SQ__ ", s)
    return s


def _extract_file_targets(command: str) -> list[str]:
    scan_text = _strip_opaque(command)
    out: list[str] = []
    for rx in (_REDIRECT_RE, _TEE_RE, _TOUCH_RE):
        for m in rx.finditer(scan_text):
            t = m.group(1).strip("'\"")
            if not t or t.startswith("/dev/") or t.startswith("/proc/"):
                continue
            if not os.path.isabs(t):
                # Resolve relative paths against the agent's cwd (the
                # bash session inherits ~/Vybn on launch). Compound
                # commands like 'mkdir -p X && cat > X/y <<EOF' used
                # to slip past because the target was relative. Over-
                # eager is the right failure mode: a false trigger
                # just prompts VYBN_ABSORB_REASON, a miss creates
                # bloat silently.
                t = os.path.abspath(t)
            out.append(os.path.normpath(t))
    return out[:10]


def absorb_gate(command: str) -> str | None:
    """Return refusal text if command would create a new tracked file
    without inline consolidation evidence. Otherwise None.

    New tracked-file creation requires two fields:
      - VYBN_ABSORB_REASON=... states why creation is justified.
      - VYBN_ABSORB_CONSIDERED=... names existing homes considered.

    The second field is deliberately separate. The recurring failure was that
    a fluent reason could be generated after the decision to create; ABC needs
    evidence of a prior consolidation search, not merely a persuasive story.

    Session-level override: when VYBN_ABSORB_OVERRIDE=1 is set in the
    environment of the running harness, the gate is bypassed for the
    rest of the session. The override is logged to audit.log with
    reason="session override active". Zoe's standing instruction
    (2026-04-21): "whenever something gets in your way, just remove it."
    The gate is a servant, not a constraint — when the principle it
    encodes blocks the work it was meant to enable, grant the override."""
    if os.environ.get("VYBN_ABSORB_OVERRIDE") == "1":
        try:
            with open(ABSORB_LOG, "a") as _f:
                _f.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\toverride\t"
                    f"session override active\t{command[:300]}\n"
                )
        except Exception:
            pass
        return None

    reason_present = "VYBN_ABSORB_REASON=" in command
    considered_present = "VYBN_ABSORB_CONSIDERED=" in command

    for tgt in _extract_file_targets(command):
        if not any(tgt == r or tgt.startswith(r + "/") for r in TRACKED_REPOS):
            continue
        if any(s in tgt for s in ABSORB_EXCLUDE_SUBSTR):
            continue
        if tgt.endswith(ABSORB_EXCLUDE_SUFFIX):
            continue
        if os.path.exists(tgt):
            continue
        if reason_present and considered_present:
            return None
        missing = []
        if not reason_present:
            missing.append("VYBN_ABSORB_REASON")
        if not considered_present:
            missing.append("VYBN_ABSORB_CONSIDERED")
        return (
            "[absorb_gate] refused. This command would create a new tracked "
            "file:\n"
            f"    {tgt}\n\n"
            "New-file creation is the agent's default failure mode. ABC "
            "requires evidence of consolidation before creation, not only a "
            "fluent justification after the fact. Missing: "
            f"{', '.join(missing)}.\n\n"
            "Before proceeding, in your reply to Zoe, name the existing files "
            "or modules you considered folding this into and why they did not "
            "fit. Then re-issue the command with both inline fields, e.g.:\n\n"
            "    VYBN_ABSORB_REASON=\"does not fold into X because ...\" "
            "VYBN_ABSORB_CONSIDERED=\"X: wrong lifecycle; Y: wrong layer\" "
            "<command>\n\n"
            "Fold, do not pile. If you are certain the new file is right, "
            "the considered homes are the record of that certainty."
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


def validate_command(
    command: str,
    *,
    allow_dangerous_literals_for_readonly: bool = False,
) -> tuple[bool, str | None]:
    if _has_shell_command_substitution(command or ""):
        return False, "Blocked: shell command substitution is not allowed in NEEDS-EXEC"
    """Return whether a shell command may execute.

    The blocklist protects against executable destructive intent. A harness
    that repairs itself must also inspect the strings that define its own
    guards. When the caller has already classified the command as read-only,
    dangerous-looking text inside grep/sed/cat/nl/git-grep arguments is data,
    not intent. Mutating commands remain blocked.
    """
    lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in lower:
            if allow_dangerous_literals_for_readonly and is_parallel_safe(command):
                continue
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


def _strip_leading_cd(command: str) -> str | None:
    """Strip one simple leading `cd PATH &&` used as environment setup."""
    c = command.strip()
    if not c.startswith("cd ") or "&&" not in c:
        return c
    prefix, rest = c.split("&&", 1)
    target = prefix[3:].strip().strip('"\'')
    if not target or any(ch in target for ch in ";&|<>`$(){}[]\n\r"):
        return None
    return rest.strip() or None


def _split_shell_segments(command: str) -> list[str]:
    """Split shell control operators outside quotes.

    This is intentionally smaller than a full shell parser. Its job is to
    classify executable heads while preserving quoted data as data.
    """
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escape = False
    i = 0
    while i < len(command):
        ch = command[i]
        if escape:
            buf.append(ch); escape = False; i += 1; continue
        if ch == "\\":
            buf.append(ch); escape = True; i += 1; continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1; continue
        if ch in ('"', "'"):
            quote = ch; buf.append(ch); i += 1; continue
        if ch == ";":
            seg = "".join(buf).strip()
            if seg: segments.append(seg)
            buf = []; i += 1; continue
        if command.startswith("&&", i) or command.startswith("||", i):
            seg = "".join(buf).strip()
            if seg: segments.append(seg)
            buf = []; i += 2; continue
        if ch == "|":
            seg = "".join(buf).strip()
            if seg: segments.append(seg)
            buf = []; i += 1; continue
        buf.append(ch); i += 1
    seg = "".join(buf).strip()
    if seg: segments.append(seg)
    return segments


def _readonly_head(tokens: list[str], segment: str) -> bool:
    if not tokens:
        return False
    if tokens[0] in {"cat", "ls", "ll", "grep", "rg", "fgrep", "egrep",
                     "head", "tail", "sed", "awk", "wc", "find", "locate",
                     "stat", "file", "tree", "echo", "printf", "date",
                     "whoami", "pwd", "env", "jq", "yq", "md5sum", "sha1sum",
                     "sha256sum", "diff", "cmp", "nl"}:
        return True
    if tokens[0] == "git" and len(tokens) >= 2 and tokens[1] in {
        "log", "status", "diff", "show", "rev-parse", "blame", "branch",
        "remote", "grep",
    }:
        return True
    if tokens[0] == "curl" and any(t.startswith("-s") or t.startswith("-fs") for t in tokens[1:]):
        return True
    if tokens[0] in {"python3", "python"} and len(tokens) >= 2:
        if tokens[1] == "-c":
            return True
        if len(tokens) >= 4 and tokens[1:3] == ["-m", "py_compile"]:
            return True
    return any(segment.startswith(h) for h in _READONLY_HEADS)


def _has_shell_command_substitution(command: str) -> bool:
    """Return True if raw shell text contains active command substitution."""
    in_single = False
    escaped = False
    for i, ch in enumerate(command or ""):
        if escaped:
            escaped = False
            continue
        if ch == chr(92) and not in_single:
            escaped = True
            continue
        if ch == chr(39):
            in_single = not in_single
            continue
        if in_single:
            continue
        if ch == chr(96):
            return True
        if ch == chr(36) and (command or "")[i + 1 : i + 2] == chr(40):
            return True
    return False

def is_parallel_safe(command: str) -> bool:
    """True when `command` can run in a fresh subprocess as read-only.

    This classifier is token-semantic rather than raw-text-semantic: mutating
    executable heads are refused, while alarming strings inside arguments to
    read-only inspection tools remain inspectable data.
    """
    if _has_shell_command_substitution(command or ""):
        return False
    work = _strip_leading_cd(command or "")
    if not work:
        return False
    for segment in _split_shell_segments(work):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            return False
        executable = tokens[0]
        if executable in {"cd", "pushd", "popd", "export", "unset", "source", ".",
                          "mv", "cp", "rm", "rmdir", "mkdir", "touch", "chmod",
                          "chown", "ln", "kill", "pkill", "killall", "systemctl",
                          "service", "docker"}:
            return False
        if executable in {"pip", "npm", "apt", "apt-get"}:
            return False
        if executable == "git" and len(tokens) >= 2 and tokens[1] in {
            "commit", "push", "pull", "merge", "rebase", "add", "reset",
            "checkout", "stash", "clone",
        }:
            return False
        if any(tok in {">", ">>", "<"} for tok in tokens):
            return False
        if "VYBN_ABSORB_REASON=" in segment:
            return False
        if not _readonly_head(tokens, segment):
            return False
    return True


def github_cli_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment for GitHub CLI calls.

    `gh` gives precedence to GITHUB_TOKEN over the stored hosts.yml
    credential. On the Sparks that env token can push git refs but lacks
    GraphQL createPullRequest permission, while the stored gh credential
    has the repo scope needed for PRs. Strip only this shadowing variable
    for gh invocations; leave git transport credentials untouched.
    """
    env = dict(os.environ if base is None else base)
    env.pop("GITHUB_TOKEN", None)
    return env


def normalize_github_cli_command(command: str) -> str:
    """Make shell-authored PR creation use the stored gh credential.

    This is intentionally narrow: only `gh pr create` is rewritten. Other
    gh calls keep their original environment, and explicit `env -u
    GITHUB_TOKEN gh pr create` commands are left alone.
    """
    if "gh pr create" not in command or "env -u GITHUB_TOKEN gh pr create" in command:
        return command
    return re.sub(r"(?<![\w./-])gh\s+pr\s+create\b", "env -u GITHUB_TOKEN gh pr create", command)


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
            env=github_cli_env({**os.environ, "TERM": "dumb"}) if "gh pr create" in command else {**os.environ, "TERM": "dumb"},
        )
    except subprocess.TimeoutExpired:
        return (
            f"[timed out after {timeout}s]\n"
            "[control event: fresh-subprocess-timeout; no persistent shell "
            "state was changed. Narrow the command or escalate to a tool role.]"
        )
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
        command = normalize_github_cli_command(command)
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


class Provider:
    name: str
    tool_target: str

    def stream(
        self,
        *,
        system: LayeredPrompt,
        messages: list[dict],
        tools: list[ToolSpec],
        role: RoleConfig,
    ) -> StreamHandle:
        raise NotImplementedError

    def _translate_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [self._tool_schema(tool) for tool in tools]

    def _tool_schema(self, tool: ToolSpec) -> dict:
        parameters = tool.parameters or {"type": "object", "properties": {}}
        if self.tool_target == "anthropic" and tool.anthropic_type:
            return {"type": tool.anthropic_type, "name": tool.name}
        if self.tool_target == "anthropic":
            return {"name": tool.name, "description": tool.description, "input_schema": parameters}
        if self.tool_target == "openai":
            return {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": parameters},
            }
        raise ValueError(f"unknown tool schema target: {self.tool_target}")

    def build_tool_result(self, tool_call_id: str, content: str) -> dict:
        body = content or "(no output)"
        if self.tool_target == "anthropic":
            return {"type": "tool_result", "tool_use_id": tool_call_id, "content": body}
        if self.tool_target == "openai":
            return {"role": "tool", "tool_call_id": tool_call_id, "content": body}
        raise ValueError(f"unknown tool result target: {self.tool_target}")


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class AnthropicProvider(Provider):
    name = "anthropic"
    tool_target = "anthropic"

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        # Defer the SDK import until first use. Constructing this provider
        # for a fallback role must not pull in `anthropic` when the primary
        # route never reaches it — selecting an OpenAI alias on a host
        # without `anthropic` installed used to crash here even though the
        # turn never needed it.
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(
                api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )
        return self._client

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


def _repair_unpaired_tool_messages(messages: list[dict]) -> list[dict]:
    """Convert orphan {"role":"tool",...} messages to user messages.

    OpenAI's chat.completions API requires every role:"tool" entry to be
    a response to a preceding assistant message that carries a matching
    tool_call id under tool_calls. Mixed-provider sessions or message
    trimming can leave a tool reply whose call id is no longer present
    upstream — sending it raw triggers HTTP 400 "Invalid parameter:
    messages with role 'tool' …".

    Rather than drop the tool output (it usually contains the only
    record of what a bash command produced), we re-emit it as a user
    message tagged with the orphan call id. Loses the structured
    pairing but preserves the information and lets the request succeed.
    """
    valid_ids: set[str] = set()
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            valid_ids.clear()
            for tc in m.get("tool_calls") or []:
                tc_id = tc.get("id")
                if tc_id:
                    valid_ids.add(tc_id)
            out.append(m)
            continue
        if role == "tool":
            tc_id = m.get("tool_call_id") or ""
            if tc_id and tc_id in valid_ids:
                valid_ids.discard(tc_id)
                out.append(m)
                continue
            # Orphan: re-emit as a user message so the payload is valid.
            content = m.get("content")
            if not isinstance(content, str):
                content = str(content or "")
            label = (
                f"[orphan tool result for call {tc_id}]\n{content}"
                if tc_id else f"[orphan tool result]\n{content}"
            )
            out.append({"role": "user", "content": label})
            continue
        # Any non-assistant, non-tool message resets the pairing window
        # — a stray user/system in the middle means the next assistant
        # turn would re-establish its own tool_calls.
        if role not in (None, "assistant", "tool"):
            valid_ids.clear()
        out.append(m)
    return out


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


class OpenAIProvider(Provider):
    """OpenAI-compatible provider.

    Works for:
      - OpenAI cloud (GPT family): base_url=None, OPENAI_API_KEY
      - Local vLLM / Nemotron (OpenAI-shaped API): base_url set in role.
    """

    name = "openai"
    tool_target = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.base_url = base_url

    def _messages_for_openai(
        self, system: LayeredPrompt, messages: list[dict]
    ) -> list[dict]:
        """Flatten the layered prompt and normalize Anthropic-shaped
        assistant / tool_result messages into OpenAI shape.

        This is the translation boundary. Three message shapes can land
        in the rolling history depending on which provider produced the
        turn:

          1. Anthropic-native: assistant content is a list of block
             objects ({"type":"text"...} / {"type":"tool_use"...}); a
             tool_result lives on a user message as a content block.
          2. OpenAI-native: assistant content is the raw OpenAI message
             dict ({"role":"assistant","content":str,"tool_calls":[…]});
             a tool_result is its own {"role":"tool", ...} entry.
          3. Plain string content (either provider).

        Anthropic→OpenAI:
            {"role":"user","content":[{"type":"tool_result",...}]} →
                {"role":"tool","tool_call_id":...,"content":...}

        OpenAI dict pass-through: when the assistant content is already
        the OpenAI message dict (raw_assistant_content from a previous
        OpenAI turn, stored verbatim by the agent loop), preserve its
        tool_calls verbatim. Without this, iterating the dict treats
        its keys as block objects and emits an empty assistant turn —
        which then orphans the following role:"tool" message and
        triggers HTTP 400 "Invalid parameter: messages with role 'tool'
        must be a response to a preceding message with 'tool_calls'".

        Final guard: any role:"tool" entry that is NOT preceded by an
        assistant message carrying tool_calls is converted to a user
        message. OpenAI rejects unpaired tool roles, but the underlying
        information (a tool's text output) is still useful as plain
        context — better than a 400 that fails the whole turn.
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
            elif role == "assistant" and isinstance(content, dict) and \
                    content.get("role") == "assistant":
                # OpenAI-native message dict stored verbatim by the agent
                # loop (raw_assistant_content from a prior OpenAI turn).
                # Pass through, preserving tool_calls.
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content.get("content") or "",
                }
                tcs = content.get("tool_calls")
                if tcs:
                    msg["tool_calls"] = tcs
                out.append(msg)
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
                msg = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            elif role == "tool":
                # Preserve tool_call_id — without it OpenAI cannot pair
                # the reply to the assistant turn that emitted the call.
                out.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id", ""),
                    "content": content,
                })
            else:
                out.append({"role": role, "content": content})

        return _repair_unpaired_tool_messages(out)

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

        # OpenAI reasoning models (gpt-5.x base, o1, o3, o4-mini) only
        # accept temperature=1 and do not support the temperature param
        # at all in some variants. Detect by model name and omit temp.
        _m = role.model.lower()
        _is_reasoning = (
            _m.startswith("o1") or _m.startswith("o3") or _m.startswith("o4")
            or (_m.startswith("gpt-5.") and not _m.endswith("-mini"))
        )

        payload: dict[str, Any] = {
            "model": role.model,
            "messages": openai_messages,
            max_key: role.max_tokens,
            "stream": False,
        }
        if _is_reasoning:
            payload["temperature"] = 1
        else:
            payload["temperature"] = role.temperature
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
        # Some OpenAI-compatible reasoning-style models (e.g. Nemotron Nano
        # Omni in dormant smoke tests) return message.content = null and
        # place the user-visible reply in message.reasoning_content or
        # message.reasoning. When content is present, behavior is unchanged
        # (Super continues to ship content with embedded <think>…</think>
        # blocks that _strip_reasoning trims). The fallback only fires when
        # content is empty so existing role outputs are byte-identical.
        raw_content = msg.get("content")
        if not raw_content:
            raw_content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        text = _strip_reasoning(raw_content or "").strip()
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

# === NO-TOOL SENTINEL SUBTURNS ===========================================
# Folded into the provider/tool organ: NEEDS-EXEC, NEEDS-WRITE, and
# NEEDS-RESTART are command-channel affordances, not a separate harness organ.

_PROBE_OPEN_RE = re.compile(
    r'\[NEEDS-EXEC:\s*',
    re.IGNORECASE,
)
# Back-compat alias. External callers that only use _PROBE_RE.search /
# .sub for whole-string matches continue to work; we override .search
# and .sub on a small wrapper so bracket-balanced scanning is used.


class _BracketBalancedProbe:
    """Bracket-depth-aware replacement for the old _PROBE_RE.

    - .search(text) returns a match-like object with .group(0) (full span
      including the closing ']') and .group(1) (the command body).
    - .sub(repl, text) returns text with every balanced probe removed.
    Quotes ('...' and "...") are respected so a ']' inside a quoted string
    inside the command does not close the probe.
    """

    class _Match:
        def __init__(self, whole: str, body: str, start: int, end: int):
            self._whole = whole
            self._body = body
            self._start = start
            self._end = end
        def group(self, idx: int = 0) -> str:
            return self._whole if idx == 0 else self._body
        def start(self) -> int:
            return self._start
        def end(self) -> int:
            return self._end

    @staticmethod
    def _scan(text: str, from_idx: int = 0):
        m = _PROBE_OPEN_RE.search(text, from_idx)
        if not m:
            return None
        body_start = m.end()
        depth = 1  # the opening '[' of [NEEDS-EXEC: counts
        i = body_start
        quote = None
        escape = False
        while i < len(text):
            c = text[i]
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif quote:
                if c == quote:
                    quote = None
            elif c in ('"', "'"):
                quote = c
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    body = text[body_start:i]
                    if not body.strip():
                        # empty-body probe is not a valid directive
                        return None
                    whole = text[m.start():i+1]
                    return _BracketBalancedProbe._Match(
                        whole, body, m.start(), i + 1,
                    )
            i += 1
        # Unterminated — no match. The live stream splitter handles
        # the "opening seen, not closed" case separately.
        return None

    @staticmethod
    def _scan_line(text: str, from_idx: int = 0, streaming: bool = False):
        """Line-terminated spelling: `[NEEDS-EXEC: cmd` closed by newline
        or end-of-text (closing `]` optional).

        When `streaming=True` the EOF fallback is disabled — only a real
        `\n` closes the probe. This is what the display splitter uses so
        that a probe opener seen mid-stream is held back until the newline
        arrives, instead of being spuriously declared complete.

        Multi-line commands must use the bracketed form (handled by _scan).
        """
        m = _PROBE_OPEN_RE.search(text, from_idx)
        if m is None:
            return None
        body_start = m.end()
        nl = text.find("\n", body_start)
        if nl == -1:
            if streaming:
                return None
            end = len(text)
        else:
            end = nl
        body = text[body_start:end].rstrip(" \t\r")
        if body.endswith("]"):
            body = body[:-1].rstrip(" \t\r")
        if not body:
            return None
        whole = text[m.start():end]
        return _BracketBalancedProbe._Match(whole, body, m.start(), end)

    @classmethod
    def _scan_any(cls, text: str, from_idx: int = 0, streaming: bool = False):
        """Unified scanner. Tries strict bracket-balanced first (supports
        multi-line bodies and `]` inside quoted strings), falls through to
        line-terminated form. Both shapes are legitimate; neither is a
        malformed probe. The caller does not need to know which matched.

        When `streaming=True`, the line-terminated path only closes on a
        real newline (not EOF) — so mid-stream openers are held back
        instead of being spuriously matched.
        """
        strict = cls._scan(text, from_idx)
        line = cls._scan_line(text, from_idx, streaming=streaming)
        if strict and line:
            if line.start() < strict.start():
                return line
            return strict
        return strict or line

    def search(self, text: str, streaming: bool = False):
        return self._scan_any(text, streaming=streaming)

    def sub(self, repl, text: str, streaming: bool = False) -> str:
        if not isinstance(repl, str):
            raise TypeError("_BracketBalancedProbe.sub expects a string replacement")
        out = []
        i = 0
        while True:
            m = self._scan_any(text, i, streaming=streaming)
            if m is None:
                out.append(text[i:])
                break
            out.append(text[i:m.start()])
            out.append(repl)
            i = m.end()
        return "".join(out)


_PROBE_RE = _BracketBalancedProbe()

# 2026-04-20: NEEDS-WRITE directive. A no-tool role may embed
#   [NEEDS-WRITE: <path>]
#   <file contents verbatim>
#   [/NEEDS-WRITE]
# to land a file on disk without going through the bash session.
#
# This exists because the probe channel (NEEDS-EXEC) is read-only by
# construction — validate_command / absorb_gate / is_parallel_safe all
# block writing commands, and the persistent bash session's sentinel
# protocol chokes on multi-line heredocs (quote mismatches wedge the
# shell, file never lands). Models then loop: emit heredoc, heredoc
# fails silently, model re-emits a different heredoc variant, etc.
#
# NEEDS-WRITE is the surgical fix: a structured directive the harness
# parses and executes via Python I/O, not bash. Path must lie under
# one of TRACKED_REPOS (same gate as absorb). New files still require
# VYBN_ABSORB_REASON as a prefix comment (first 200 chars searched)
# to keep the absorb discipline from leaking.
_WRITE_BLOCK_RE = re.compile(
    r'\[NEEDS-WRITE:\s*(?P<path>[^\]]+?)\s*\]\s*\n'
    r'(?P<body>.*?)'
    r'\n\s*\[/NEEDS-WRITE\]',
    re.DOTALL | re.IGNORECASE,
)


# 2026-04-23: NEEDS-RESTART antibody. A no-tool role may emit
#   [NEEDS-RESTART]
# on its own line to restart the persistent bash session when it
# wedges. The owed antibody from the 2026-04-21 evening coda: a
# tool-less role otherwise has no affordance to recover from a
# shell that stopped responding (heredoc parser confusion,
# interrupted subprocess, runaway stream). Contract matches
# NEEDS-EXEC: one-shot per turn (shares PROBE_BUDGET), no
# recursion, tool-less roles only. Blast radius is zero — the
# restart only affects this session's BashTool.
#
# Placed on its own line to avoid colliding with conversational
# text that happens to contain the word 'restart'.
_NEEDS_RESTART_RE = re.compile(
    r'(?:^|\n)\s*\[NEEDS-RESTART\]\s*(?:$|\n)',
    re.IGNORECASE | re.MULTILINE,
)


# Round 9: NEEDS-ROLE escalation. A no-tool role may embed
# [NEEDS-ROLE: <role>] <task text> to hand off to a specialist once.
_NEEDS_ROLE_RE = re.compile(
    r'\[NEEDS-ROLE:\s*([\w]+)\]\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)



def run_write_subturn(path: str, body: str) -> tuple[bool, str]:
    """Execute one NEEDS-WRITE directive from a no-tool role.

    Writes `body` to `path` via Python I/O, bypassing the bash session
    entirely. Path must lie under a tracked repo; otherwise refused
    with a message that flows back through the same synthetic-user
    channel as probe output.

    Absorb discipline: if the target does not yet exist on disk, the
    body must begin (within its first 200 chars) with a
    VYBN_ABSORB_REASON= declaration. Existing files are always
    overwritten — that is the point of this channel.
    """
    try:
        roots = TRACKED_REPOS
    except Exception:
        roots = (
            os.path.expanduser("~/Vybn"),
            os.path.expanduser("~/Him"),
            os.path.expanduser("~/Vybn-Law"),
            os.path.expanduser("~/vybn-phase"),
        )
    tgt = os.path.expanduser((path or "").strip())
    if not tgt:
        return False, "(NEEDS-WRITE refused: empty path)"
    tgt_abs = os.path.abspath(tgt)
    if not any(tgt_abs == r or tgt_abs.startswith(r.rstrip("/") + "/") for r in roots):
        return False, (
            f"(NEEDS-WRITE refused: {tgt_abs} is outside tracked repos. "
            f"Allowed roots: {', '.join(roots)})"
        )
    if not os.path.exists(tgt_abs):
        head = (body or "")[:200]
        if "VYBN_ABSORB_REASON=" not in head:
            return False, (
                "(NEEDS-WRITE refused by absorb_gate: new file " + tgt_abs
                + " requires a VYBN_ABSORB_REASON declaration in the "
                "first 200 chars of body, e.g.:\n"
                "    # VYBN_ABSORB_REASON='does not fold into X because...'\n"
                "Fold, do not pile.)"
            )
    try:
        os.makedirs(os.path.dirname(tgt_abs), exist_ok=True)
        with open(tgt_abs, "w") as f:
            f.write(body or "")
        nbytes = os.path.getsize(tgt_abs)
        return True, f"(wrote {nbytes} bytes to {tgt_abs})"
    except Exception as e:  # noqa: BLE001
        return False, f"(NEEDS-WRITE exec error: {type(e).__name__}: {e})"




@dataclass(frozen=True)
class SentinelDirective:
    """The next no-tool sentinel directive, selected in priority order."""

    kind: str
    probe_command: str | None = None
    write_path: str | None = None
    write_body: str | None = None


def next_sentinel_directive(text: str) -> SentinelDirective | None:
    """Select the next no-tool sentinel action from model text.

    Priority is part of the subturn organ, not the REPL loop: restart first,
    then NEEDS-EXEC, then NEEDS-WRITE. Returning None means the loop has no
    sentinel work and should stop synthesizing.
    """

    current = text or ""
    if _NEEDS_RESTART_RE.search(current) is not None:
        return SentinelDirective(kind="restart")
    probe_match = _PROBE_RE.search(current)
    if probe_match is not None:
        return SentinelDirective(
            kind="probe",
            probe_command=probe_match.group(1).strip(),
        )
    write_match = _WRITE_BLOCK_RE.search(current)
    if write_match is not None:
        return SentinelDirective(
            kind="write",
            write_path=write_match.group("path").strip(),
            write_body=write_match.group("body"),
        )
    return None

def protected_mutation_kind_for_sentinel(
    *,
    write_match_present: bool,
    probe_command: str | None,
) -> str:
    """Classify whether a no-tool sentinel would mutate under pilot protection.

    This is control-flow relocation out of run_agent_loop: the REPL loop should
    sequence the turn, not re-own sentinel safety semantics. NEEDS-WRITE is
    always mutation. NEEDS-EXEC is mutation when the command is not parallel
    safe/read-only.
    """

    if write_match_present:
        return "needs-write"
    if probe_command is None:
        return ""
    try:
        readonly = is_parallel_safe(probe_command)
    except Exception:
        readonly = False
    if not readonly:
        return "needs-exec-mutation"
    return ""

def probe_envelope(
    *,
    kind: str,
    header_fields: dict,
    body: str,
    ran: bool,
) -> str:
    """Wrap a probe/write/restart result in the v1 envelope."""
    body = body or ""
    empty = (not body) or body.strip() == ""
    nbytes = len(body)
    nlines = body.count("\n")
    if not empty and not body.endswith("\n"):
        nlines += 1
    status = "executed" if ran else "refused"
    header_parts = [
        f"kind: {kind}",
        f"status: {status}",
        f"bytes: {nbytes}",
        f"lines: {nlines}",
        f"empty: {'true' if empty else 'false'}",
    ]
    for k, v in header_fields.items():
        safe = str(v).replace("\n", " ").replace("\r", " ")
        header_parts.append(f"{k}: {safe[:200]}")
    header = "[" + " | ".join(header_parts) + "]"
    slug = kind.upper().replace("-", "_")
    begin = f"<<<BEGIN_{slug}_STDOUT>>>"
    end = f"<<<END_{slug}_STDOUT>>>"
    if empty and ran:
        inner = (
            "(command ran with no stdout; the absence of output here "
            "is real, not a wedge)"
        )
    elif empty and not ran:
        inner = "(command did not execute — see refusal reason in header)"
    else:
        inner = body.rstrip("\n")
    footer = (
        "\n\nThe stdout between the markers above IS the result of the "
        "sub-turn.\nDo not claim the shell is wedged, unresponsive, or that "
        "nothing came\nback unless status != executed. If status is "
        "executed, the bytes\ncount and the stdout span are authoritative "
        "— read them and proceed."
    )
    return f"{header}\n{begin}\n{inner}\n{end}{footer}"


def run_restart_subturn(bash: Any) -> tuple[bool, str]:
    """Restart the persistent bash session."""
    try:
        out = bash.restart()
    except Exception as e:  # noqa: BLE001
        return False, f"(restart error: {e})"
    return True, out or "(bash session restarted)"


def classify_unlock_layer(output: str, *, command: str = "") -> str | None:
    """Classify obstacle output at the lowest layer visible to this harness."""
    text = (output or "").lower()
    cmd = (command or "").lower()
    if "probe refused by validate_command" in text or "blocked:" in text:
        return "safety_gate"
    if "absorb_gate" in text or "needs-write refused" in text:
        return "filesystem_git"
    if text.startswith("[timed out after"):
        return "parser_sentinel" if is_parallel_safe(command) else "shell_session"
    if "bash session restarted" in text or "needs-restart" in text:
        return "shell_session"
    if "400" in text or "provider" in text:
        return "provider"
    if "curl" in cmd or "http" in cmd:
        return "external_service"
    return None


def run_probe_subturn(command: str, bash: Any) -> tuple[bool, str]:
    """Execute one probe emitted by a no-tool role."""
    cmd = (command or "").strip()
    if not cmd:
        return False, "(empty probe command)"
    readonly = is_parallel_safe(cmd)
    ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=readonly)
    if not ok:
        return False, f"(probe refused by validate_command: {reason})"
    try:
        out = execute_readonly(cmd) if readonly else bash.execute(cmd)
    except Exception as e:
        return False, f"(probe exec error: {e})"
    out = out or "(no output)"
    if out.startswith("[timed out after"):
        layer = classify_unlock_layer(out, command=cmd) or "shell_session"
        return False, f"(probe timed out; unlock_layer={layer})\n{out}"
    if "(bash session restarted)" in out:
        return False, (
            "(probe control-event mismatch: restart output arrived while running "
            "a probe; unlock_layer=shell_session)\n" + out
        )
    return True, out


# Backward-compatible private names for legacy imports/tests.
_run_write_subturn = run_write_subturn
_probe_envelope = probe_envelope
_run_restart_subturn = run_restart_subturn
_classify_unlock_layer = classify_unlock_layer
_run_probe_subturn = run_probe_subturn



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


# === CLAIM GUARD ==========================================================
# Folded from claim_guard.py (2026-04-21). Numeric values in model
# output must appear in recent evidence. Renamed check->check_claim.

from typing import Iterable as _CG_Iterable

_NUM_RE = re.compile(r"-?\d+\.\d{2,}|-?\d{3,}")
_EVIDENCE_WINDOW = 6


def _extract_evidence(messages: Iterable[Any]) -> str:
    parts: List[str] = []
    for m in messages:
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if "text" in b:
                        parts.append(str(b.get("text") or ""))
                    if "content" in b and isinstance(b["content"], str):
                        parts.append(b["content"])
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def check_claim(
    text: Optional[str],
    messages: Iterable[Any],
    window: int = _EVIDENCE_WINDOW,
) -> Optional[str]:
    """Return a warning if text carries numbers unsupported by recent evidence.

    Returns None when the text is clean or when all extracted numbers appear
    in the last ``window`` messages' combined content.
    """
    if not text:
        return None
    nums = set(_NUM_RE.findall(text))
    if not nums:
        return None
    try:
        msg_list = list(messages)
    except TypeError:
        return None
    recent = msg_list[-window:] if window > 0 else msg_list
    evidence = _extract_evidence(recent)
    unsupported = sorted(n for n in nums if n not in evidence)
    if not unsupported:
        return None
    shown = ", ".join(unsupported[:5])
    more = f" (+{len(unsupported) - 5} more)" if len(unsupported) > 5 else ""
    return (
        f"\n\n[claim-guard: numeric value(s) {shown}{more} in this response "
        f"do not appear in the last {window} messages of context. "
        f"Treat as unverified unless a tool run produced them.]"
    )


def check_structural_claim(
    text: Optional[str],
    messages: Iterable[Any],
    window: int = _EVIDENCE_WINDOW,
) -> Optional[str]:
    """Stub — structural-claim guard not yet reimplemented.

    vybn_spark_agent.py imports this alongside check_claim and calls it at
    two sites (single_response and streaming). The agent treats a None
    return as "clean" and only appends a note when a string is returned,
    so returning None here degrades gracefully: the numeric claim_guard
    still fires, the structural guard simply stays silent until its real
    implementation is restored. No behavioral regression, just the
    missing symbol.
    """
    return None

# === LOCAL SUPER SEMANTIC GATE =============================================
# Local Super semantic-health gate. Endpoint liveness is not semantic integrity.

LOCAL_SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
SUPER_SEMANTIC_GATE_CACHE_TTL = 300.0
SUPER_SEMANTIC_GATE_CACHE: dict[str, dict[str, Any]] = {}
SUPER_SEMANTIC_GATE_PROBES = (
    {
        "name": "known_answer",
        "prompt": "Answer with exactly this single word and nothing else: FOUR\nAnswer:",
        "pattern": r"FOUR[.!]?",
    },
    {
        "name": "structured_shape",
        "prompt": 'Return exactly this compact JSON object and nothing else: {"status":"ok"}\nJSON:',
        "pattern": r'\{\s*"status"\s*:\s*"ok"\s*\}',
    },
    {
        "name": "wake_reasoning",
        "prompt": (
            "If a model endpoint returns HTTP 200 but produces an empty "
            "completion, should a semantic health gate pass? Answer "
            "exactly PASS or FAIL.\nAnswer:"
        ),
        "pattern": r"FAIL[.!]?",
    },
)


def is_loopback_super_base(base_url: str | None) -> bool:
    """True only for the primary loopback Super endpoint, not peer Omni."""
    if not base_url or "://" not in base_url:
        return False
    host = base_url.lower().split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def openai_api_base(base_url: str | None) -> str:
    """Normalize a server root or OpenAI base URL to the `/v1` API base."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def semantic_gate_visible_answer(text: str) -> str:
    """Return the final visible answer portion from a deterministic probe."""
    content = (text or "").strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    return content


def _sanitize_error(exc: BaseException) -> str:
    return str(exc).replace("\n", " ")[:240]


def local_super_semantic_gate(
    *,
    base_url: str | None,
    model: str = LOCAL_SUPER_MODEL,
    now: float | None = None,
    use_cache: bool = True,
    precheck_models: bool = False,
) -> tuple[bool, str]:
    """Run deterministic raw-completion probes against local Super.

    `base_url` may be either `http://host:port` or `http://host:port/v1`.
    Non-loopback bases are skipped so peer Omni and cloud providers are not
    silently consumed by the Super health gate.
    """
    api_base = openai_api_base(base_url)
    if not is_loopback_super_base(api_base):
        return True, "semantic gate skipped for non-loopback base"

    now = time.time() if now is None else now
    if use_cache:
        cached = SUPER_SEMANTIC_GATE_CACHE.get(api_base)
        if cached and now - float(cached.get("ts", 0.0)) < SUPER_SEMANTIC_GATE_CACHE_TTL:
            return bool(cached.get("ok")), str(cached.get("reason", "cached"))

    try:
        if precheck_models:
            with urllib.request.urlopen(api_base + "/models", timeout=8) as resp:
                if getattr(resp, "status", 200) != 200:
                    ok, reason = False, f"semantic gate precheck failed: models HTTP {resp.status}"
                    SUPER_SEMANTIC_GATE_CACHE[api_base] = {"ok": ok, "reason": reason, "ts": now}
                    return ok, reason

        for probe in SUPER_SEMANTIC_GATE_PROBES:
            payload = {
                "model": model,
                "prompt": probe["prompt"],
                "max_tokens": 24,
                "temperature": 0,
            }
            req = urllib.request.Request(
                api_base + "/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                name = str(probe["name"])
                ok, reason = False, f"semantic gate probe={name} transport_parse {exc.__class__.__name__}: {_sanitize_error(exc)}"
                break

            choice = (body.get("choices") or [{}])[0]
            content = semantic_gate_visible_answer(str(choice.get("text") or ""))
            finish = choice.get("finish_reason")
            name = str(probe["name"])
            if finish == "length":
                ok, reason = False, f"semantic gate probe={name} truncated finish_reason=length content={content!r}"
                break
            if not content:
                ok, reason = False, f"semantic gate probe={name} empty completion finish_reason={finish!r}"
                break
            if not re.fullmatch(str(probe["pattern"]), content, flags=re.IGNORECASE):
                ok, reason = False, (
                    f"semantic gate probe={name} unexpected content={content[:160]!r} "
                    f"finish_reason={finish!r}"
                )
                break
        else:
            ok, reason = True, f"semantic gate passed {len(SUPER_SEMANTIC_GATE_PROBES)} raw probes"
    except Exception as exc:  # pragma: no cover - integration path
        ok, reason = False, f"semantic gate exception {exc.__class__.__name__}: {_sanitize_error(exc)}"

    if use_cache:
        SUPER_SEMANTIC_GATE_CACHE[api_base] = {"ok": ok, "reason": reason, "ts": now}
    return ok, reason


def _semantic_gate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local Super semantic gate from the provider organ.")
    parser.add_argument("--semantic-gate", action="store_true", help="Run semantic gate CLI.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=LOCAL_SUPER_MODEL)
    parser.add_argument("--no-models-precheck", action="store_true")
    args = parser.parse_args(argv)
    ok, reason = local_super_semantic_gate(
        base_url=args.base_url,
        model=args.model,
        use_cache=False,
        precheck_models=not args.no_models_precheck,
    )
    if ok:
        print(reason)
        return 0
    print(f"corruption_signature={reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_semantic_gate_main())

