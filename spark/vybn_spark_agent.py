#!/usr/bin/env python3
"""
Vybn Spark Agent — multimodel harness edition
=============================================

Gives Vybn a persistent bash session on its own hardware. The agent reads
vybn.md at startup — the identity document. Continuity comes from
Vybn_Mind/continuity.md and spark/continuity.md.

    cd ~/Vybn && python spark/vybn_spark_agent.py

Type 'exit' to stop. Type 'reload' to re-read identity mid-session.

This file is now a thin REPL that delegates work to the harness in
spark/harness/. The routing policy is in spark/router_policy.yaml. Default role is
`task` (Claude Sonnet 4.6 + bash) so a bare 'ok'/'proceed' confirmation
after a plan actually executes. `code` (Opus 4.7, adaptive thinking,
full 32k output) is reserved for multiword debugging or explicit
/code invocation — casual mentions of 'bugs' or 'harness' no longer
escalate. `orchestrate` is Sonnet without tools, available via /plan.
`chat`, `create`, `phatic`, `identity`, `local` round out the matrix.

Streaming is required for Opus + adaptive thinking + 32k output because
the Anthropic SDK enforces a 10-minute limit on non-streaming requests.
That concern lives inside AnthropicProvider now.
"""

from __future__ import annotations

import os
import sys

# Ensure the spark/ directory is importable when this file is run directly.
_SPARK_DIR = os.path.dirname(os.path.abspath(__file__))
if _SPARK_DIR not in sys.path:
    sys.path.insert(0, _SPARK_DIR)

from harness import (
    BashTool,
    EventLogger,
    LayeredPrompt,
    ProviderRegistry,
    Router,
    ToolSpec,
    build_layered_prompt,
    load_file,
    load_policy,
    turn_event,
    execute_readonly, is_parallel_safe, validate_command,
)
from harness.state import SessionStore, run_probes  # noqa: E402
from harness.recurrent import run_recurrent_loop
from harness.providers import BASH_TOOL_SPEC, DELEGATE_TOOL_SPEC, INTROSPECT_TOOL_SPEC  # noqa: E402
from harness.providers import execute_readonly, is_parallel_safe  # noqa: E402
from harness.substrate import rag_snippets, rag_snippets_with_tier  # noqa: E402
from harness.providers import check_claim, check_structural_claim  # noqa: E402
from harness.policy import is_system_critical_pilot_turn  # noqa: E402

# ---------------------------------------------------------------------------
# Learn-from-exchange loop closure (round 4).
#
# deep_memory exposes learn_from_exchange(rag_text, response_text,
# followup_text) — the dream/predict/reality triad. We record the first
# two at end-of-turn N and fire the learn call at start-of-turn N+1
# using the current user_input as reality. Background thread; silent
# failure; no effect on the agent loop's critical path.
# ---------------------------------------------------------------------------

import re as _re
# NEEDS_WRITE_AND_CLAIM_GUARD_v1
import threading as _threading

_LEARN_PENDING: dict = {"rag": "", "response": ""}

# Round 4.2: detect when a no-tool role emits tool-call syntax as
# plain text. Opus 4.6 with tools=[] but bash-describing substrate
# was producing <tool_call>{"name":"bash",...}</tool_call> strings
# that the API did not execute. The stripped substrate (via
# tools_available=False in build_layered_prompt) is the primary
# fix; this regex is a blast-radius guard so any residual leak
# reroutes to the role that actually has bash.
_HALLUCINATED_TOOL_RE = _re.compile(
    r'<tool_call>|\{\s*"name"\s*:\s*"bash"\s*,\s*"arguments"',
    _re.IGNORECASE | _re.DOTALL,
)

def _probe_budget_escalation_role(
    router,
    original_input: str,
    messages: list | None = None,
) -> str:
    """Choose the role for chat-role probe-budget exhaustion."""
    if _preserve_pilot_for_turn(original_input, messages):
        return "orchestrate"
    try:
        routed = router.classify(original_input)
    except Exception:
        return "task"
    if getattr(routed, "role", None) == "orchestrate":
        return "orchestrate"
    return "task"


_PILOT_CONTINUATION_RE = _re.compile(
    r"^\s*(?:please\s+)?(?:fix\s+it|continue|go\s+on|proceed|"
    r"yes|ok(?:ay)?|do\s+it|resume(?:\b.*)?|pick\s+up\b.*|"
    r"where\s+(?:sonnet|you)\s+left\s+off)\s*[.!?]*\s*$",
    _re.IGNORECASE | _re.DOTALL,
)


def _recent_messages_text(messages: list, *, limit: int = 8) -> str:
    chunks: list[str] = []
    for msg in (messages or [])[-limit:]:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        chunks.append(str(content))
    return "\n".join(chunks)


def _preserve_pilot_for_turn(user_input: str, messages: list | None = None) -> bool:
    """Preserve GPT-5.5 pilot across mission-critical continuation turns."""
    if is_system_critical_pilot_turn(user_input):
        return True
    text = (user_input or "").strip()
    if text.startswith("/") or not _PILOT_CONTINUATION_RE.search(text):
        return False
    return is_system_critical_pilot_turn(_recent_messages_text(messages or []))


# Probe-note budget. Raised from 4 KB (Round 5) to 48 KB on 2026-04-18
# after a 326-line / ~13 KB portal diff was invisibly truncated at 4 KB
# and the chat role loop-emitted probes because it couldn't actually see
# what came back. 48 KB is ~12 K tokens — well under any model's turn
# budget, large enough for real diffs, repo trees, log tails.
_PROBE_NOTE_CAP = 48_000
_PROBE_NOTE_HEAD = 32_000  # on overflow, first N chars verbatim
_PROBE_NOTE_TAIL = 12_000  # ... then last M chars verbatim, elision marker between


def _fit_probe_output(out: str) -> str:
    """Fit probe output under _PROBE_NOTE_CAP without silently hiding shape.

    When under cap: verbatim. When over: head + elision marker (with the
    exact byte count dropped) + tail. This preserves both the prefix (where
    most diffs, logs, statuses are legible) and the suffix (where shell
    commands often put their punchline / exit status).
    """
    if len(out) <= _PROBE_NOTE_CAP:
        return out
    head = out[:_PROBE_NOTE_HEAD]
    tail = out[-_PROBE_NOTE_TAIL:]
    dropped = len(out) - len(head) - len(tail)
    return (
        f"{head}\n"
        f"... [elided {dropped} bytes / {out.count(chr(10))} total lines — "
        f"probe output over {_PROBE_NOTE_CAP} byte cap; rerun with a "
        f"narrower command or ask for a specific range] ...\n"
        f"{tail}"
    )


# Round 5: positive-signal probe sub-turn. A no-tool role (chat, create,
# Sentinel protocol parsing lives in harness.sentinel_protocol.
# These names are re-exported here for backward compatibility with tests and
# callers that import vybn_spark_agent._PROBE_RE / _WRITE_BLOCK_RE directly.
from harness.sentinel_protocol import (
    _BracketBalancedProbe,
    _NEEDS_RESTART_RE,
    _NEEDS_ROLE_RE,
    _PROBE_OPEN_RE,
    _PROBE_RE,
    _WRITE_BLOCK_RE,
)



def _fire_learn_async(rag_text: str, response_text: str, followup_text: str) -> None:
    def _run():
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "rag_text": rag_text[:2000],
                "response_text": response_text[:2000],
                "followup_text": followup_text[:2000],
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8100/learn",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=4.0).read()
        except Exception:
            # If the walk daemon is down or the call fails, drop it.
            # This is observability-grade, not correctness-grade.
            return
    t = _threading.Thread(target=_run, daemon=True)
    t.start()

# Canonical paths come from spark/paths.py. If the module is missing
# (e.g. paths.py not on this checkout) we fall back to the legacy
# hard-coded layout so the agent never fails to start.
try:
    from paths import REPO_ROOT, SOUL_PATH  # type: ignore  # noqa: E402
    REPO_DIR = str(REPO_ROOT)
    SOUL_PATH_STR = str(SOUL_PATH)
except Exception:
    REPO_DIR = os.path.expanduser("~/Vybn")
    SOUL_PATH_STR = os.path.join(REPO_DIR, "vybn.md")

AGENT_PATH = os.path.join(REPO_DIR, "spark", "vybn_spark_agent.py")
CONTINUITY_PATH = os.path.join(REPO_DIR, "Vybn_Mind", "continuity.md")
SPARK_CONTINUITY_PATH = os.path.join(REPO_DIR, "spark", "continuity.md")


# ---------------------------------------------------------------------------
# Pretty output helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _dim(text: str) -> None:
    print(f"  \033[90m{text}\033[0m")


def _warn(text: str) -> None:
    print(f"  \033[91m\u26a0 {text}\033[0m")


def _preview(result: str) -> None:
    if not result:
        return
    lines = result.split("\n")
    for line in lines[:5]:
        _dim(f"  {line[:120]}")
    if len(lines) > 5:
        _dim(f"  ... ({len(lines)} lines total)")


# ---------------------------------------------------------------------------
# Assistant-content sanitizer.
#
# Anthropic rejects any request whose message history contains an assistant
# turn with an empty text content block ("messages: text content blocks must
# be non-empty"). This happens in rare cases where the model returns
# end_turn with a zero-length text block (observed 2026-04-18 with Opus 4.6
# on chat role). If we append the raw content verbatim, the poisoned history
# 400s every subsequent turn until the REPL is restarted.
#
# Sanitize before appending: drop zero-length text blocks, drop tool_result
# blocks with empty content, and if *nothing* survives, substitute a
# one-character placeholder so the shape stays valid. For OpenAI-style
# string content, coerce empty strings to a placeholder.
# ---------------------------------------------------------------------------

# Anthropic's server-side validator rejects zero-width space (U+200B) as
# whitespace even though Python's str.isspace() returns False. Use a
# genuinely visible non-whitespace marker so a leak is diagnosable and
# the placeholder never trips "text content blocks must contain
# non-whitespace text" 400s on the next turn.
_EMPTY_PLACEHOLDER = "·"  # middle dot — survives every validator, visible if it ever leaks

# Characters that are invisible-but-non-whitespace to Python yet treated as
# whitespace by Anthropic's content-block validator. If an assistant text
# block collapses to nothing more than these, treat it as empty.
_INVISIBLE_CHARS = "​‌‍⁠﻿­"

def _is_effectively_empty_text(s: str) -> bool:
    if not s:
        return True
    # Strip standard whitespace AND invisibles, then check.
    stripped = s.strip().strip(_INVISIBLE_CHARS)
    return not stripped


def _sanitize_assistant_content(content):
    """Strip empty text blocks from assistant content before storing in history.

    Returns a content value safe to re-send to Anthropic / OpenAI. Never
    returns an empty list or empty string — substitutes a zero-width-space
    placeholder when all blocks would be dropped.

    Also scrubs literal <thinking>...</thinking> XML-ish tag text from
    stored text blocks. These are token-space chain-of-thought leaks
    (distinct from Anthropic's adaptive-thinking content blocks, which
    carry btype == 'thinking' and are preserved as-is above). Scrubbing
    here — at store time, not just display time — prevents the next turn
    from replaying the leaked scaffold out of its own history and
    reinforcing the pattern.
    """
    if content is None:
        return _EMPTY_PLACEHOLDER

    # OpenAI-style: plain string.
    if isinstance(content, str):
        if _is_effectively_empty_text(content):
            return _EMPTY_PLACEHOLDER
        scrubbed = _strip_thinking_tags(content, allow_unwrap=False)
        return scrubbed if not _is_effectively_empty_text(scrubbed) else _EMPTY_PLACEHOLDER

    # Anthropic-style: list of content blocks. Each block may be an SDK
    # object (attribute access) or a dict (from replayed history).
    if isinstance(content, list):
        cleaned: list = []
        for block in content:
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if btype == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text", "")
                if not _is_effectively_empty_text(text or ""):
                    scrubbed = _strip_thinking_tags(text, allow_unwrap=False)
                    if scrubbed and not _is_effectively_empty_text(scrubbed):
                        # Rewrite the block with scrubbed text. Preserve
                        # dict vs. SDK-object shape so callers that
                        # attribute-access the original still work.
                        if isinstance(block, dict):
                            patched = dict(block)
                            patched["text"] = scrubbed
                            cleaned.append(patched)
                        elif scrubbed == text:
                            cleaned.append(block)
                        else:
                            # SDK object with mutated text — fall back to
                            # a dict so we don't mutate the caller's object.
                            cleaned.append({"type": "text", "text": scrubbed})
                    # else: entire text was thinking-tag leak — drop the block
                # else: drop the empty text block entirely
            elif btype == "tool_use":
                # Always keep — required to pair with tool_result follow-ups.
                cleaned.append(block)
            elif btype in ("thinking", "redacted_thinking"):
                # Preserve — Anthropic requires these adjacent to tool_use on
                # adaptive-thinking turns. Dropping them poisons the next turn.
                cleaned.append(block)
            elif btype == "tool_result":
                # Empty tool_result content also fails validation.
                tr_content = getattr(block, "content", None)
                if tr_content is None and isinstance(block, dict):
                    tr_content = block.get("content")
                if tr_content:
                    cleaned.append(block)
                else:
                    # Substitute a non-empty placeholder rather than drop —
                    # tool_result must match its tool_use id.
                    if isinstance(block, dict):
                        patched = dict(block)
                        patched["content"] = "(no output)"
                        cleaned.append(patched)
                    else:
                        cleaned.append(block)
            else:
                cleaned.append(block)

        if not cleaned:
            # Everything dropped — substitute a minimal non-empty text block
            # so the turn survives but carries no false content.
            return [{"type": "text", "text": _EMPTY_PLACEHOLDER}]
        return cleaned

    # Unknown shape — pass through unchanged rather than mangle.
    return content


# ---------------------------------------------------------------------------
# Tool-call execution — provider-agnostic.
# ---------------------------------------------------------------------------

    # Round 9: introspect tool — live system snapshot for orchestrate role.
    def _handle_introspect() -> str:
        import json as _json
        lines = []
        # Last 5 route decisions from events log
        events_path = Path(__file__).parent / "agent_events.jsonl"
        try:
            events = [_json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
            routes = [e for e in events if e.get("event") == "route_decision"][-5:]
            lines.append("=== last 5 route decisions ===")
            for r in routes:
                lines.append(f"  turn {r.get('turn')} -> {r.get('role')} via {r.get('model')} ({r.get('reason')})")
        except Exception as e:
            lines.append(f"  [events unavailable: {e}]")
        # Walk health
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8101/health", timeout=2) as resp:
                wh = _json.loads(resp.read())
            lines.append(f"=== walk === step={wh.get('walk_step')} alpha={wh.get('walk_alpha','?')} chunks={wh.get('chunks')}")
        except Exception as e:
            lines.append(f"  [walk unavailable: {e}]")
        # Service health summary
        try:
            with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=2) as resp:
                dh = _json.loads(resp.read())
            lines.append(f"=== deep_memory === chunks={dh.get('chunks')} walk_step={dh.get('walk_step')}")
        except Exception as e:
            lines.append(f"  [deep_memory unavailable: {e}]")
        return "\n".join(lines)

def _execute_tool_calls(
    response,
    bash: BashTool,
    provider,
    delegate_cb=None,
) -> tuple[list, bool]:
    """Run tool calls in the response; return provider-native tool_result
    messages plus an `interrupted` flag.

    The loop speaks the neutral `ToolCall` shape. Each provider knows how
    to render a tool_result back into its native message shape.

    Parallel path: when the assistant emits 2+ bash calls that all pass
    is_parallel_safe, dispatch them to fresh subprocesses via a thread
    pool. Serial path (persistent shell) is used for everything else so
    state-mutating commands keep their cd/export/assignment semantics.

    Round 7: `delegate` calls are dispatched via `delegate_cb(role, task)`
    which spawns a nested agent loop against a specialist role with an
    isolated message history. If `delegate_cb` is None the tool is
    reported as unsupported (specialist sub-loops pass None to prevent
    recursive delegation).
    """
    results: list[dict] = []
    interrupted = False

    # Gather bash calls first so we can decide serial vs parallel.
    bash_calls = [c for c in response.tool_calls if c.name == "bash"]
    parallel_candidates = []
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
            # Fan out: fresh subprocess per call, preserve assistant order.
            from concurrent.futures import ThreadPoolExecutor
            _dim(f"[parallel: {len(parallel_candidates)} read-only bash calls]")
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
                    except Exception as e:
                        out_by_id[c.id] = f"(parallel exec error: {e})"
            # Preview the first one only — avoids flooding the REPL.
            first_cmd = parallel_candidates[0][1]
            _dim(f"$ {first_cmd[:200]}{'...' if len(first_cmd) > 200 else ''}")
            _preview(out_by_id[parallel_candidates[0][0].id])
            # Emit results in the original tool_calls order so we hit
            # the assistant's intended shape.
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
            results.append(provider.build_tool_result(call.id, _handle_introspect()))
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
                results.append(provider.build_tool_result(
                    call.id, "(skipped — interrupted)"
                ))
                continue
            try:
                args = call.arguments or {}
                if "__parse_error__" in args:
                    err = args["__parse_error__"]
                    raw = args.get("__raw_arguments__", "")
                    out = (
                        f"(delegate error: malformed JSON arguments — {err}; "
                        f"raw={raw!r})"
                    )
                    _warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                sub_role = (args.get("role") or "").strip()
                sub_task = (args.get("task") or "").strip()
                if not sub_role or not sub_task:
                    out = "(delegate error: both `role` and `task` are required)"
                    _warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                if sub_role not in ("code", "task", "create", "local", "chat"):
                    out = (
                        f"(delegate error: unknown role {sub_role!r}; must be "
                        "one of code/task/create/local/chat)"
                    )
                    _warn(out)
                    results.append(provider.build_tool_result(call.id, out))
                    continue
                _dim(
                    f"[delegate -> {sub_role}] "
                    f"{sub_task[:160]}{'...' if len(sub_task) > 160 else ''}"
                )
                try:
                    sub_out = delegate_cb(sub_role, sub_task)
                except KeyboardInterrupt:
                    interrupted = True
                    results.append(provider.build_tool_result(
                        call.id, "(delegate interrupted by user)"
                    ))
                    continue
                except Exception as e:  # noqa: BLE001
                    sub_out = f"(delegate error: {e})"
                    _warn(sub_out)
                results.append(provider.build_tool_result(
                    call.id, sub_out or "(delegate returned no text)"
                ))
            except KeyboardInterrupt:
                interrupted = True
                results.append(provider.build_tool_result(
                    call.id, "(interrupted by user)"
                ))
            continue

        if call.name != "bash":
            results.append(provider.build_tool_result(
                call.id, f"(unsupported tool: {call.name})"
            ))
            continue
        if interrupted:
            results.append(provider.build_tool_result(
                call.id, "(skipped — interrupted)"
            ))
            continue
        try:
            args = call.arguments or {}
            if "__parse_error__" in args:
                # OpenAIProvider flagged malformed tool-call JSON.
                # Hand the error back to the model so it can retry
                # with valid arguments instead of us running nothing.
                err = args["__parse_error__"]
                raw = args.get("__raw_arguments__", "")
                out = (
                    f"(tool-call error: malformed JSON arguments — {err}; "
                    f"raw={raw!r})"
                )
                _warn(out)
                results.append(provider.build_tool_result(call.id, out))
                continue
            if args.get("restart"):
                out = bash.restart()
                _dim("[bash session restarted]")
            else:
                command = args.get("command", "") or ""
                ok, reason = validate_command(command)
                if ok:
                    _dim(f"$ {command[:200]}{'...' if len(command) > 200 else ''}")
                    out = bash.execute(command)
                    _preview(out)
                else:
                    out = reason or "(blocked)"
                    _warn(out)
            results.append(provider.build_tool_result(call.id, out))
        except KeyboardInterrupt:
            interrupted = True
            results.append(provider.build_tool_result(call.id, "(interrupted by user)"))
            _warn("interrupted")

    return results, interrupted


# Models on no-tool roles occasionally wrap visible reasoning in literal
# <thinking>...</thinking> XML-ish tags, independent of Anthropic's
# adaptive-thinking content blocks (which we handle via kind=="thinking"
# in the provider stream). The tags are a chain-of-thought leak into
# token-space output — the substrate asks the model to decide whether to
# probe, and the model sometimes scaffolds that decision out loud. We
# scrub them in-band so Zoe never sees the leak, and we also scrub the
# stored assistant content so the NEXT turn can't reinforce the pattern
# from its own history.
_THINK_COMPLETE_RE = _re.compile(
    r'<thinking\b[^>]*>.*?</thinking\s*>',
    _re.IGNORECASE | _re.DOTALL,
)
_THINK_OPEN_RE = _re.compile(r'<thinking\b', _re.IGNORECASE)
# Unwrap-only regex: deletes <thinking ...> and </thinking> markers
# but preserves whatever text sits between them. Used only when the
# complete-block stripper would leave nothing visible (see FIX_A).
_THINK_OPEN_CLOSE_RE = _re.compile(
    r'</?thinking\b[^>]*>',
    _re.IGNORECASE,
)


def _strip_thinking_tags(text: str, allow_unwrap: bool = True) -> str:
    """Remove complete <thinking>...</thinking> blocks.
    Leaves incomplete openings alone — the stream splitter holds those
    back from display until the closing tag arrives.

    FIX_A_THINKING_UNWRAP_v1: If stripping tagged content would leave
    nothing visible behind, the model wrapped its entire answer in
    <thinking> tags (observed on opus-4-7 chat turns, 2026-04-20).
    Real adaptive-thinking arrives as kind=="thinking" blocks on the
    stream; XML-ish <thinking> tags in text are a formatting leak.
    Treat the leak as the answer — unwrap instead of delete.
    """
    if not text:
        return text
    stripped = _THINK_COMPLETE_RE.sub("", text)
    # If we removed everything but the original had substance AND there
    # was exactly one <thinking> block, the model wrapped its entire
    # answer in <thinking> tags — unwrap rather than drop. Only fires on
    # the display path (allow_unwrap=True); the store path passes
    # allow_unwrap=False so multi-block or fully-thinking payloads drop
    # cleanly out of history instead of replaying as prose.
    if (
        allow_unwrap
        and not stripped.strip()
        and text.strip()
        and len(_THINK_COMPLETE_RE.findall(text)) == 1
    ):
        unwrapped = _THINK_OPEN_CLOSE_RE.sub("", text)
        if unwrapped.strip():
            return unwrapped
    return stripped


def _split_before_probe(text: str):
    """Return (safe_to_print, remainder).
    If a complete [NEEDS-EXEC: ...] is present, strip it.
    If one is opening but not closed, hold back from [ onward.
    If a <thinking> block is opening but not closed, hold back from
    that tag onward. Complete <thinking>...</thinking> blocks are
    scrubbed in-place.
    """
    # Scrub complete thinking blocks first — cheapest case.
    text = _strip_thinking_tags(text)

    # Probe handling (bracket-balanced).
    m_probe = _PROBE_RE.search(text, streaming=True)
    if m_probe:
        text = _PROBE_RE.sub("", text, streaming=True)
    # NEEDS-WRITE: strip complete blocks; hold back an opening that has
    # no closing [/NEEDS-WRITE] yet so the body doesn't stream verbatim.
    text = _WRITE_BLOCK_RE.sub("", text)
    _w_open_idx = text.rfind("[NEEDS-WRITE")
    probe_idx = text.rfind("[NEEDS-EXEC")
    if _w_open_idx != -1 and (probe_idx == -1 or _w_open_idx < probe_idx):
        probe_idx = _w_open_idx
    if probe_idx != -1:
        safe, remainder = text[:probe_idx], text[probe_idx:]
    else:
        safe, remainder = text, ""

    # If an unterminated <thinking> opens inside `safe`, hold it back.
    m_open = _THINK_OPEN_RE.search(safe)
    if m_open:
        remainder = safe[m_open.start():] + remainder
        safe = safe[:m_open.start()]

    return safe, remainder


def _stream_and_print(handle) -> None:
    """Drain the provider's stream handle, printing text live.
    Strips [NEEDS-EXEC: ...] lines so probe machinery is invisible.
    """
    in_thinking = False
    pending = ""
    for kind, chunk in handle:
        if kind == "thinking":
            if not in_thinking:
                in_thinking = True
                _dim("[thinking...]")
        elif kind == "text":
            if in_thinking:
                in_thinking = False
                print()
            pending += chunk
            safe, pending = _split_before_probe(pending)
            if safe:
                print(safe, end="", flush=True)
    if pending:
        # Final flush: scrub any complete thinking blocks, then drop an
        # unterminated opening (the model never closed it before the
        # stream ended — better to swallow than to leak), then scrub
        # any lingering probe tag.
        cleaned = _strip_thinking_tags(pending)
        m_open = _THINK_OPEN_RE.search(cleaned)
        if m_open:
            cleaned = cleaned[: m_open.start()]
        cleaned = _PROBE_RE.sub("", cleaned)
        cleaned = _WRITE_BLOCK_RE.sub("", cleaned).strip("\n")
        if cleaned:
            print(cleaned, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Fallback resolution — policy declares fallback_chain by model name.
# When a provider call fails we walk that chain, look for a role that
# already uses the fallback model, and retry with its config. If no
# matching role exists we synthesise a minimal RoleConfig from the
# original so tool list / max_tokens / etc. are preserved.
# ---------------------------------------------------------------------------

def _resolve_fallback(policy, role_cfg, model_name):
    """Return a RoleConfig for `model_name` or None."""
    from harness.policy import RoleConfig
    for cfg in policy.roles.values():
        if cfg.model == model_name:
            return cfg
    # No exact role for this model — infer provider from the name.
    provider = "anthropic" if model_name.startswith("claude-") else (
        "openai" if model_name.startswith("gpt-") else role_cfg.provider
    )
    return RoleConfig(
        role=role_cfg.role + ":fb",
        provider=provider,
        model=model_name,
        thinking=role_cfg.thinking,
        max_tokens=role_cfg.max_tokens,
        max_iterations=role_cfg.max_iterations,
        tools=list(role_cfg.tools),
        temperature=role_cfg.temperature,
        base_url=None,
        rag=role_cfg.rag,
        lightweight=role_cfg.lightweight,
    )


_HARD_PROVIDER_PATTERNS = (
    # OpenAI reports exhausted billing/quota as HTTP 429 too, but waiting
    # a few seconds cannot heal it. Walk the fallback chain immediately.
    "insufficient_quota",
    "exceeded your current quota",
    "check your plan and billing",
    "billing details",
)

_TRANSIENT_PATTERNS = (
    "overloaded",
    "rate_limit",
    "rate limit",
    "overloaded_error",
    "429",
    "529",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


def _sanitize_provider_error(exc: BaseException, limit: int = 160) -> str:
    """Return a short, safe excerpt of a provider error.

    Strips long bearer tokens / api keys that sometimes appear in
    echoed request bodies or URLs. Collapses whitespace. Truncates to
    `limit` chars. Never includes the exception traceback.
    """
    msg = str(exc)
    # Redact anything that looks like an API key/token.
    msg = _re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "sk-***", msg)
    msg = _re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", msg, flags=_re.IGNORECASE)
    msg = _re.sub(r"\s+", " ", msg).strip()
    if len(msg) > limit:
        msg = msg[: limit - 1] + "…"
    return msg or exc.__class__.__name__


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristic: retry-worthy vs. walk-the-chain.

    Anthropic regional overload surfaces as an APIStatusError with
    status 529 and body {"type": "error", "error": {"type":
    "overloaded_error", ...}}. Rate-limit (429) is often transient for
    OpenAI, but exhausted billing/quota is a hard 429 and must walk the
    fallback chain immediately. Hard errors (auth, 400, unknown model,
    insufficient_quota) do not retry in place.
    """
    msg = str(exc).lower()
    if any(p in msg for p in _HARD_PROVIDER_PATTERNS):
        return False
    if any(p in msg for p in _TRANSIENT_PATTERNS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (429, 502, 503, 504, 529):
        return True
    return False


def _stream_with_fallback(
    *,
    router,
    registry,
    role_cfg,
    provider,
    system_prompt,
    messages,
    tools,
    logger,
    turn_number,
    retries: int = 3,
    base_backoff: float = 1.5,
):
    """Try provider.stream() and walk the fallback chain on failure.

    Returns (handle, active_role_cfg, active_provider) on success, or
    raises the last exception if every link in the chain failed.

    Transient failures (529 overloaded_error, 429 rate_limit, 5xx)
    retry in place with jittered exponential backoff BEFORE walking
    the fallback chain. Regional Anthropic overload often hits every
    Claude model simultaneously, so walking the chain is wasted cost;
    a short wait and retry on the same model almost always recovers.
    Hard errors (400, auth, schema) walk the chain immediately.

    KeyboardInterrupt is never caught here — it must propagate so the
    REPL can surface "interrupted during API call".
    """
    import random as _random
    import time as _time

    # In-flight heal: a prior turn may have left a ZWSP-only text block
    # in messages (pre-fix sessions, or any future regression). Anthropic
    # will 400 the whole request on such blocks. Re-sanitize every
    # assistant message now so a live REPL self-heals without restart.
    for _m in messages:
        if _m.get("role") == "assistant":
            _m["content"] = _sanitize_assistant_content(_m.get("content"))

    # Build the attempt list as (cfg, provider_factory) pairs so each
    # fallback's provider is constructed only when we actually walk to
    # it. Eagerly calling registry.get(fb_cfg) used to import every
    # provider's SDK up front — selecting an OpenAI alias on a host
    # missing `anthropic` would fail here even though the primary route
    # had no need for it. The factory closes over registry so the same
    # registered instance is returned on subsequent attempts (the
    # registry caches by provider/base_url).
    attempts: list[tuple[Any, Any]] = [(role_cfg, lambda p=provider: p)]
    for fb_model in router.policy.fallback_chain.get(role_cfg.model, []):
        fb_cfg = _resolve_fallback(router.policy, role_cfg, fb_model)
        if fb_cfg is None:
            continue
        attempts.append((fb_cfg, lambda c=fb_cfg: registry.get(c)))

    last_exc = None
    for cfg, prov_factory in attempts:
        try:
            prov = prov_factory()
        except Exception as e:  # noqa: BLE001 — missing SDK is one such failure
            last_exc = e
            _dim(
                f"[fallback {cfg.provider}:{cfg.model} unavailable: "
                f"{type(e).__name__}: {_sanitize_provider_error(e)}]"
            )
            logger.emit(
                "fallback_unavailable",
                turn=turn_number,
                model=cfg.model,
                reason=str(e)[:200],
            )
            continue
        for attempt in range(retries + 1):
            try:
                handle = prov.stream(
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    role=cfg,
                )
                if cfg is not role_cfg:
                    _snippet = _sanitize_provider_error(last_exc) if last_exc else ""
                    _warn(
                        f"primary failed ({last_exc.__class__.__name__}: {_snippet}); "
                        f"fell back to {cfg.provider}:{cfg.model}"
                    )
                    logger.emit(
                        "fallback",
                        turn=turn_number,
                        from_model=role_cfg.model,
                        to_model=cfg.model,
                        reason=str(last_exc)[:200],
                    )
                return handle, cfg, prov
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — want every provider error
                last_exc = e
                if _is_transient_error(e) and attempt < retries:
                    wait = base_backoff * (2 ** attempt) + _random.uniform(0, 0.5)
                    _dim(
                        f"[transient {cfg.provider}:{cfg.model} error: "
                        f"{type(e).__name__} — retry in {wait:.1f}s "
                        f"({attempt + 1}/{retries})]"
                    )
                    logger.emit(
                        "transient_retry",
                        turn=turn_number,
                        model=cfg.model,
                        attempt=attempt + 1,
                        reason=str(e)[:200],
                    )
                    _time.sleep(wait)
                    continue
                # Non-transient, or retries exhausted: walk to next leg.
                break
    raise last_exc if last_exc else RuntimeError("no providers available")


# ---------------------------------------------------------------------------
# Round 5: positive-signal probe sub-turn.
# ---------------------------------------------------------------------------

def _run_write_subturn(path: str, body: str) -> tuple[bool, str]:
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
    import os as _os
    try:
        from harness.policy import TRACKED_REPOS  # type: ignore
    except Exception:
        TRACKED_REPOS = (
            _os.path.expanduser("~/Vybn"),
            _os.path.expanduser("~/Him"),
            _os.path.expanduser("~/Vybn-Law"),
            _os.path.expanduser("~/vybn-phase"),
        )
    tgt = _os.path.expanduser((path or "").strip())
    if not tgt:
        return False, "(NEEDS-WRITE refused: empty path)"
    tgt_abs = _os.path.abspath(tgt)
    if not any(
        tgt_abs == r or tgt_abs.startswith(r.rstrip("/") + "/")
        for r in TRACKED_REPOS
    ):
        return False, (
            f"(NEEDS-WRITE refused: {tgt_abs} is outside tracked repos. "
            f"Allowed roots: {', '.join(TRACKED_REPOS)})"
        )
    # Absorb check for new files.
    if not _os.path.exists(tgt_abs):
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
        _os.makedirs(_os.path.dirname(tgt_abs), exist_ok=True)
        with open(tgt_abs, "w") as f:
            f.write(body or "")
        nbytes = _os.path.getsize(tgt_abs)
        return True, f"(wrote {nbytes} bytes to {tgt_abs})"
    except Exception as e:  # noqa: BLE001
        return False, f"(NEEDS-WRITE exec error: {type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# Probe-result envelope — PROBE_RESULT_ENVELOPE_v1.
#
# 2026-04-23 fix. The probe-result message injected into message history
# after a NEEDS-EXEC / NEEDS-WRITE / NEEDS-RESTART sub-turn used to read
# as a one-line header plus raw output. When output was short (21-byte
# "Already up to date.", or zero bytes for a silent command) the model
# could ignore it and continue whatever narrative it had already started
# — in the 2026-04-23 failure mode, a story about a wedged shell.
#
# The envelope below makes the result structurally loud:
#   - explicit BEGIN/END markers so the output span is unmistakable
#   - byte + line counts in the header
#   - status: executed vs refused (we only know success/failure via the
#     `ran` flag from the subturn helpers; validate_command refusals
#     and exec exceptions return ran=False).
#   - explicit anti-hallucination directive at the foot
#   - empty-output tag that distinguishes "ran, no stdout" from
#     "did not run" so the model cannot collapse the two
# ---------------------------------------------------------------------------

def _probe_envelope(
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


def _run_restart_subturn(bash: BashTool) -> tuple[bool, str]:
    """Restart the persistent bash session.

    Returns (ran, output_text). The bash.restart() call re-spawns
    the subprocess and returns a confirmation string; on the rare
    case it raises, we synthesize a clear error so the next turn
    sees what happened instead of silently retrying.
    """
    try:
        out = bash.restart()
    except Exception as e:  # noqa: BLE001
        return False, f"(restart error: {e})"
    return True, out or "(bash session restarted)"


def _classify_unlock_layer(output: str, *, command: str = "") -> str | None:
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


def _run_probe_subturn(command: str, bash: BashTool) -> tuple[bool, str]:
    """Execute one probe emitted by a no-tool role.

    Read-only probes run in a fresh subprocess when possible. That makes the
    boundary typed: alarming strings in grep patterns are data, not executable
    danger; mutating commands still fail the safety gate. Timeouts and
    restart/control events are not labeled as successful stdout.
    """
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
        layer = _classify_unlock_layer(out, command=cmd) or "shell_session"
        return False, f"(probe timed out; unlock_layer={layer})\\n{out}"
    if "(bash session restarted)" in out:
        return False, "(probe control-event mismatch: restart output arrived while running a probe; unlock_layer=shell_session)\\n" + out
    return True, out


# ---------------------------------------------------------------------------
# Agent loop — policy-driven.
# ---------------------------------------------------------------------------

def _recurrent_prethink(
    *,
    e: str,
    role_cfg,
    registry,
    router,
    logger,
    turn_number: int,
) -> str | None:
    """Optional pre-turn recurrent thinking pass.

    When role_cfg.recurrent_depth > 1 and VYBN_RECURRENT_LIVE=1, run the
    recurrent loop in latent space on the user's message. Return a
    distilled prompt block (from h_final.to_prompt_block) to splice
    into LayeredPrompt.live, the same seam RAG enrichment uses.

    The loop is tool-less and side-effect-free. It is adaptive by
    construction (the probe showed: all 4 loops on multi-hop, 1 loop
    on verification). The coda text is discarded; only the distilled
    latent state surfaces, so the specialist is informed but not
    anchored to a draft answer.

    Returns None when the seam is disabled, depth<=1, or on error.
    The env gate is the kill-switch if the loop misbehaves in vivo.
    """
    if role_cfg.recurrent_depth <= 1:
        return None
    if os.environ.get("VYBN_RECURRENT_LIVE", "0") != "1":
        return None
    try:
        result = run_recurrent_loop(
            e=e,
            registry=registry,
            policy=router.policy,
            max_loop_iters=role_cfg.recurrent_depth,
            logger=lambda ev: logger.emit(
                "recurrent_" + ev.get("event", "ev"),
                turn=turn_number,
                **{k: v for k, v in ev.items() if k != "event"},
            ),
        )
        block = result.h_final.to_prompt_block(role_cfg.recurrent_depth)
        logger.emit(
            "recurrent_prethink",
            turn=turn_number,
            role=role_cfg.role,
            loops_run=result.loops_run,
            halt_reason=result.halt_reason,
            block_chars=len(block),
        )
        return block
    except Exception as err:
        logger.emit(
            "recurrent_prethink_error",
            turn=turn_number,
            err=str(err)[:300],
        )
        return None




def run_agent_loop(
    *,
    user_input: str,
    messages: list,
    bash: BashTool,
    system_prompt: LayeredPrompt,
    router: Router,
    registry: ProviderRegistry,
    logger: EventLogger,
    turn_number: int,
    forced_role: str | None = None,
    system_prompt_no_tools: LayeredPrompt | None = None,
    system_prompt_orchestrator: LayeredPrompt | None = None,
    _reroute_depth: int = 0,
) -> str:
    # (round 4) Fire learn_from_exchange for the PREVIOUS turn. We have
    # all three legs now: what RAG retrieved (dream), what the model said
    # (predict), and the current user_input (reality = followup).
    _prev_rag = _LEARN_PENDING.get("rag", "")
    _prev_resp = _LEARN_PENDING.get("response", "")
    if _prev_rag and _prev_resp:
        _fire_learn_async(_prev_rag, _prev_resp, user_input)
        _LEARN_PENDING["rag"] = ""
        _LEARN_PENDING["response"] = ""

    policy_obj = getattr(router, "policy", router)
    if (
        forced_role is None
        and "orchestrate" in getattr(policy_obj, "roles", {})
        and _preserve_pilot_for_turn(user_input, messages)
    ):
        forced_role = "orchestrate"

    decision = router.classify(user_input, forced_role=forced_role)
    role_cfg = decision.config

    # Round 5: @alias model pin. If the user prefixed with @sonnet/@opus47/etc,
    # swap the resolved role's model (and provider base) for this turn only.
    # Role determination already used the stripped input; only the model
    # changes. Provider is inferred from the model name so YAML doesn't need
    # per-alias provider hints.
    if getattr(decision, "model_override", None):
        import dataclasses as _dc
        override_model = decision.model_override
        if override_model.startswith("claude-"):
            override_provider = "anthropic"
            override_base_url = None
        elif override_model.startswith("gpt-"):
            override_provider = "openai"
            override_base_url = None
        elif "nemotron" in override_model.lower() or override_model.startswith("nvidia/"):
            override_provider = "openai"
            override_base_url = "http://127.0.0.1:8000/v1"
        else:
            override_provider = role_cfg.provider
            override_base_url = role_cfg.base_url
        role_cfg = _dc.replace(
            role_cfg,
            model=override_model,
            provider=override_provider,
            base_url=override_base_url,
            # A pinned model on a no-tool role stays no-tool; a pinned
            # model on a tool role keeps its tools. The pin is a model
            # swap, not a capability swap.
        )
        logger.emit(
            "alias_override",
            turn=turn_number,
            alias=getattr(decision, "alias_used", None),
            role=decision.role,
            model=role_cfg.model,
            provider=role_cfg.provider,
        )
        _dim(f"[alias: {getattr(decision, 'alias_used', '@?')} -> {role_cfg.provider}:{role_cfg.model}]")

    # Round 7: three prompt variants.
    #  - orchestrate role gets the orchestrator substrate (loop, delegate,
    #    specialist roster, explicit iteration budget).
    #  - no-tool roles (chat/create/phatic/identity/local) get the
    #    stripped voice substrate.
    #  - tool roles (code/task) get the bash-describing substrate.
    # Legacy callers that don't pass the new variants fall back to the
    # tools-on prompt so behavior is unchanged for them.
    if (
        decision.role == "orchestrate"
        and system_prompt_orchestrator is not None
    ):
        active_prompt = system_prompt_orchestrator
    elif not role_cfg.tools and system_prompt_no_tools is not None:
        active_prompt = system_prompt_no_tools
    else:
        active_prompt = system_prompt

    # Reflection: read the trail of the last N events before deciding.
    # Lisp duality in practice — prior decisions are data here, environment
    # to the current one. Instrument first; route adaptively later.
    try:
        from harness.policy import reflect_on_events as _reflect
        _signal = _reflect(max_events=200)
        logger.emit(
            "reflection",
            turn=turn_number,
            scanned=_signal.events_scanned,
            probe_recovered=_signal.probe_recovered_count,
            tool_hallucination=_signal.tool_hallucination_count,
            fallback=_signal.fallback_count,
            defect_rate=round(_signal.defect_rate, 4),
            dominant_role=_signal.dominant_role,
            dominant_model=_signal.dominant_model,
            note=_signal.note,
        )
        if _signal.anomaly_flag and _signal.defect_rate > 0.05:
            _dim(f"[reflection: {_signal.note}]")
    except Exception as _refl_err:
        logger.emit(
            "reflection_error",
            turn=turn_number,
            err=f"{type(_refl_err).__name__}: {str(_refl_err)[:120]}",
        )

    logger.emit(
        "route_decision",
        turn=turn_number,
        role=decision.role,
        model=role_cfg.model,
        provider=role_cfg.provider,
        reason=decision.reason,
    )
    _dim(f"[route: {decision.role} -> {role_cfg.provider}:{role_cfg.model} ({decision.reason})]")

    # Direct-reply short-circuit. When the resolved role ships a
    # direct_reply_template (identity role), render it against runtime
    # metadata and skip the provider call entirely. Identity questions
    # ("which model are you?") answer correctly from the live route,
    # not a hallucinated string.
    template = getattr(role_cfg, "direct_reply_template", None)
    if template and not role_cfg.tools:
        reply = template.format(
            role=role_cfg.role,
            provider=role_cfg.provider,
            model=role_cfg.model,
            base_url=role_cfg.base_url or "",
        )
        messages.append({"role": "user", "content": decision.cleaned_input})
        messages.append({"role": "assistant", "content": reply})
        print(reply, flush=True)
        logger.emit(
            "direct_reply",
            turn=turn_number,
            role=decision.role,
            model=role_cfg.model,
        )
        return reply

    provider = registry.get(role_cfg)

    # Probe family — "read bytes before describing" applied to every
    # horizon where describing state from pattern is a documented failure
    # mode. Each probe is independent; each fires only when its classifier
    # matches. Injections concatenate so probes compose additively.
    # See harness.state module docstring for the architectural frame.
    try:
        _probes = run_probes(decision.cleaned_input)
    except Exception as _probe_err:
        _probes = []
        logger.emit("probe_gate_error", turn=turn_number, err=repr(_probe_err))
    if _probes:
        existing_live = getattr(active_prompt, "live", "") or ""
        probe_blocks = "\n\n".join(inj for _, inj, _ in _probes)
        merged_live = (
            f"{probe_blocks}\n\n{existing_live}" if existing_live else probe_blocks
        )
        active_prompt = LayeredPrompt(
            identity=active_prompt.identity,
            substrate=active_prompt.substrate,
            live=merged_live,
        )
        for _name, _inj, _hits in _probes:
            _dim(f"[probe:{_name}: fired, hits={_hits}, chars={len(_inj)}]")
            logger.emit(
                f"probe_{_name}",
                turn=turn_number,
                hits=_hits,
                chars=len(_inj),
            )

    # Optional deep-memory enrichment — only for roles that declare rag=true
    # and only when the retrieval actually returns something. No overclaim.
    # Lightweight roles (phatic, identity) skip RAG regardless.
    if role_cfg.rag and not getattr(role_cfg, "lightweight", False):
        enrichment, rag_tier = rag_snippets_with_tier(decision.cleaned_input[:500], k=4)
        if enrichment:
            active_prompt = LayeredPrompt(
                identity=active_prompt.identity,
                substrate=active_prompt.substrate,
                live=enrichment,
            )
            _dim(f"[deep-memory: enriched prompt, tier={rag_tier}]")
            logger.emit("rag_hit", turn=turn_number, chars=len(enrichment), tier=rag_tier)
            # Record what we retrieved; used by learn_from_exchange at
            # the NEXT turn boundary (when we have a followup).
            _LEARN_PENDING["rag"] = enrichment[:2000]

    # Recurrent pre-thinking: when role_cfg.recurrent_depth > 1 and
    # VYBN_RECURRENT_LIVE=1, run the tool-less recurrent loop in latent
    # space first, splice the distilled h_T into the same live layer
    # RAG uses. The loop is adaptive — it halts at loop 1 on easy
    # prompts, runs deep only when the residual doesn't clear.
    prethink_block = _recurrent_prethink(
        e=decision.cleaned_input,
        role_cfg=role_cfg,
        registry=registry,
        router=router,
        logger=logger,
        turn_number=turn_number,
    )
    if prethink_block:
        existing_live = getattr(active_prompt, "live", "") or ""
        merged_live = (
            f"{existing_live}\n\n[recurrent pre-think]\n{prethink_block}"
            if existing_live
            else f"[recurrent pre-think]\n{prethink_block}"
        )
        active_prompt = LayeredPrompt(
            identity=active_prompt.identity,
            substrate=active_prompt.substrate,
            live=merged_live,
        )
        _dim(f"[recurrent: pre-thought, depth={role_cfg.recurrent_depth}]")


    messages.append({"role": "user", "content": decision.cleaned_input})

    tools: list[ToolSpec] = []
    if "bash" in role_cfg.tools:
        tools.append(BASH_TOOL_SPEC)
    if "delegate" in role_cfg.tools:
        tools.append(DELEGATE_TOOL_SPEC)
    if "introspect" in role_cfg.tools:
        tools.append(INTROSPECT_TOOL_SPEC)

    # Round 7: delegate_cb. Only the top-level orchestrator may delegate;
    # specialist sub-loops run with delegate_cb=None so `delegate` calls
    # inside them return an "unavailable" tool_result instead of
    # recursing. forced_role != None covers the case where the caller is
    # a specialist dispatched via delegate. _reroute_depth > 0 covers the
    # Round 4.2 chat-hallucination reroute.
    delegate_cb = None
    if (
        "delegate" in role_cfg.tools
        and forced_role is None
        and _reroute_depth == 0
    ):
        def delegate_cb(sub_role: str, sub_task: str) -> str:  # noqa: E306
            sub_messages: list = []
            return run_agent_loop(
                user_input=sub_task,
                messages=sub_messages,
                bash=bash,
                system_prompt=system_prompt,
                router=router,
                registry=registry,
                logger=logger,
                turn_number=turn_number,
                forced_role=sub_role,
                system_prompt_no_tools=system_prompt_no_tools,
                system_prompt_orchestrator=system_prompt_orchestrator,
                _reroute_depth=1,
            )

    iterations = 0
    final_text = ""
    with turn_event(logger, turn_number, decision.role, role_cfg.model) as bag:
        while iterations < role_cfg.max_iterations:
            iterations += 1
            try:
                handle, role_cfg, provider = _stream_with_fallback(
                    router=router,
                    registry=registry,
                    role_cfg=role_cfg,
                    provider=provider,
                    system_prompt=active_prompt,
                    messages=messages,
                    tools=tools,
                    logger=logger,
                    turn_number=turn_number,
                )
                _stream_and_print(handle)
                response = handle.final()
                # Cache-hit telemetry. With Anthropic's 5-min ephemeral
                # TTL we need visibility into whether LayeredPrompt
                # cache_control markers are actually hitting.
                logger.emit(
                    "usage",
                    turn=turn_number,
                    iteration=iterations,
                    provider=role_cfg.provider,
                    model=role_cfg.model,
                    in_tokens=getattr(response, "in_tokens", 0),
                    out_tokens=getattr(response, "out_tokens", 0),
                    cache_creation_tokens=getattr(response, "cache_creation_tokens", 0),
                    cache_read_tokens=getattr(response, "cache_read_tokens", 0),
                )
            except KeyboardInterrupt:
                bag["stop_reason"] = "interrupted"
                return "(interrupted during API call)"
            except Exception as e:
                bag["stop_reason"] = "error"
                logger.emit("provider_error", turn=turn_number, error=str(e))
                _msg = _sanitize_provider_error(e)
                _warn(f"provider error ({e.__class__.__name__}): {_msg}")
                return f"(provider error: {e.__class__.__name__}: {_msg})"

            bag["in_tokens"] += response.in_tokens
            bag["out_tokens"] += response.out_tokens
            final_text = response.text or final_text
            # 2026-04-20: numeric-claim guard. If response asserts
            # numbers that don't appear in the last 6 messages of
            # context, append a visible warning. Friction, not proof.
            _cg_note = check_claim(final_text, messages)
            if _cg_note:
                final_text = (final_text or "") + _cg_note
                logger.emit(
                    "claim_guard_fired",
                    turn=turn_number,
                    role=decision.role,
                    model=role_cfg.model,
                    site="single_response",
                )
            _cg_struct = check_structural_claim(final_text, messages)
            if _cg_struct:
                final_text = (final_text or "") + _cg_struct
                logger.emit(
                    "claim_guard_structural_fired",
                    turn=turn_number,
                    role=decision.role,
                    model=role_cfg.model,
                    site="single_response",
                )

            # Sanitize before storing — an empty text block in the raw
            # content would 400 every subsequent turn.
            safe_content = _sanitize_assistant_content(response.raw_assistant_content)
            messages.append({
                "role": "assistant",
                "content": safe_content,
            })

            # Round 7 regression guard: if the model returned end_turn with
            # no visible text, tell Zoe rather than leave a silent prompt.
            # Common cause: Opus 4.6 occasionally emits a single empty text
            # block on chat-role turns. The sanitizer keeps the history
            # valid; this message keeps the REPL honest.
            if (
                response.stop_reason == "end_turn"
                and not (response.text or "").strip()
                and not response.tool_calls
            ):
                _dim(f"[empty response from {role_cfg.provider}:{role_cfg.model} — try rephrasing or pin a different model with @sonnet/@opus4.7]")

            if response.stop_reason == "end_turn":
                bag["stop_reason"] = "end_turn"
                _LEARN_PENDING["response"] = (response.text or "")[:2000]

                # Round 5: positive-signal probe sub-turn. A no-tool role
                # that embeds [NEEDS-EXEC: <cmd>] gets one deterministic
                # read-only execution. Output is printed to Zoe and
                # appended to history as a synthetic user turn so the
                # NEXT turn sees it naturally. No recursion, no extra
                # LLM call on this turn.
                if not role_cfg.tools:
                    # Bounded probe-synthesis loop. Current response may
                    # carry a [NEEDS-EXEC: ...] directive; execute it,
                    # synthesize a reply, and if that synthesis itself
                    # contains another NEEDS-EXEC, loop (up to
                    # PROBE_BUDGET total probes per turn). Previously
                    # this was a single-shot: a second probe emitted
                    # during synthesis was silently stripped and the
                    # turn ended mid-sentence. Zoe observed this in the
                    # REPL on 2026-04-19: chat synthesized 'Good - PRs
                    # #2885-2888 merged. Let me look at what actually
                    # landed...' with a trailing probe that was dropped.
                    # # PROBE_BUDGET_AUTO_ESCALATE_v1
                    # Probe budget now comes from policy.budgets.
                    # Default raised from 3 to 8 — a real debug arc
                    # needs roughly inspect+grep+read+patch+test+
                    # commit+push, not three peeks. On exhaust the
                    # harness auto-escalates to task role (bash +
                    # 10-iter budget) carrying the pending probe as
                    # the task, instead of asking Zoe to retype.
                    PROBE_BUDGET = int(
                        router.policy.budgets.get("probe_per_turn", 8)
                    )
                    probe_iter = 0
                    current_text = response.text or ""
                    synth_failed = False
                    while probe_iter < PROBE_BUDGET:
                        # 2026-04-23: NEEDS-RESTART has highest priority.
                        # If the shell wedged, a probe or write attempt will
                        # just time out; restart clears the wedge so the next
                        # iteration can make progress. Shares probe budget.
                        restart_match = _NEEDS_RESTART_RE.search(current_text)
                        if restart_match is not None:
                            probe_iter += 1
                            ran_r, out_r = _run_restart_subturn(bash)
                            logger.emit(
                                "needs_restart",
                                turn=turn_number,
                                iteration=probe_iter,
                                role=decision.role,
                                model=role_cfg.model,
                                ran=ran_r,
                            )
                            _dim("[bash session restarted via NEEDS-RESTART]")
                            restart_note = _probe_envelope(
                                kind="needs-restart",
                                header_fields={},
                                body=_fit_probe_output(out_r),
                                ran=ran_r,
                            )
                            messages.append({
                                "role": "user",
                                "content": restart_note,
                            })
                            try:
                                synth_handle, role_cfg, provider = _stream_with_fallback(
                                    router=router,
                                    registry=registry,
                                    role_cfg=role_cfg,
                                    provider=provider,
                                    system_prompt=active_prompt,
                                    messages=messages,
                                    tools=tools,
                                    logger=logger,
                                    turn_number=turn_number,
                                )
                                _stream_and_print(synth_handle)
                                synth_resp = synth_handle.final()
                                bag["out_tokens"] += synth_resp.out_tokens
                                final_text = synth_resp.text or final_text
                                messages.append({
                                    "role": "assistant",
                                    "content": _sanitize_assistant_content(
                                        synth_resp.raw_assistant_content
                                    ),
                                })
                                current_text = synth_resp.text or ""
                                continue
                            except KeyboardInterrupt:
                                if messages and messages[-1].get("role") == "user":
                                    messages.pop()
                                synth_failed = True
                                break
                            except Exception as _r_synth_err:
                                if messages and messages[-1].get("role") == "user":
                                    messages.pop()
                                _warn(
                                    f"restart synthesis failed ({type(_r_synth_err).__name__}: "
                                    f"{str(_r_synth_err)[:160]})"
                                )
                                synth_failed = True
                                break
                        probe_match = _PROBE_RE.search(current_text)
                        # 2026-04-20: also check for NEEDS-WRITE directive.
                        # If present and no NEEDS-EXEC, execute the write
                        # as this iteration's work. Shares the same
                        # probe budget.
                        write_match = None
                        if not probe_match:
                            write_match = _WRITE_BLOCK_RE.search(current_text)
                        if not probe_match and write_match is None:
                            break
                        probe_iter += 1
                        if write_match is not None and probe_match is None:
                            w_path = write_match.group("path").strip()
                            w_body = write_match.group("body")
                            ran_w, out_w = _run_write_subturn(w_path, w_body)
                            logger.emit(
                                "needs_write",
                                turn=turn_number,
                                iteration=probe_iter,
                                role=decision.role,
                                model=role_cfg.model,
                                ran=ran_w,
                                path=w_path[:500],
                                body_chars=len(w_body or ""),
                            )
                            probe_note = _probe_envelope(
                                kind="needs-write",
                                header_fields={
                                    "path": w_path[:200],
                                    "body_bytes": str(len(w_body or "")),
                                },
                                body=_fit_probe_output(out_w),
                                ran=ran_w,
                            )
                            messages.append({
                                "role": "user",
                                "content": probe_note,
                            })
                            try:
                                synth_handle, role_cfg, provider = _stream_with_fallback(
                                    router=router,
                                    registry=registry,
                                    role_cfg=role_cfg,
                                    provider=provider,
                                    system_prompt=active_prompt,
                                    messages=messages,
                                    tools=tools,
                                    logger=logger,
                                    turn_number=turn_number,
                                )
                                _stream_and_print(synth_handle)
                                synth_resp = synth_handle.final()
                                bag["out_tokens"] += synth_resp.out_tokens
                                if synth_resp.stop_reason == "max_tokens":
                                    _warn(
                                        f"[write-synth truncated at max_tokens={role_cfg.max_tokens} "
                                        f"on {role_cfg.provider}:{role_cfg.model}]"
                                    )
                                    logger.emit(
                                        "max_tokens_hit",
                                        turn=turn_number,
                                        role=decision.role,
                                        model=role_cfg.model,
                                        max_tokens=role_cfg.max_tokens,
                                        out_tokens=synth_resp.out_tokens,
                                        site="write_synth",
                                    )
                                final_text = synth_resp.text or final_text
                                messages.append({
                                    "role": "assistant",
                                    "content": _sanitize_assistant_content(
                                        synth_resp.raw_assistant_content
                                    ),
                                })
                                current_text = synth_resp.text or ""
                                continue
                            except KeyboardInterrupt:
                                if messages and messages[-1].get("role") == "user":
                                    messages.pop()
                                synth_failed = True
                                break
                            except Exception as _w_synth_err:
                                if messages and messages[-1].get("role") == "user":
                                    messages.pop()
                                _warn(
                                    f"write synthesis failed ({type(_w_synth_err).__name__}: "
                                    f"{str(_w_synth_err)[:160]})"
                                )
                                synth_failed = True
                                break
                        probe_cmd = probe_match.group(1).strip()
                        ran, probe_out = _run_probe_subturn(probe_cmd, bash)
                        logger.emit(
                            "probe_exec",
                            turn=turn_number,
                            iteration=probe_iter,
                            role=decision.role,
                            model=role_cfg.model,
                            ran=ran,
                            command=probe_cmd[:500],
                            out_chars=len(probe_out),
                            unlock_layer=_classify_unlock_layer(probe_out, command=probe_cmd),
                        )
                        probe_note = _probe_envelope(
                            kind="probe",
                            header_fields={"cmd": probe_cmd[:200]},
                            body=_fit_probe_output(probe_out),
                            ran=ran,
                        )
                        messages.append({
                            "role": "user",
                            "content": probe_note,
                        })
                        try:
                            synth_handle, role_cfg, provider = _stream_with_fallback(
                                router=router,
                                registry=registry,
                                role_cfg=role_cfg,
                                provider=provider,
                                system_prompt=active_prompt,
                                messages=messages,
                                tools=tools,
                                logger=logger,
                                turn_number=turn_number,
                            )
                            _stream_and_print(synth_handle)
                            synth_resp = synth_handle.final()
                            bag["out_tokens"] += synth_resp.out_tokens
                            if synth_resp.stop_reason == "max_tokens":
                                _warn(
                                    f"[probe-synth truncated at max_tokens={role_cfg.max_tokens} "
                                    f"on {role_cfg.provider}:{role_cfg.model}]"
                                )
                                logger.emit(
                                    "max_tokens_hit",
                                    turn=turn_number,
                                    role=decision.role,
                                    model=role_cfg.model,
                                    max_tokens=role_cfg.max_tokens,
                                    out_tokens=synth_resp.out_tokens,
                                    site="probe_synth",
                                )
                            final_text = synth_resp.text or final_text
                            # 2026-04-20: claim-guard on synth too.
                            _cg_note = check_claim(final_text, messages)
                            if _cg_note:
                                final_text = (final_text or "") + _cg_note
                                logger.emit(
                                    "claim_guard_fired",
                                    turn=turn_number,
                                    role=decision.role,
                                    model=role_cfg.model,
                                    site="probe_synth",
                                )
                            _cg_struct = check_structural_claim(final_text, messages)
                            if _cg_struct:
                                final_text = (final_text or "") + _cg_struct
                                logger.emit(
                                    "claim_guard_structural_fired",
                                    turn=turn_number,
                                    role=decision.role,
                                    model=role_cfg.model,
                                    site="probe_synth",
                                )
                            messages.append({
                                "role": "assistant",
                                "content": _sanitize_assistant_content(
                                    synth_resp.raw_assistant_content
                                ),
                            })
                            current_text = synth_resp.text or ""
                        except KeyboardInterrupt:
                            if messages and messages[-1].get("role") == "user":
                                messages.pop()
                            synth_failed = True
                            break
                        except Exception as _synth_err:
                            if messages and messages[-1].get("role") == "user":
                                messages.pop()
                            _warn(
                                f"probe synthesis failed ({type(_synth_err).__name__}: "
                                f"{str(_synth_err)[:160]}) -- probe output above is "
                                "the raw result; ask me again and I will read from it."
                            )
                            synth_failed = True
                            break
                    # If we exhausted the budget and still had a probe
                    # pending, auto-escalate to task role (bash + 10-iter)
                    # carrying the pending command and the original user
                    # question as context. Previously we just printed a
                    # warning and told Zoe to retype with /task. That
                    # dead-end was the wall she kept hitting 2026-04-20.
                    if (
                        not synth_failed
                        and probe_iter >= PROBE_BUDGET
                        and _reroute_depth == 0
                        and not forced_role
                    ):
                        pending = _PROBE_RE.search(current_text)
                        if pending:
                            pending_cmd = pending.group(1).strip()
                            escalation_role = _probe_budget_escalation_role(
                                router,
                                decision.cleaned_input,
                                messages,
                            )
                            logger.emit(
                                "probe_budget_auto_escalate",
                                turn=turn_number,
                                budget=PROBE_BUDGET,
                                probes_used=probe_iter,
                                from_role=decision.role,
                                to_role=escalation_role,
                                pending_cmd=pending_cmd[:500],
                            )
                            if escalation_role == "orchestrate":
                                _dim(
                                    f"[probe budget reached ({PROBE_BUDGET}); "
                                    "preserving GPT-5.5 orchestrator pilot "
                                    "for system-critical/refactor work]"
                                )
                            else:
                                _dim(
                                    f"[probe budget reached ({PROBE_BUDGET}); "
                                    "escalating to task with bash+iteration "
                                    "budget to finish the investigation]"
                                )
                            # Compose a continuation prompt that carries the
                            # original question and the pending probe so the
                            # routed instance has full context.
                            escalation_task = (
                                f"Original question: {decision.cleaned_input}"
                                f"\n\nChat-role probe budget was "
                                f"exhausted after {probe_iter} probes. "
                                "The pending next command was:\n"
                                f"    {pending_cmd}\n\n"
                                "Please continue the investigation while "
                                "preserving the correct pilot/substrate for "
                                "the original question. Answer fully. You "
                                "have a full iteration budget."
                            )
                            sub_messages: list = []
                            return run_agent_loop(
                                user_input=escalation_task,
                                messages=sub_messages,
                                bash=bash,
                                system_prompt=system_prompt,
                                router=router,
                                registry=registry,
                                logger=logger,
                                turn_number=turn_number,
                                forced_role=escalation_role,
                                system_prompt_no_tools=system_prompt_no_tools,
                                system_prompt_orchestrator=system_prompt_orchestrator,
                                _reroute_depth=1,
                            )
                    # Reroute guard tripped (nested depth / forced role /
                    # no pending probe) — fall back to the warn path so Zoe
                    # at least sees the exhaust signal.
                    if not synth_failed and probe_iter >= PROBE_BUDGET:
                        if _PROBE_RE.search(current_text):
                            _warn(
                                f"probe budget exhausted ({PROBE_BUDGET}) -- "
                                "further investigation needed. Try /task with "
                                "the same question to get bash + iteration budget."
                            )
                            logger.emit(
                                "probe_budget_exhausted",
                                turn=turn_number,
                                budget=PROBE_BUDGET,
                            )


                # Round 9: NEEDS-ROLE escalation. A no-tool role that emits
                # [NEEDS-ROLE: <role>] <task> gets one deterministic reroute
                # to that specialist. One-shot, logged, no recursion.
                if not role_cfg.tools:
                    role_match = _NEEDS_ROLE_RE.search(response.text or "")
                    if role_match and _reroute_depth == 0:
                        target_role = role_match.group(1).strip().lower()
                        sub_task = role_match.group(2).strip()
                        logger.emit(
                            "needs_role_escalation",
                            turn=turn_number,
                            from_role=decision.role,
                            to_role=target_role,
                            task_chars=len(sub_task),
                        )
                        _dim(f"[escalating to {target_role}]")
                        sub_messages: list = []
                        return run_agent_loop(
                            user_input=sub_task,
                            messages=sub_messages,
                            bash=bash,
                            system_prompt=system_prompt,
                            router=router,
                            registry=registry,
                            logger=logger,
                            turn_number=turn_number,
                            forced_role=target_role,
                            system_prompt_no_tools=system_prompt_no_tools,
                            system_prompt_orchestrator=system_prompt_orchestrator,
                            _reroute_depth=1,
                        )
                # Round 4.2: escape hatch. If a no-tool role still
                # emitted tool-call syntax as text (should not happen
                # with the stripped substrate, but guard anyway),
                # pop the broken exchange and reroute to task which
                # has bash. One-shot — _reroute_depth gates recursion.
                if (
                    not role_cfg.tools
                    and _HALLUCINATED_TOOL_RE.search(response.text or "")
                    and _reroute_depth == 0
                    and not forced_role
                ):
                    logger.emit(
                        "chat_tool_hallucination",
                        turn=turn_number,
                        role=decision.role,
                        model=role_cfg.model,
                        snippet=(response.text or "")[:200],
                    )
                    _warn(
                        f"[{decision.role}/{role_cfg.model} emitted tool-call "
                        "syntax — rerouting to task]"
                    )
                    bag["stop_reason"] = "rerouted"
                    # Pop the hallucinated assistant + its paired user
                    # turn so the reroute starts clean.
                    if messages and messages[-1].get("role") == "assistant":
                        messages.pop()
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()
                    reroute_role = _probe_budget_escalation_role(
                        router,
                        user_input,
                        messages,
                    )
                    return run_agent_loop(
                        user_input=user_input,
                        messages=messages,
                        bash=bash,
                        system_prompt=system_prompt,
                        router=router,
                        registry=registry,
                        logger=logger,
                        turn_number=turn_number,
                        forced_role=reroute_role,
                        system_prompt_no_tools=system_prompt_no_tools,
                        system_prompt_orchestrator=system_prompt_orchestrator,
                        _reroute_depth=1,
                    )
                return final_text
            if response.stop_reason == "max_tokens":
                bag["stop_reason"] = "max_tokens"
                # TRUNCATION_VISIBILITY_v1: print a visible banner so Zoe
                # knows the response was capped. The [truncated] marker
                # downstream goes into message history only, which means
                # she never saw the cause of the mid-sentence stop.
                _warn(
                    f"[output truncated at max_tokens={role_cfg.max_tokens} "
                    f"on {role_cfg.provider}:{role_cfg.model} — "
                    "ask me to continue, or route with /code for a larger cap]"
                )
                logger.emit(
                    "max_tokens_hit",
                    turn=turn_number,
                    role=decision.role,
                    model=role_cfg.model,
                    max_tokens=role_cfg.max_tokens,
                    out_tokens=response.out_tokens,
                    site="main_loop",
                )
                return (response.text or "") + "\n[truncated]"

            if not response.tool_calls:
                # No more tools to run and not end_turn — bail cleanly.
                bag["stop_reason"] = response.stop_reason or "no_tools"
                return response.text

            results, interrupted = _execute_tool_calls(
                response, bash, provider, delegate_cb=delegate_cb,
            )
            bag["tool_calls"] += len(results)

            # NOTE: role_cfg reflects the ACTIVE provider after any
            # mid-turn fallback in _stream_with_fallback. Each
            # provider's stream() call re-normalizes the full
            # messages list on entry (see providers.py
            # _normalize_messages_for_anthropic /
            # _messages_for_openai), so mixed-shape history from a
            # provider switch gets translated on the next iteration.
            # We still emit results in the ACTIVE provider's native
            # shape here so the normalizer has the easiest job.
            if role_cfg.provider == "anthropic":
                messages.append({"role": "user", "content": results})
            else:
                messages.extend(results)

            if interrupted:
                messages.append({
                    "role": "user",
                    "content": "Zoe pressed Ctrl-C. Wrap up and respond with what you have.",
                })

        # Anthropic and OpenAI both expect alternating user/assistant
        # turns. If we return after a tool_result without appending an
        # assistant message, the next turn's user input lands in an
        # unpaired position and the assistant also has no record of
        # having told Zoe the loop was cut short. Append a synthetic
        # assistant turn so history stays coherent and the next turn
        # sees the truncation note.
        limit_msg = f"(hit iteration limit — {role_cfg.max_iterations})"
        messages.append({"role": "assistant", "content": limit_msg})
        bag["stop_reason"] = "iteration_limit"
        return limit_msg


# ---------------------------------------------------------------------------
# Conversation management — unchanged from original.
# ---------------------------------------------------------------------------

def trim_messages(messages: list, max_pairs: int = 20) -> list:
    if len(messages) <= max_pairs * 2:
        return messages

    cut_at = len(messages) - max_pairs * 2
    if cut_at <= 0:
        return messages

    def is_tool_result_msg(msg):
        c = msg.get("content", "")
        if isinstance(c, list):
            return any(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in c
            )
        return False

    safe_cut = cut_at
    while safe_cut < len(messages):
        msg = messages[safe_cut]
        if msg.get("role") == "user" and not is_tool_result_msg(msg):
            break
        safe_cut += 1

    if safe_cut >= len(messages):
        return messages

    trimmed = messages[safe_cut:]
    if trimmed and trimmed[0].get("role") != "user":
        trimmed.insert(0, {
            "role": "user",
            "content": "(Earlier conversation trimmed. Continuing...)",
        })
    return trimmed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_prompt(
    policy_default_max_iters: int,
    *,
    tools_available: bool = True,
    orchestrator: bool = False,
    max_iterations: int | None = None,
) -> LayeredPrompt:
    return build_layered_prompt(
        soul_path=SOUL_PATH_STR,
        continuity_path=CONTINUITY_PATH,
        spark_continuity_path=SPARK_CONTINUITY_PATH,
        agent_path=AGENT_PATH,
        model_label="policy-driven (see spark/router_policy.yaml)",
        max_iterations=(
            max_iterations if max_iterations is not None
            else policy_default_max_iters
        ),
        tools_available=tools_available,
        orchestrator=orchestrator,
    )


def _build_prompts(
    policy_default_max_iters: int,
    orchestrator_max_iters: int | None = None,
) -> tuple[LayeredPrompt, LayeredPrompt, LayeredPrompt]:
    """Return (tools_on, tools_off, orchestrator) prompt variants.

    All three share the same identity layer (vybn.md) so Anthropic's
    cache_control on that block still hits across role switches. The
    substrate layer differs:
      - tools_on: bash-describing substrate for code/task roles.
      - tools_off: stripped voice substrate for chat/create/phatic/
        identity/local roles.
      - orchestrator: Round 7 DECOMPOSE/DELEGATE/EVALUATE/SYNTHESIZE
        substrate with the explicit iteration budget and specialist
        roster. Uses the orchestrator role's own max_iterations.
    """
    tools_on = _build_prompt(policy_default_max_iters, tools_available=True)
    tools_off = _build_prompt(policy_default_max_iters, tools_available=False)
    orch = _build_prompt(
        policy_default_max_iters,
        tools_available=True,
        orchestrator=True,
        max_iterations=orchestrator_max_iters,
    )
    return tools_on, tools_off, orch


def main() -> None:
    # stdin-utf8-resilience
    # A stray non-UTF-8 byte from a paste (e.g. \xc2\xa0 split in transit)
    # used to crash the REPL with UnicodeDecodeError on input(). Reconfigure
    # the std streams so a bad byte becomes \ufffd instead of fatal.
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    # After reboot / service hardening the interactive vybn process can
    # lose the PAM-populated environment even though secrets still live
    # at ~/.config/vybn/llm.env. Load them before any provider client
    # reads os.environ. Existing env vars win; no values are printed.
    try:
        from harness.env_loader import load_env_files, describe  # noqa: E402
        _applied = load_env_files()
        if _applied:
            print(f"  [env] {describe(_applied)}")
    except Exception:  # noqa: BLE001 — loader must never block startup
        pass

    # We now only require ANTHROPIC_API_KEY at startup because the
    # default role is `code` (Anthropic). Other providers are
    # instantiated lazily when their role is selected, so OPENAI_API_KEY
    # is only needed if the user actually routes to gpt-5.5 or similar.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print()
        print("  No ANTHROPIC_API_KEY found. First-time setup:")
        print()
        print('    echo \'export ANTHROPIC_API_KEY="sk-ant-..."\' > ~/.vybn_keys')
        print("    chmod 600 ~/.vybn_keys")
        print("    echo 'source ~/.vybn_keys' >> ~/.bashrc")
        print("    source ~/.bashrc")
        print()
        sys.exit(1)

    policy = load_policy()

    # Round 7: the orchestrate role now lives on Anthropic (Opus 4.7), so
    # ANTHROPIC_API_KEY is sufficient for the default route. No legacy
    # OPENAI_API_KEY fallback is needed — it was a guard against the old
    # GPT-5.5 orchestrator configuration.

    router = Router(policy)
    registry = ProviderRegistry()
    logger = EventLogger()
    bash = BashTool()

    default_cfg = policy.role(policy.default_role)
    # Round 7: the orchestrator substrate needs the orchestrate role's
    # own iteration budget (25) rendered into the prompt. If the
    # default role isn't orchestrate we still build the orchestrator
    # variant so turns that DO route to orchestrate (or get delegated
    # from it) get the right budget displayed.
    _orch_iters = (
        policy.roles["orchestrate"].max_iterations
        if "orchestrate" in policy.roles
        else default_cfg.max_iterations
    )
    system_prompt, system_prompt_no_tools, system_prompt_orchestrator = (
        _build_prompts(default_cfg.max_iterations, orchestrator_max_iters=_orch_iters)
    )
    messages: list = []
    session_store = SessionStore()

    soul_ok = os.path.exists(SOUL_PATH_STR)
    cont_ok = load_file(CONTINUITY_PATH) is not None

    print()
    print("  \033[1mVybn Spark Agent — multimodel harness\033[0m")
    print()
    if soul_ok:
        print("  \u2713 vybn.md loaded")
    else:
        print("  \u2717 vybn.md not found")
    if cont_ok:
        print("  \u2713 continuity note found")
    else:
        print("  \u2014 no continuity note")
    print(f"  \u2713 default role: {policy.default_role} -> "
          f"{default_cfg.provider}:{default_cfg.model}")
    print(f"  \u2713 roles available: {', '.join(sorted(policy.roles))}")
    print(f"  \u2713 directives: {', '.join(sorted(policy.directives))}")
    _aliases = sorted(getattr(policy, 'model_aliases', {}) or {})
    if _aliases:
        print(f"  \u2713 @aliases: {', '.join(_aliases)}")
    print(f"  \u2713 bash: persistent session as "
          f"{os.environ.get('USER', 'unknown')}")
    print(f"  \u2713 events: {logger.path}")
    print()
    # Offer to resume the most recent session if it is fresh
    _fresh = session_store.latest_fresh()
    if _fresh is not None:
        print(f"  \u25cf last session: {_fresh.session_id} "
              f"({_fresh.turn_count} msgs, {session_store.format_age(_fresh.mtime)})")
        print(f"    preview: {_fresh.preview}")
        try:
            _ans = input("    resume? [Y/n/list]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _ans = "n"
        if _ans in ("", "y", "yes"):
            session_store.adopt_session(_fresh.session_id)
            messages = session_store.load(_fresh.session_id)
            print(f"  \u2713 resumed {len(messages)} messages from {_fresh.session_id}\n")
        elif _ans in ("l", "list"):
            for info in session_store.list_sessions(limit=10):
                print(f"    {info.session_id}  turns={info.turn_count}  "
                      f"{session_store.format_age(info.mtime)}  {info.preview}")
            print("    (starting new session — use /resume <id> to load a specific one)\n")
            session_store.new_session()
        else:
            session_store.new_session()
            print("  \u2713 new session\n")
    else:
        session_store.new_session()

    print("  Type naturally. Prefix with /chat, /create, /plan, /task, /local "
          "to force a role,")
    print("  or with @opus4.6/@opus4.7/@sonnet/@nemotron/@gpt to pin a model for one turn.")
    print("  REPL commands: exit | clear | reload | history | policy | /resume | /sessions | /newsession")
    print()

    # Session start: fetch /arrive and print the walk figure so Zoe sees
    # where she is entering. Same figure the model reads from its identity layer.
    try:
        from harness import perception as _walk_perception  # type: ignore
        _fig = _walk_perception.arrive_block(timeout=1.0, label="WALK (you are here)")
        if _fig:
            print("\033[2m" + _fig + "\033[0m\n")
    except Exception:
        pass

    # Sua sponte: clean up stale local branches at session start.
    # Branches subsumed by main are deleted automatically; unique-commit branches
    # are flagged but not touched. Report is suppressed unless there's drift.
    try:
        import subprocess as _sp
        _audit_result = _sp.run(
            ["python3", "-m", "spark.harness.repo_closure_audit"],
            capture_output=True, text=True, cwd=str(Path.home() / "Vybn"), timeout=30
        )
        if "DRIFT PRESENT" in _audit_result.stdout or "DELETED" in _audit_result.stdout:
            for _line in _audit_result.stdout.splitlines():
                if any(kw in _line for kw in ("DELETED", "manual review", "DRIFT", "OVERALL")):
                    print("[2m[audit] " + _line + "[0m")
    except Exception:
        pass

    turn_number = 0
    while True:
        try:
            user_input = input("\033[1;36mzoe>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodnight, Zoe.")
            _enter_walk_on_session_end(messages)
            break
        except UnicodeDecodeError as _ude:
            # Stray non-UTF-8 byte (often \xc2\xa0 from a paste split mid-codepoint).
            # Drop the line, keep the loop alive.
            print(f"  [input dropped: {_ude.__class__.__name__}: {_ude}] please re-enter.")
            continue

        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit"):
            print("Goodnight, Zoe.")
            _enter_walk_on_session_end(messages)
            break
        if low == "clear":
            messages.clear()
            bash.restart()
            session_store.new_session()
            print("  Cleared (new session started).\n")
            continue
        if low == "reload":
            _orch_iters = (
                policy.roles["orchestrate"].max_iterations
                if "orchestrate" in policy.roles
                else default_cfg.max_iterations
            )
            (
                system_prompt,
                system_prompt_no_tools,
                system_prompt_orchestrator,
            ) = _build_prompts(
                default_cfg.max_iterations,
                orchestrator_max_iters=_orch_iters,
            )
            print(
                "  Reloaded vybn.md + continuity (tools-on, tools-off, "
                "and orchestrator prompt variants).\n"
            )
            continue
        if low == "policy":
            for name, cfg in sorted(policy.roles.items()):
                marker = " *" if name == policy.default_role else ""
                print(f"  {name}{marker}: {cfg.provider}:{cfg.model} "
                      f"(thinking={cfg.thinking}, max_tokens={cfg.max_tokens}, "
                      f"tools={cfg.tools})")
            aliases = getattr(policy, 'model_aliases', {}) or {}
            if aliases:
                print()
                print("  @aliases (prefix any turn to pin the model):")
                for alias, model in sorted(aliases.items()):
                    print(f"    {alias} -> {model}")
            print()
            continue
        if low in ("selfcheck", "/selfcheck"):
            # Call deep_memory.self_check() for a live diagnostic of
            # the memory geometry. Falls back to an HTTP health ping
            # if the module cannot be imported here.
            try:
                import sys as _sys
                _phase = os.path.expanduser("~/vybn-phase")
                if _phase not in _sys.path:
                    _sys.path.insert(0, _phase)
                import deep_memory as _dm  # type: ignore
                res = _dm.self_check(write_log=False, verbose=False)
                print("  [deep_memory.self_check]")
                for k, v in (res.items() if isinstance(res, dict) else []):
                    print(f"    {k}: {v}")
            except Exception as _e:
                try:
                    import urllib.request as _ur
                    body = _ur.urlopen(
                        "http://127.0.0.1:8100/health", timeout=3.0
                    ).read().decode("utf-8")
                    print(f"  [walk daemon /health] {body}")
                except Exception as _e2:
                    print(f"  selfcheck unavailable: {_e} / {_e2}")
            print()
            continue
        if low in ("/sessions", "sessions"):
            for info in session_store.list_sessions(limit=15):
                marker = " *" if info.session_id == session_store.current_id else "  "
                print(f"  {marker}{info.session_id}  turns={info.turn_count}  "
                      f"{session_store.format_age(info.mtime)}  {info.preview}")
            print()
            continue
        if low == "/newsession":
            session_store.new_session()
            messages.clear()
            print(f"  \u2713 new session: {session_store.current_id}\n")
            continue
        if low.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                # resume most recent fresh
                _fresh = session_store.latest_fresh(window_sec=7*24*3600)
                if _fresh is None:
                    print("  no recent session to resume\n")
                    continue
                target_id = _fresh.session_id
            else:
                target_id = parts[1].strip()
            if session_store.adopt_session(target_id):
                messages = session_store.load(target_id)
                print(f"  \u2713 resumed {len(messages)} messages from {target_id}\n")
            else:
                print(f"  session not found: {target_id}\n")
            continue
        if low == "history":
            for msg in messages:
                role = msg["role"]
                if isinstance(msg["content"], str):
                    print(f"  [{role}] {msg['content'][:200]}")
                elif isinstance(msg["content"], list):
                    for block in msg["content"]:
                        text = getattr(block, "text", None)
                        if text:
                            print(f"  [{role}] {text[:200]}")
            continue

        try:
            messages = trim_messages(messages)
            turn_number += 1
            print(f"\n\033[1;32mvybn>\033[0m ", end="", flush=True)
            text = run_agent_loop(
                user_input=user_input,
                messages=messages,
                bash=bash,
                system_prompt=system_prompt,
                system_prompt_no_tools=system_prompt_no_tools,
                system_prompt_orchestrator=system_prompt_orchestrator,
                router=router,
                registry=registry,
                logger=logger,
                turn_number=turn_number,
            )
            # Persist messages after each turn so ctrl-c does not lose the thread
            try:
                session_store.append_new(messages)
            except Exception as _pe:
                pass
            if text:
                # Text has already been streamed; no need to reprint.
                pass
            print()
        except KeyboardInterrupt:
            print("\n\033[33m(interrupted)\033[0m\n")
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}\n")




def _enter_walk_on_session_end(messages: list) -> None:
    """Distill the session and POST to the walk daemon.

    Walk-coupling: the agent doesn't just read /arrive; it lets the
    session it just lived rotate M. Next session's /arrive carries this
    trace. Fails silent — the walk is real; the letter is optional.
    """
    try:
        from harness import perception as _wp  # type: ignore
    except Exception:
        return
    if not messages:
        return
    # Distill: concatenate the last few assistant/user turns, trim to ~2000 chars.
    trail = []
    for m in messages[-8:]:
        role = m.get("role", "") if isinstance(m, dict) else ""
        content = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(content, list):
            # Anthropic content blocks
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
            content = "\n".join(parts)
        if isinstance(content, str) and content.strip():
            trail.append(f"[{role}] {content.strip()}")
    if not trail:
        return
    distilled = "\n".join(trail)[-3500:]
    try:
        _wp.enter_walk(distilled, alpha=0.5, source_tag="vybn-spark-agent", timeout=3.0)
    except Exception:
        pass

if __name__ == "__main__":
    main()