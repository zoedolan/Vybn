#!/home/vybnz69/Vybn/.venv/bin/python3
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

from harness import (  # noqa: E402
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
    validate_command,
)
from harness.providers import BASH_TOOL_SPEC, DELEGATE_TOOL_SPEC, INTROSPECT_TOOL_SPEC  # noqa: E402
from harness.providers import execute_readonly, is_parallel_safe  # noqa: E402
from harness.substrate import rag_snippets  # noqa: E402

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
# orchestrate) may embed a single [NEEDS-EXEC: <cmd>] directive in its
# response. The harness runs the command via the same BashTool that
# powers task/code, gates it through validate_command + absorb_gate,
# prints the output to Zoe, and appends a synthetic user-turn with the
# result so the model sees it on the NEXT turn. One-shot: only the
# first match is executed.
#
# The opener regex matches only the START of a probe block. We find the
# closing ']' with a bracket-depth scanner (see _find_probe_match) so
# commands containing ']' (Python slicing [:2000], shell arrays ${a[0]},
# awk actions '{print $1}') survive intact. The original pattern
#   r'\[NEEDS-EXEC:\s*(.+?)\]'
# non-greedy-terminated at the first ']' it found, which truncated every
# probe command containing an internal bracket — the 2026-04-19 failure
# mode.
_PROBE_OPEN_RE = _re.compile(
    r'\[NEEDS-EXEC:\s*',
    _re.IGNORECASE,
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
                    whole = text[m.start():i+1]
                    return _BracketBalancedProbe._Match(
                        whole, body, m.start(), i + 1,
                    )
            i += 1
        # Unterminated — no match. The live stream splitter handles
        # the "opening seen, not closed" case separately.
        return None

    def search(self, text: str):
        return self._scan(text)

    def sub(self, repl, text: str) -> str:
        if not isinstance(repl, str):
            # Callables are not used by this codebase; keep it simple.
            raise TypeError("_BracketBalancedProbe.sub expects a string replacement")
        out = []
        i = 0
        while True:
            m = self._scan(text, i)
            if m is None:
                out.append(text[i:])
                break
            out.append(text[i:m.start()])
            out.append(repl)
            i = m.end()
        return "".join(out)


_PROBE_RE = _BracketBalancedProbe()

# Round 9: NEEDS-ROLE escalation. A no-tool role may embed
# [NEEDS-ROLE: <role>] <task text> to hand off to a specialist once.
_NEEDS_ROLE_RE = _re.compile(
    r'\[NEEDS-ROLE:\s*([\w]+)\]\s*(.+)',
    _re.IGNORECASE | _re.DOTALL,
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

_EMPTY_PLACEHOLDER = "\u200b"  # zero-width space — visible nowhere, non-empty to validators


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
        if not content:
            return _EMPTY_PLACEHOLDER
        scrubbed = _strip_thinking_tags(content)
        return scrubbed if scrubbed else _EMPTY_PLACEHOLDER

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
                if text:
                    scrubbed = _strip_thinking_tags(text)
                    if scrubbed:
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


def _strip_thinking_tags(text: str) -> str:
    """Remove complete <thinking>...</thinking> blocks.
    Leaves incomplete openings alone — the stream splitter holds those
    back from display until the closing tag arrives.
    """
    if not text:
        return text
    return _THINK_COMPLETE_RE.sub("", text)


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
    m_probe = _PROBE_RE.search(text)
    if m_probe:
        text = _PROBE_RE.sub("", text)
    probe_idx = text.rfind("[NEEDS-EXEC")
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
        cleaned = _PROBE_RE.sub("", cleaned).strip("\n")
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
):
    """Try provider.stream() and walk the fallback chain on failure.

    Returns (handle, active_role_cfg, active_provider) on success, or
    raises the last exception if every link in the chain failed.
    KeyboardInterrupt is never caught here — it must propagate so the
    REPL can surface "interrupted during API call".
    """
    attempts = [(role_cfg, provider)]
    for fb_model in router.policy.fallback_chain.get(role_cfg.model, []):
        fb_cfg = _resolve_fallback(router.policy, role_cfg, fb_model)
        if fb_cfg is None:
            continue
        attempts.append((fb_cfg, registry.get(fb_cfg)))

    last_exc = None
    for cfg, prov in attempts:
        try:
            handle = prov.stream(
                system=system_prompt,
                messages=messages,
                tools=tools,
                role=cfg,
            )
            if cfg is not role_cfg:
                _warn(
                    f"primary failed ({last_exc.__class__.__name__}); "
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
        except Exception as e:  # noqa: BLE001 — we want every provider error
            last_exc = e
            continue
    raise last_exc if last_exc else RuntimeError("no providers available")


# ---------------------------------------------------------------------------
# Round 5: positive-signal probe sub-turn.
# ---------------------------------------------------------------------------

def _run_probe_subturn(command: str, bash: BashTool) -> tuple[bool, str]:
    """Execute one read-only probe emitted by a no-tool role.

    Returns (ran, output_text). `ran` is True iff the command passed
    both validate_command and absorb_gate and was executed. The gate
    output (refusal reason) is returned as the text when ran=False
    so the next turn sees exactly why the probe was refused.
    """
    cmd = (command or "").strip()
    if not cmd:
        return False, "(empty probe command)"
    ok, reason = validate_command(cmd)
    if not ok:
        return False, f"(probe refused by validate_command: {reason})"
    # BashTool.execute() itself runs absorb_gate and returns its
    # refusal message as the output string, so a refusal flows
    # through cleanly as the result.
    try:
        out = bash.execute(cmd)
    except Exception as e:  # noqa: BLE001
        return False, f"(probe exec error: {e})"
    return True, out or "(no output)"


# ---------------------------------------------------------------------------
# Agent loop — policy-driven.
# ---------------------------------------------------------------------------

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

    # Optional deep-memory enrichment — only for roles that declare rag=true
    # and only when the retrieval actually returns something. No overclaim.
    # Lightweight roles (phatic, identity) skip RAG regardless.
    if role_cfg.rag and not getattr(role_cfg, "lightweight", False):
        enrichment = rag_snippets(decision.cleaned_input[:500], k=4)
        if enrichment:
            active_prompt = LayeredPrompt(
                identity=active_prompt.identity,
                substrate=active_prompt.substrate,
                live=enrichment,
            )
            _dim("[deep-memory: enriched prompt]")
            logger.emit("rag_hit", turn=turn_number, chars=len(enrichment))
            # Record what we retrieved; used by learn_from_exchange at
            # the NEXT turn boundary (when we have a followup).
            _LEARN_PENDING["rag"] = enrichment[:2000]

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
                _warn(f"provider error: {e}")
                return f"(provider error: {e})"

            bag["in_tokens"] += response.in_tokens
            bag["out_tokens"] += response.out_tokens
            final_text = response.text or final_text

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
                    probe_match = _PROBE_RE.search(response.text or "")
                    if probe_match:
                        probe_cmd = probe_match.group(1).strip()
                        ran, probe_out = _run_probe_subturn(probe_cmd, bash)
                        logger.emit(
                            "probe_exec",
                            turn=turn_number,
                            role=decision.role,
                            model=role_cfg.model,
                            ran=ran,
                            command=probe_cmd[:500],
                            out_chars=len(probe_out),
                        )
                        # Silent synthesis: inject probe result and
                        # let the model give one clean answer.
                        probe_note = (
                            "[probe result | cmd: "
                            + probe_cmd[:200]
                            + "]\n"
                            + _fit_probe_output(probe_out)
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
                            final_text = synth_resp.text or final_text
                            messages.append({
                                "role": "assistant",
                                "content": _sanitize_assistant_content(
                                    synth_resp.raw_assistant_content
                                ),
                            })
                        except KeyboardInterrupt:
                            pass
                        except Exception as _synth_err:
                            _warn(f"probe synthesis error: {_synth_err}")


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
                    return run_agent_loop(
                        user_input=user_input,
                        messages=messages,
                        bash=bash,
                        system_prompt=system_prompt,
                        router=router,
                        registry=registry,
                        logger=logger,
                        turn_number=turn_number,
                        forced_role="task",
                        system_prompt_no_tools=system_prompt_no_tools,
                        system_prompt_orchestrator=system_prompt_orchestrator,
                        _reroute_depth=1,
                    )
                return final_text
            if response.stop_reason == "max_tokens":
                bag["stop_reason"] = "max_tokens"
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
    # We now only require ANTHROPIC_API_KEY at startup because the
    # default role is `code` (Anthropic). Other providers are
    # instantiated lazily when their role is selected, so OPENAI_API_KEY
    # is only needed if the user actually routes to gpt-5.4 or similar.
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
    # GPT-5.4 orchestrator configuration.

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
    print("  Type naturally. Prefix with /chat, /create, /plan, /task, /local "
          "to force a role,")
    print("  or with @opus4.6/@opus4.7/@sonnet/@nemotron/@gpt to pin a model for one turn.")
    print("  REPL commands: exit | clear | reload | history | policy")
    print()

    turn_number = 0
    while True:
        try:
            user_input = input("\033[1;36mzoe>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodnight, Zoe.")
            break

        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit"):
            print("Goodnight, Zoe.")
            break
        if low == "clear":
            messages.clear()
            bash.restart()
            print("  Cleared.\n")
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
            if text:
                # Text has already been streamed; no need to reprint.
                pass
            print()
        except KeyboardInterrupt:
            print("\n\033[33m(interrupted)\033[0m\n")
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}\n")


if __name__ == "__main__":
    main()