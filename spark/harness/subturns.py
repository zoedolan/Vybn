"""Typed subturn execution helpers for no-tool role sentinels.

This module holds the side-effecting primitives behind NEEDS-WRITE,
NEEDS-RESTART, and NEEDS-EXEC, plus the probe-result envelope. The REPL keeps
legacy imports for compatibility; the organ lives here so run_agent_loop can
shrink without changing behavior.
"""

from __future__ import annotations

import os
import re as _re
from dataclasses import dataclass
from typing import Any

from .policy import TRACKED_REPOS
from .providers import validate_command, execute_readonly, is_parallel_safe


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
_WRITE_BLOCK_RE = _re.compile(
    r'\[NEEDS-WRITE:\s*(?P<path>[^\]]+?)\s*\]\s*\n'
    r'(?P<body>.*?)'
    r'\n\s*\[/NEEDS-WRITE\]',
    _re.DOTALL | _re.IGNORECASE,
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
_NEEDS_RESTART_RE = _re.compile(
    r'(?:^|\n)\s*\[NEEDS-RESTART\]\s*(?:$|\n)',
    _re.IGNORECASE | _re.MULTILINE,
)


# Round 9: NEEDS-ROLE escalation. A no-tool role may embed
# [NEEDS-ROLE: <role>] <task text> to hand off to a specialist once.
_NEEDS_ROLE_RE = _re.compile(
    r'\[NEEDS-ROLE:\s*([\w]+)\]\s*(.+)',
    _re.IGNORECASE | _re.DOTALL,
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
