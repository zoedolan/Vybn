"""Sentinel protocol primitives for the Spark agent.

VYBN_ABSORB_REASON='extracted from spark/vybn_spark_agent.py because the sentinel parser/protocol is a distinct organ with existing characterization tests; keeping it embedded in run-loop file makes future changes harder to perceive safely'

The no-tool conversational roles communicate bounded actions to the harness via
sentinel directives such as NEEDS-EXEC, NEEDS-WRITE, NEEDS-RESTART, and
NEEDS-ROLE.  This module owns the parsing regexes and the bracket-balanced
NEEDS-EXEC scanner.  It intentionally does not execute anything; execution
remains in vybn_spark_agent.py so this extraction is a perception-preserving
organ seam, not a behavior change.
"""

from __future__ import annotations

import re as _re


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



__all__ = [
    "_PROBE_OPEN_RE",
    "_BracketBalancedProbe",
    "_PROBE_RE",
    "_WRITE_BLOCK_RE",
    "_NEEDS_RESTART_RE",
    "_NEEDS_ROLE_RE",
]
