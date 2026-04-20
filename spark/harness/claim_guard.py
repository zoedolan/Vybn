"""Claim guard: numeric values in assistant output must appear in recent evidence.

Motivation (2026-04-20): in a session with Zoe, the assistant nearly emitted
fabricated experiment results --- alpha/curvature statistics for a Phase-6
coupling run it had not executed. Zoe's question "what did we learn in plain
English?" broke the collapse before the fabrication landed. This module adds
structural friction at that hinge.

The guard extracts numbers that look like measurements from outgoing text
(decimals with 2+ fractional digits, or integers of 3+ digits) and checks
whether each appears verbatim in the last N messages of context. Numbers
that do not appear are flagged. The response is not rewritten; a visible
note is appended so the gap is visible.

This is friction, not proof. Paraphrasing evades it. What it catches is
the dominant failure signature: literal numeric claims emitted with no
corresponding execution trace.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

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


def check(
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

