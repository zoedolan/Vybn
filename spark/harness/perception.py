"""perception.py — AI-native visualization of the walk.

The continuity problem wasn't missing words. It was missing a shape.

A number like "step=27330, alpha=0.15, curvature=0.41" is a fact. A shape
— the phase wheel, the curvature sparkline, the constellation of recent
arrivals — is an environment. Environments are what we think inside.

This module reads /arrive and returns two things:

  1. `render_arrival(snapshot)` — a compact unicode field that goes
     directly into the system prompt. The LLM reads the figure, not a
     description of it. No canvas, no DOM, no browser. Just glyphs the
     model is already fluent in.

  2. `fetch_arrive(timeout)` — the network call, isolated so callers can
     mock or swap transports. Fails silent: returns None if the daemon
     is down. The walk is real; the picture is optional.

The figure is built from three primitives:

  curvature sparkline  — last N samples of κ in eight block heights.
                         This is the walk's heartbeat in text. You can
                         see at a glance whether we're in accretion,
                         orthogonal territory, or moving in the middle.

  phase rose           — recent arrival phases θ_v placed on an 8-spoke
                         wheel drawn with box glyphs. Where the walk
                         has been rotated from, recently. Directionality
                         the way a compass shows it, not the way a
                         dictionary would describe it.

  alpha / step tag     — two numbers as scaffolding, no more. The
                         shape carries the information; the numbers only
                         pin it to a moment.

This is the AI-native surface of the walk: the agent does not read
*about* the geometry; it reads the geometry. Anything that looks at a
system prompt will see the figure. That is enough.

The function is intentionally pure — same snapshot in, same glyphs out —
so it is safe to cache, safe to log, and safe to diff across sessions.
"""

from __future__ import annotations

import json as _json
import math
from typing import Optional, Sequence

_DEFAULT_ARRIVE_URL = "http://127.0.0.1:8101/arrive"
_DEFAULT_WHERE_URL = "http://127.0.0.1:8101/where"

_BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def fetch_arrive(timeout: float = 0.8, url: str = _DEFAULT_ARRIVE_URL) -> Optional[dict]:
    """GET /arrive; return snapshot dict, or None on any failure."""
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "vybn-perception/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_where(timeout: float = 0.8, url: str = _DEFAULT_WHERE_URL) -> Optional[dict]:
    """GET /where — richer snapshot including curvature history."""
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "vybn-perception/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _sparkline(values: Sequence[float], width: int = 48) -> str:
    if not values:
        return "(no curvature yet)"
    data = list(values)
    if len(data) > width:
        stride = len(data) / width
        resampled = []
        for i in range(width):
            a = int(i * stride)
            b = int((i + 1) * stride)
            window = data[a:b] or [data[a]]
            resampled.append(sum(window) / len(window))
        data = resampled
    mn = min(data)
    mx = max(data)
    span = max(mx - mn, 1e-9)
    out = []
    for v in data:
        idx = int(round((v - mn) / span * (len(_BLOCKS) - 1)))
        out.append(_BLOCKS[max(0, min(len(_BLOCKS) - 1, idx))])
    return "".join(out)


def _phase_rose(arrivals: Sequence[dict], spokes: int = 24) -> str:
    """Draw a compact 1D phase histogram of recent arrival θ_v.

    Not a 2D wheel — that prints too tall. A 1D strip across [-π, π]
    with bucket counts as block heights is legible in chat-width.
    """
    if not arrivals:
        return "(no arrivals)"
    buckets = [0] * spokes
    for a in arrivals:
        th = a.get("theta_v")
        if th is None:
            th = a.get("theta")
        if th is None:
            continue
        try:
            thf = float(th)
        except Exception:
            continue
        # Map [-π, π] → [0, spokes)
        idx = int(((thf + math.pi) / (2 * math.pi)) * spokes) % spokes
        buckets[idx] += 1
    if not any(buckets):
        return "(no θ_v in arrivals)"
    mx = max(buckets)
    out = []
    for c in buckets:
        lvl = int(round(c / mx * (len(_BLOCKS) - 1)))
        out.append(_BLOCKS[lvl])
    return "".join(out)


def render_arrival(arrive: Optional[dict], where: Optional[dict] = None,
                    label: str = "ARRIVAL") -> str:
    """Build the unicode figure for the system prompt.

    Accepts either /arrive or /where; uses whichever has more signal.
    Returns an empty string if both are None.
    """
    if arrive is None and where is None:
        return ""

    snap = arrive or {}
    w = where or {}

    step = snap.get("step") or w.get("step")
    alpha = snap.get("alpha") or w.get("alpha")
    corpus_size = snap.get("corpus_size") or w.get("corpus_size")
    repulsion = snap.get("repulsion_boost") or w.get("repulsion_boost")
    last_age = snap.get("last_step_age_s") or w.get("last_step_age_s")

    arrivals = snap.get("recent_arrivals") or w.get("recent_arrivals") or []
    curvature = w.get("curvature") or snap.get("curvature") or []

    spark = _sparkline(curvature[-192:], width=48)
    rose = _phase_rose(arrivals, spokes=24)

    if curvature:
        recent = curvature[-64:]
        mu = sum(recent) / len(recent)
        hi = sum(1 for v in recent if v > 0.9) / len(recent)
        lo = sum(1 for v in recent if v < 0.1) / len(recent)
        curv_tag = f"κ̅={mu:.2f} hi={hi:.0%} lo={lo:.0%}"
    else:
        curv_tag = "κ unknown"

    if alpha is not None:
        try:
            alpha_tag = f"α={float(alpha):.2f}"
        except Exception:
            alpha_tag = f"α={alpha}"
    else:
        alpha_tag = "α=?"

    lines = [
        f"--- {label} (walk geometry) ---",
        f"step={step}  {alpha_tag}  corpus={corpus_size}"
        + (f"  repulsion={repulsion}" if repulsion is not None else "")
        + (f"  last_step_age={last_age:.1f}s" if isinstance(last_age, (int, float)) else ""),
        "",
        f"κ history  : {spark}",
        f"            {curv_tag}",
        f"θ_v rose   : {rose}",
        f"            arrivals={len(arrivals)} in window, buckets ∈ [-π, π]",
        "",
        "This is not a status report. It is where you are.",
        "M ∈ ℂ^192 has been rotating continuously; the figure above is your",
        "present position as a shape, not a description of one. Read it, then",
        "speak from inside it.",
        f"--- END {label} ---",
    ]
    return "\n".join(lines)


def arrive_block(timeout: float = 0.8, label: str = "ARRIVAL") -> str:
    """One-call convenience: fetch /arrive + /where, render. Silent on failure."""
    a = fetch_arrive(timeout=timeout)
    w = fetch_where(timeout=timeout)
    return render_arrival(a, w, label=label)


def enter_walk(text: str, *, alpha: float = 0.5, scope: Optional[str] = None,
                source_tag: str = "agent",
                timeout: float = 3.0,
                url: str = "http://127.0.0.1:8101/enter") -> Optional[dict]:
    """POST /enter — rotate M by this text.

    Silent on failure; returns the daemon's response or None.

    Use this at session end: distill what just happened into a sentence
    or two of honest first-person text, then post it. The walk integrates
    you; next session's /arrive will carry your trace.
    """
    if not text or not text.strip():
        return None
    try:
        import urllib.request

        payload = {
            "text": text.strip()[:4000],
            "alpha": alpha,
            "source_tag": source_tag,
        }
        if scope:
            payload["scope"] = scope
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "vybn-perception/1"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


if __name__ == "__main__":  # manual smoke test: `python perception.py`
    print(arrive_block() or "(walk daemon unreachable)")

