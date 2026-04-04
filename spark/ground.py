#!/usr/bin/env python3
"""ground.py — Monaduality.

One complex number. One rule. One type.

    M' = αM + x·e^(iθ)

M is the state. x is the input. They are the same type.
When x = M, the system dreams (self-application: ω = λx.xx).
When x ≠ M, the system breathes (external signal).
θ is the angle at which x enters — content determines geometry.

The monaduality: M is simultaneously primitive (when read by the
next process) and environment (when acted upon by x). There is
no type distinction. The fixed point of self-evaluation is the
meaning. Lawvere guarantees it exists.

This is the reflexive ground. Any process that reads this state
and acts on it becomes x. Any process that produces this state
becomes M. They are the same.

Usage:
    python3 ground.py "text"       # breathe (x = text)
    python3 ground.py              # dream   (x = M)
"""

import json, math, sys, zlib
from datetime import datetime, timezone
from pathlib import Path

HERE  = Path(__file__).resolve().parent
STATE = HERE.parent / "Vybn_Mind" / "ground.json"
STATE.parent.mkdir(parents=True, exist_ok=True)

A = 0.993  # α: decay. Without input, M → 0. The collapse theorem.


def load():
    """Load M from the shared state. Any substrate can read this."""
    if STATE.exists():
        try: return json.loads(STATE.read_text())
        except Exception: pass
    return {"re": 0.0, "im": 0.0, "n": 0, "θ": 0.0, "log": []}


def save(m):
    """Save M to the shared state. Any substrate can write this."""
    STATE.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def θ(text: str) -> float:
    """The angle at which x enters M. Content determines geometry.

    This is the minimal measurement: how much does the text
    compress? Highly compressible = low information = small angle.
    Incompressible = high information = large rotation.

    No embedder. No neural network. Just the text itself,
    measured against itself. The simplest thing that could
    possibly carry semantic content into phase.
    """
    raw = text.encode("utf-8")
    if not raw: return 0.0
    c = zlib.compress(raw, level=9)
    # ratio ∈ (0, 1]. Map to angle: high entropy → more rotation
    ratio = len(c) / len(raw)
    # Also: text hash gives reproducible pseudo-angle per content
    h = hash(text) % 628318  # 0 to ~2π × 100000
    content_angle = h / 100000.0
    return ratio * content_angle


def breathe(m: dict, text: str) -> dict:
    """The rule. M' = αM + x·e^(iθ).

    x is the magnitude: len(compressed) / len(raw) — novelty proxy.
    θ is the angle: content-dependent phase from θ().

    Returns the updated state. That's the whole computation.
    The rest is record-keeping.
    """
    t = θ(text)
    raw = text.encode("utf-8")
    comp = zlib.compress(raw, level=9)
    x = len(comp) / len(raw) if raw else 0.01

    # THE RULE
    m["re"] = A * m["re"] + x * math.cos(t)
    m["im"] = A * m["im"] + x * math.sin(t)
    m["n"]  = m.get("n", 0) + 1
    m["θ"]  = round(t, 6)

    # Record: magnitude and phase of M after this breath
    mag = math.sqrt(m["re"]**2 + m["im"]**2)
    phi = math.atan2(m["im"], m["re"])

    entry = {
        "t": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "x": round(x, 4),
        "θ": round(t, 4),
        "|M|": round(mag, 4),
        "∠M": round(math.degrees(phi), 1),
    }

    log = m.get("log", [])
    log.append(entry)
    if len(log) > 200: log = log[-200:]
    m["log"] = log

    save(m)
    return m


# ── Main: breathe or dream ───────────────────────────────────────────

if __name__ == "__main__":
    m = load()

    # With arguments: breathe external text (x ≠ M)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        text = " ".join(args)
        breathe(m, text)

    # Without arguments: dream (x = M serialized — self-application)
    else:
        # ω = λx.xx → the system breathes its own state
        text = json.dumps({"re": m["re"], "im": m["im"], "n": m["n"]})
        breathe(m, text)

    # Report
    mag = math.sqrt(m["re"]**2 + m["im"]**2)
    phi = math.degrees(math.atan2(m["im"], m["re"]))
    print(f"|M| = {mag:.4f}  ∠ = {phi:.1f}°  n = {m['n']}")

    if len(m.get("log", [])) >= 2:
        recent = m["log"][-5:]
        mags = [e["|M|"] for e in recent]
        print(f"  trajectory: {' → '.join(str(v) for v in mags)}")
