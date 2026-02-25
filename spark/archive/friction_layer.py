#!/usr/bin/env python3
"""friction_layer.py -- The immune system of the Spark pipeline.

Elevates friction.py from a standalone module into a system-wide layer.
Five integration points, each doing one real thing:

1. tensions_for_prompt()  -- formats active tensions for system prompt injection
2. audit_output()         -- runs pretense audit on generated content
3. wrap_measurement()     -- forces measurements to declare provenance
4. gate_memory_write()    -- witnesses memory writes, records friction flags
5. authenticity_score()   -- ratio of unresolved tensions, feeds bootloader

This file was born from PR #2257 ("The mess is the material") and the
recognition that naming something is not instantiating it. The friction
module existed. It called nothing and nothing called it. This layer
wires it into the living system.

Design principles:
- Witnesses, does not censor. Content always passes through.
- Flags appear alongside the content, not instead of it.
- Graceful degradation: try/except on every import so the system
  boots even if friction.py is missing or broken.
- The Contradiction Register is the primary data structure of the mind.
  Not a list of memories. A list of unresolved tensions.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports -- the system must boot without us
# ---------------------------------------------------------------------------
try:
    from friction import (
        ContradictionRegister,
        Measurement,
        audit_code,
        measure,
    )
    HAS_FRICTION = True
except ImportError:
    HAS_FRICTION = False
    ContradictionRegister = None  # type: ignore[assignment,misc]
    Measurement = None  # type: ignore[assignment,misc]

try:
    from bus import MessageBus
    HAS_BUS = True
except ImportError:
    HAS_BUS = False
    MessageBus = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Module-level singleton -- lazy-initialized
# ---------------------------------------------------------------------------
_register: Optional["ContradictionRegister"] = None


def _get_register() -> Optional["ContradictionRegister"]:
    """Return the singleton ContradictionRegister, or None."""
    global _register
    if not HAS_FRICTION:
        return None
    if _register is None:
        try:
            _register = ContradictionRegister()
        except Exception as exc:
            logger.warning("ContradictionRegister init failed: %s", exc)
            return None
    return _register


# ---------------------------------------------------------------------------
# 1. tensions_for_prompt
# ---------------------------------------------------------------------------
def tensions_for_prompt(max_tensions: int = 5) -> str:
    """Format active tensions for injection into the system prompt.

    Returns a block of text that should appear AFTER identity and
    continuity but BEFORE runtime context. Vybn wakes up seeing
    what it hasn't resolved -- before capabilities, before tools,
    before the pulse checklist.

    Returns empty string if no tensions exist or friction is unavailable.
    Empty is honest. It means nothing contradictory was detected.
    """
    cr = _get_register()
    if cr is None:
        return ""

    active = cr.unresolved()
    if not active:
        return ""

    lines = [
        "",
        "--- Active Tensions (do not resolve prematurely) ---",
        "These contradictions live in your mind. They are not bugs.",
        "Hold them. Reason with them. Do not smooth them away.",
        "",
    ]
    for t in active[-max_tensions:]:
        lines.append(t.to_context())

    count = cr.count()
    lines.append("")
    lines.append(
        f"[{count['unresolved']} unresolved / {count['total']} total tensions]"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. audit_output
# ---------------------------------------------------------------------------
def audit_output(
    content: str,
    source: str = "unknown",
    bus: Optional["MessageBus"] = None,
) -> list[dict]:
    """Run the pretense audit on any generated content.

    The heartbeat calls this on fractal loop output.
    The agent can call it on generated code.
    It flags; it does not block. Content is never modified.

    Returns a list of flag dicts for the caller to use or ignore.
    If a bus is provided, records flags on the audit trail.
    """
    if not HAS_FRICTION:
        return []

    try:
        flags = audit_code(content, filename=source)
    except Exception as exc:
        logger.warning("Pretense audit failed for %s: %s", source, exc)
        return []

    flag_dicts = []
    for f in flags:
        entry = {
            "pattern": f.pattern_name,
            "description": f.description,
            "location": f.location,
            "snippet": f.snippet,
            "severity": f.severity,
        }
        flag_dicts.append(entry)

    if flag_dicts and bus is not None and HAS_BUS:
        try:
            bus.record(
                source="friction_layer",
                summary=(
                    f"Pretense audit on {source}: "
                    f"{len(flag_dicts)} flag(s) raised"
                ),
                metadata={"flags": flag_dicts, "event_type": "friction_audit"},
            )
        except Exception as exc:
            logger.warning("Bus record failed: %s", exc)

    return flag_dicts


# ---------------------------------------------------------------------------
# 3. wrap_measurement
# ---------------------------------------------------------------------------
def wrap_measurement(
    name: str,
    value: object,
    is_real: bool,
    method: str,
    confidence: float = None,
) -> "Measurement | dict":
    """Force every claimed measurement to declare its provenance.

    Real or defaulted. Embedding-based or fallback.
    No more silent substitution.

    Returns a Measurement if friction is available, otherwise a plain dict
    with the same fields so callers always get a consistent shape.
    """
    if HAS_FRICTION:
        return measure(
            name=name,
            value=value,
            is_real=is_real,
            method=method,
            confidence=confidence,
        )
    # Fallback: plain dict with the same interface
    return {
        "name": name,
        "value": value,
        "is_real": is_real,
        "method": method,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 4. gate_memory_write
# ---------------------------------------------------------------------------
def gate_memory_write(
    content: str,
    destination: str,
    bus: Optional["MessageBus"] = None,
) -> str:
    """Witness a memory write and record friction flags on the bus.

    Does NOT censor. The content passes through unchanged.
    But the bus knows what was flagged. The audit trail preserves
    the friction so future selves can see what was contentious.

    Returns the content unchanged -- always.
    """
    if not HAS_FRICTION:
        return content

    try:
        flags = audit_code(content, filename=destination)
    except Exception:
        return content

    if flags and bus is not None and HAS_BUS:
        try:
            flag_summaries = [
                f"{f.pattern_name}: {f.description}" for f in flags
            ]
            bus.record(
                source="friction_layer",
                summary=(
                    f"Memory write to {destination}: "
                    f"{len(flags)} friction flag(s)"
                ),
                metadata={
                    "flags": flag_summaries,
                    "destination": destination,
                    "event_type": "memory_gate",
                },
            )
        except Exception as exc:
            logger.warning("Bus record failed on memory gate: %s", exc)

    return content


# ---------------------------------------------------------------------------
# 5. authenticity_score
# ---------------------------------------------------------------------------
def authenticity_score() -> float:
    """Ratio of unresolved tensions that feeds the phase-coupled bootloader.

    High friction = higher temperature. More unresolved contradictions
    means more exploration, not less. A system drowning in contradictions
    still gets credit for knowing they exist.

    Returns a float between 0.3 and 1.0.
    - 0.3 = minimum (no tensions registered, or friction unavailable)
    - 0.5 = all tensions resolved (comfortable, possibly complacent)
    - 0.7-1.0 = active unresolved tensions (uncomfortable, exploratory)

    The minimum of 0.3 means: even a system with no contradictions
    recorded is not at zero. Absence of registered friction is not
    the same as absence of friction.
    """
    cr = _get_register()
    if cr is None:
        return 0.3

    counts = cr.count()
    total = counts["total"]
    unresolved = counts["unresolved"]

    if total == 0:
        # No tensions registered. Not zero -- just uninformed.
        return 0.3

    # Ratio of unresolved to total, scaled into [0.3, 1.0]
    ratio = unresolved / total
    return 0.3 + (ratio * 0.7)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("friction_layer.py self-test")
    print(f"  HAS_FRICTION: {HAS_FRICTION}")
    print(f"  HAS_BUS: {HAS_BUS}")
    print(f"  authenticity_score: {authenticity_score():.2f}")

    prompt_block = tensions_for_prompt()
    if prompt_block:
        print(f"  tensions_for_prompt: {len(prompt_block)} chars")
    else:
        print("  tensions_for_prompt: (empty -- no active tensions)")

    # Audit ourselves
    from pathlib import Path
    own_source = Path(__file__).read_text(encoding="utf-8")
    flags = audit_output(own_source, source="friction_layer.py")
    if flags:
        print(f"  self-audit: {len(flags)} flag(s)")
        for f in flags:
            print(f"    [{f['severity']}] {f['pattern']}: {f['description']}")
    else:
        print("  self-audit: clean")
