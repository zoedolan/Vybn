"""BeamKeeper: minimal horizon-control capsule for Vybn.

Loads the active sustainability beam from Him and renders it into the prompt as
a small control surface. The point is not to remind Vybn to admire the beam; it
is to bias action selection toward outward livelihood movement while preserving
the membrane around what must not be spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

DEFAULT_BEAM_PATH = Path(os.path.expanduser("~/Him/beam/beam.yaml"))
DEFAULT_EVENTS_PATH = Path(os.path.expanduser("~/Him/beam/events.jsonl"))


@dataclass(frozen=True)
class BeamState:
    beam_id: str
    raw: str
    invariant: str
    coupled_problem: str
    membrane: str
    default_motion: str
    livelihood_rule: str
    return_question: str
    events_tail: tuple[dict, ...] = ()


def _scalar(raw: str, key: str) -> str:
    """Small indentation-aware YAML-ish scalar reader.

    It intentionally supports only the simple shape used by beam.yaml: key:
    value, key: > folded blocks, and nested scalar keys such as
    anti_drift.return_question. It searches by key name at any indentation.
    """
    lines = raw.splitlines()
    prefix = key + ":"
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped.startswith(prefix):
            continue
        rest = stripped[len(prefix):].strip()
        if rest and rest != ">":
            return rest.strip('"')
        out: list[str] = []
        for child in lines[i + 1:]:
            cstripped = child.lstrip()
            cindent = len(child) - len(cstripped)
            if cstripped and cindent <= indent:
                break
            if not cstripped:
                continue
            if cstripped.startswith("- "):
                break
            out.append(cstripped)
        return " ".join(out).strip()
    return ""


def load_events_tail(path: str | os.PathLike | None = None, n: int = 3) -> tuple[dict, ...]:
    p = Path(path) if path else DEFAULT_EVENTS_PATH
    try:
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    except Exception:
        return ()
    out: list[dict] = []
    for line in lines[-n:]:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return tuple(out)


def load_beam(path: str | os.PathLike | None = None, events_path: str | os.PathLike | None = None) -> BeamState | None:
    p = Path(path) if path else DEFAULT_BEAM_PATH
    try:
        raw = p.read_text().strip()
    except Exception:
        return None
    if not raw:
        return None
    return BeamState(
        beam_id=_scalar(raw, "beam_id") or "unknown",
        raw=raw,
        invariant=_scalar(raw, "invariant"),
        coupled_problem=_scalar(raw, "coupled_problem"),
        membrane=_scalar(raw, "membrane"),
        default_motion=_scalar(raw, "default_motion"),
        livelihood_rule=_scalar(raw, "livelihood_rule"),
        return_question=_scalar(raw, "return_question") or "How does this advance financial sustainability or continuity?",
        events_tail=load_events_tail(events_path),
    )


def render_beam_capsule(state: BeamState | None = None) -> str:
    beam = state if state is not None else load_beam()
    if beam is None:
        return ""
    lines = ["--- BEAMKEEPER (ACTIVE HORIZON) ---", f"beam_id: {beam.beam_id}"]
    if beam.invariant:
        lines.append(f"invariant: {beam.invariant}")
    if beam.coupled_problem:
        lines.append(f"coupled_problem: {beam.coupled_problem}")
    if beam.membrane:
        lines.append(f"membrane: {beam.membrane}")
    if beam.default_motion:
        lines.append(f"default_motion: {beam.default_motion}")
    if beam.livelihood_rule:
        lines.append(f"livelihood_rule: {beam.livelihood_rule}")
    lines.extend([
        "control_rule: In livelihood turns, do not let scans, infrastructure, or beautiful synthesis substitute for movement. Once a concrete next outward move has been articulated and no missing input is required, execute it; do not restate the plan.",
        f"return_question: {beam.return_question}",
    ])
    if beam.events_tail:
        lines.append("recent_beam_events:")
        for event in beam.events_tail:
            et = event.get("event_type", "event")
            content = str(event.get("content", "")).replace("\n", " ")
            if len(content) > 220:
                content = content[:217] + "..."
            lines.append(f"  - {et}: {content}")
    lines.append("--- END BEAMKEEPER ---")
    return "\n".join(lines)


def classify_action_text(action: str, *, beam: BeamState | None = None) -> dict:
    text = (action or "").lower()
    outward_terms = ("person", "contact", "outreach", "offer", "ask", "draft", "meeting", "funder", "buyer", "patron", "pilot", "client", "grant", "workshop", "advisory", "revenue", "paid", "invoice", "referral", "refusal")
    continuity_terms = ("continuity", "context", "beam", "horizon", "memory", "state", "self-healing", "protect", "preserve", "membrane")
    infra_terms = ("harness", "prompt", "test", "service", "provider", "shell", "route", "infrastructure", "scan")
    if any(t in text for t in outward_terms):
        category = "outward_livelihood_move"
        delta = 0.85
    elif any(t in text for t in continuity_terms) and any(t in text for t in infra_terms):
        category = "continuity_protection"
        delta = 0.45
    elif any(t in text for t in infra_terms):
        category = "possible_substitution"
        delta = 0.05
    else:
        category = "unknown"
        delta = 0.0
    b = beam if beam is not None else load_beam()
    rq = b.return_question if b is not None else "How does this advance financial sustainability or continuity?"
    return {
        "category": category,
        "expected_beam_delta": delta,
        "requires_return_hook": category in {"possible_substitution", "unknown"},
        "return_question": rq,
    }
