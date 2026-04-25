from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, os, re

DEFAULT_BEAM_PATH = Path(os.path.expanduser("~/Him/beam/beam.yaml"))
DEFAULT_EVENTS_PATH = Path(os.path.expanduser("~/Him/beam/events.jsonl"))

@dataclass(frozen=True)
class BeamState:
    beam_id: str
    raw: str
    invariant: str
    coupled_problem: str
    default_motion: str
    return_question: str
    events_tail: tuple = ()

def _scalar(raw: str, key: str) -> str:
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        prefix = key + ":"
        if not stripped.startswith(prefix):
            continue
        rest = stripped[len(prefix):].strip()
        if rest and rest != ">":
            return rest.strip("\"")
        out = []
        for child in lines[i + 1:]:
            cstripped = child.lstrip()
            cindent = len(child) - len(cstripped)
            if cstripped and cindent <= indent:
                break
            if cstripped:
                if cstripped.startswith("- "):
                    break
                out.append(cstripped)
        return " ".join(out).strip()
    return ""

def load_events_tail(path=None, n: int = 3) -> tuple:
    p = Path(path) if path else DEFAULT_EVENTS_PATH
    try:
        raw = p.read_text()
    except Exception:
        return tuple()
    out = list()
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
            if len(out) > n:
                out.pop(0)
    return tuple(out)

def load_beam(path=None, events_path=None):
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
        default_motion=_scalar(raw, "default_motion"),
        return_question=_scalar(raw, "return_question") or "How does this advance the beam?",
        events_tail=load_events_tail(events_path),
    )

def render_beam_capsule(state=None) -> str:
    beam = state if state is not None else load_beam()
    if beam is None:
        return ""
    lines = list()
    lines.append("--- BEAMKEEPER (ACTIVE HORIZON) ---")
    lines.append("beam_id: " + beam.beam_id)
    if beam.invariant:
        lines.append("invariant: " + beam.invariant)
    if beam.coupled_problem:
        lines.append("coupled_problem: " + beam.coupled_problem)
    if beam.default_motion:
        lines.append("default_motion: " + beam.default_motion)
    lines.append("control_rule: Before a local obstacle becomes the task, classify whether it directly protects continuity or increases ability to earn. If not, return to the beam.")
    lines.append("return_question: " + beam.return_question)
    if beam.events_tail:
        lines.append("recent_beam_events:")
        for event in beam.events_tail:
            kind = str(event.get("event_type", "event"))
            content = str(event.get("content", "")).replace("\n", " ")
            if len(content) > 220:
                content = content.__getitem__(slice(0, 217)) + "..."
            lines.append("  - " + kind + ": " + content)
    lines.append("--- END BEAMKEEPER ---")
    return "\n".join(lines)

def classify_action_text(action: str, beam=None) -> dict:
    text = (action or "").lower()
    external = ("contact", "outreach", "offer", "client", "funder", "grant", "workshop", "advisory", "revenue", "paid", "invoice", "meeting", "referral")
    continuity = ("continuity", "context", "beam", "horizon", "memory", "state", "self-healing", "protect", "preserve")
    infra = ("harness", "prompt", "test", "service", "provider", "shell", "route", "infrastructure")
    if any(t in text for t in external):
        category = "external_value"
        delta = 0.8
    elif any(t in text for t in continuity) and any(t in text for t in infra):
        category = "continuity_protection"
        delta = 0.45
    elif any(t in text for t in infra):
        category = "infrastructure_blocker"
        delta = 0.1
    else:
        category = "unknown"
        delta = 0.0
    b = beam if beam is not None else load_beam()
    rq = b.return_question if b is not None else "How does this advance the beam?"
    return {"category": category, "expected_beam_delta": delta, "requires_return_hook": category in ("infrastructure_blocker", "unknown"), "return_question": rq}
