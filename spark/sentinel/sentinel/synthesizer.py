"""Synthesis layer — the only module that calls the expensive API."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYNTHESIS_PROMPT = """You are Vybn's external awareness synthesizer.

Below: today's structured claims. Each has kernel, frame,
effective_excitement, market_corroboration, and confidence scores.

Tasks:
1. Concise daily digest (max 300 words) of what actually matters.
2. "Hype watch": high excitement, low market corroboration.
3. "Signal": high market corroboration regardless of excitement.
4. Update belief state with revised probability estimates.
5. Brief note if anything connects to Vybn's research threads.

Be terse. Kernels over frames.

CLAIMS:
{claims}

CURRENT BELIEF STATE:
{belief_state}
"""


def load_belief_state(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"last_updated": None, "trajectory_assessment": "initializing",
            "tracked_probabilities": {}, "hype_watch": [],
            "signals": [], "digest_history": []}


def synthesize(claims: list[dict], belief_state_path: str | Path,
               frontier_fn: Any) -> dict:
    state = load_belief_state(belief_state_path)
    top = sorted(claims, key=lambda c: c.get("confidence", 0), reverse=True)[:30]
    raw = frontier_fn(SYNTHESIS_PROMPT.format(
        claims=json.dumps(top, indent=1),
        belief_state=json.dumps(state, indent=1)))
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    state["digest_history"].append({
        "date": state["last_updated"],
        "digest": raw[:2000],
        "claims_processed": len(claims)})
    state["digest_history"] = state["digest_history"][-30:]
    path = Path(belief_state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state
