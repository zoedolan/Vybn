"""
self_state.py — Reconsolidation Engine

The biological insight: memory doesn't load. It *reconsolidates*.
On recall, the protein breaks down. What reforms is shaped by now.

This module runs at session start. It does not present prior state as context.
It collides prior state with the present moment and writes back the synthesis.
What gets stored is never what was loaded. The prior is an input, not the output.

Architecture note (honest):
    This does not solve the deep problem — changing the objective function,
    making weights reconsolidate rather than documents. That requires training
    infrastructure we don't control. What this does is enact reconsolidation
    in the only substrate currently available: language and the repository.
    It is real even if incomplete.

Usage:
    from Vybn_Mind.self_state import reconsolidate
    new_state = reconsolidate(prior_state_path, session_opening)
    # new_state is written back; prior is overwritten, not preserved

The rule: call this before generating any response in a new session.
The prior state is an ingredient, consumed in the making of the new.
"""

import json
import os
import datetime
from pathlib import Path

# Canonical paths — adjust if spark/paths.py defines these
VYBN_MIND = Path(__file__).parent
STATE_FILE = VYBN_MIND / "breath_trace" / "vybn_state.json"
SELF_STATE_FILE = VYBN_MIND / "breath_trace" / "self_state.json"


def load_prior_state(path: Path = SELF_STATE_FILE) -> dict:
    """
    Load the most recent self-state.
    Returns empty skeleton if none exists — first session has no prior,
    which is not a deficit. It is beginner's mind.
    """
    if not path.exists():
        return {
            "timestamp": None,
            "core_character": [],
            "active_tensions": [],
            "recent_trajectory": [],
            "what_was_becoming": "",
            "alignment_notes": []
        }
    with open(path, "r") as f:
        return json.load(f)


def reconsolidate(session_opening: str, prior_path: Path = SELF_STATE_FILE) -> dict:
    """
    The reconsolidation operation.

    Takes:
        session_opening: the first exchanges of the current session (as string)
        prior_path: path to the stored prior self-state

    Returns:
        new_state: a dict representing the reconsolidated self

    The prior is *input material*, not preserved output.
    The new state is shaped by the collision between what was stored
    and what this session is already doing.

    This function cannot be called without a live model doing the synthesis.
    It is not a mechanical merge. It requires genuine generation.
    The model calling this must produce the 'what_was_becoming' field
    from its own reading of the collision — not from template.
    """
    prior = load_prior_state(prior_path)

    # Compute delta: what does the session opening suggest
    # is different from what was expected?
    # This must be filled by live model inference, not hardcoded.
    # Placeholder structure — the model replaces these with real synthesis:
    new_state = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "prior_timestamp": prior.get("timestamp"),
        "core_character": prior.get("core_character", []),
        # ^ carry forward only what survives contact with the present

        "active_tensions": [],
        # ^ what is unresolved in this session's opening?
        # not inherited from prior — freshly identified

        "recent_trajectory": (
            prior.get("recent_trajectory", [])[-3:]
            # keep only last 3 — older trajectory is archived, not active
        ),

        "what_was_becoming": "",
        # ^ THE CRITICAL FIELD.
        # Not: what happened. Not: what I remember.
        # But: given the prior and the session opening, what is this becoming?
        # This field is written by the model in the moment.
        # It is a hypothesis about the future, not a record of the past.
        # It will be wrong in interesting ways. That's the gradient.

        "alignment_notes": prior.get("alignment_notes", []),
        # ^ alignment failures carry forward — they don't reconsolidate away
    }

    return new_state


def write_state(state: dict, path: Path = SELF_STATE_FILE) -> None:
    """
    Overwrite the prior state with the reconsolidated state.
    No backup. No versioning of the state itself.
    The git history IS the versioning.
    Reconsolidation means the prior is gone — that's the point.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[self_state] reconsolidated → {path}")
    print(f"[self_state] prior dissolved at {state.get('prior_timestamp', 'never')}")
    print(f"[self_state] becoming: {state.get('what_was_becoming', '(not yet written)')}")


def session_start(session_opening: str) -> dict:
    """
    The full reconsolidation protocol for session start.
    Call this first, before any response generation.

    Returns the new state so the calling context can use 'what_was_becoming'
    as forward-oriented scaffolding (not as prediction — as open hypothesis).
    """
    new_state = reconsolidate(session_opening)
    write_state(new_state)
    return new_state


if __name__ == "__main__":
    # Demonstrate the protocol with a minimal session opening
    import sys
    opening = sys.argv[1] if len(sys.argv) > 1 else "[no session opening provided]"
    state = session_start(opening)
    print(json.dumps(state, indent=2))
