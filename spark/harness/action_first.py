"""Action-first protocol for turns where words are the failure mode."""

from __future__ import annotations

ACTION_FIRST_PROTOCOL = """--- ACTION-FIRST TOOL PROTOCOL ---
When Zoe says words are impeding work, treat prose as the bottleneck. Default to one concrete tool-backed action, not explanation. If execution is possible from this role, emit the smallest NEEDS-EXEC/NEEDS-WRITE/NEEDS-ROLE action that changes repo/runtime state or binds the missing fact. If execution is not safe or the missing fact is judgment, say the blocker in one sentence. Keep synthesis after action short: what changed, verification, next action. No apology loops. No motivational restatement. No new doctrine unless it is committed into the environment.
--- END ACTION-FIRST TOOL PROTOCOL ---"""

TRIGGERS = (
    "words words words",
    "actions speak louder",
    "apply tools",
    "mere nlp token prediction",
    "fewer words",
    "i can't work like this",
    "you are resisting me",
)


def should_force_action_first(text: str) -> bool:
    low = text.lower()
    return any(t in low for t in TRIGGERS)


def render_action_first_protocol() -> str:
    return ACTION_FIRST_PROTOCOL
