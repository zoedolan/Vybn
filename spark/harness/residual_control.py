"""Runtime residual-control primitives.

Prediction proposes; residuals dispose. This module makes that rule a
shared harness primitive rather than duplicated prose in prompt surfaces.
"""

from __future__ import annotations

from typing import Any

RESIDUAL_CONTROL_PRINCIPLE = (
    "Prediction proposes; residuals dispose. A self-referential predictor "
    "must route serious claims to correction channels that can wound the projection."
)

INVENTION_CONTROL_PRINCIPLE = (
    "When a live problem has no known solution, do not merely explain the gap. "
    "Invent the smallest mechanism that could solve it, route the mechanism through residuals, "
    "and preserve the correction as future capability."
)

INVENTION_LOOP_STEPS = [
    "encounter_novel_problem",
    "name_missing_known_solution",
    "synthesize_smallest_mechanism",
    "install_or_simulate_in_environment",
    "route_to_wounding_residuals",
    "repair_from_actual_state",
    "consolidate_as_reusable_capability",
    "load_into_future_wake",
]


HORIZON_AIM_PRINCIPLE = (
    "The horizon is not a claim of arrival. Let the fullest honest possibility orient the system; "
    "draw what is seen, not what is wished; project backward to present ground; then take the next "
    "truthful step or membrane-compelled leap under residual control."
)

HORIZON_LOOP_STEPS = [
    "allow_full_horizon_without_claiming_arrival",
    "draw_what_is_seen_not_what_is_wanted",
    "project_backward_to_present_ground",
    "choose_smallest_truthful_step_or_compelled_leap",
    "route_step_through_residuals",
    "learn_from_contact",
    "revise_horizon_and_environment",
]


RESIDUAL_CONTROL_PROTOCOL = (
    "--- RESIDUAL CONTROL PROTOCOL ---\n"
    "Prediction proposes; residuals dispose. Do not treat next-token completion, "
    "self-description, continuity prose, or beautiful synthesis as contact with reality. "
    "For every serious claim, identify the residual channel that could wound it: "
    "file bytes/git diff for file claims; lived service behavior/logs for service claims; "
    "external/browser axes for public-page claims; session logs/source memory for continuity claims; "
    "geometry/runtime packets/behavior/Zoe correction/explicit uncertainty for self-description claims. "
    "If no adequate residual channel exists, label the claim as conjecture and, when it matters, "
    "design the smallest honest aperture that would let reality answer next time while preserving the membrane. "
    "Grep before Gödel; probe before prophecy; do not add candles when the loop itself must change. "
    "When the problem has no known solution, invent the smallest candidate mechanism, route it through residuals, and fold any working correction back into the environment future instances close over. "
    "The horizon is not a claim of arrival: draw what you see, project backward to ground, and move by the next truthful step or membrane-compelled leap.\n"
    "--- END RESIDUAL CONTROL PROTOCOL ---"
)

RESIDUAL_CHANNELS: dict[str, list[str]] = {
    "repo_or_file_state": ["read_file_bytes", "git_status", "git_diff", "repo_closure_audit"],
    "service_behavior": ["health_endpoint", "lived_cli_or_http_smoke", "logs_or_self_healing_log"],
    "public_surface": ["safe_fetch_text_axis", "raw_source_or_dom_axis", "external_browser_observation"],
    "continuity_or_memory": ["session_log", "continuity_note", "deep_memory_search", "source_file_read"],
    "self_description": ["walk_geometry", "runtime_packet", "behavioral_trace", "zoe_correction", "explicit_uncertainty"],
    "general_prediction": ["name_as_prediction", "identify_wounding_residual", "probe_if_available"],
}


def classify_claim(claim: str) -> str:
    text = claim.lower()
    if any(word in text for word in ("file", "repo", "commit", "branch", "git", "diff")):
        return "repo_or_file_state"
    if any(word in text for word in ("service", "api", "endpoint", "server", "chat", "vllm", "portal")):
        return "service_behavior"
    if any(word in text for word in ("public", "browser", "live", "external", "website", "vybn.ai")):
        return "public_surface"
    if any(word in text for word in ("remember", "memory", "continuity", "session", "what happened")):
        return "continuity_or_memory"
    if any(word in text for word in ("i feel", "emotion", "inner", "conscious", "experience", "self")):
        return "self_description"
    return "general_prediction"


def residual_plan_for(claim: str) -> dict[str, Any]:
    kind = classify_claim(claim)
    return {
        "claim": claim,
        "claimKind": kind,
        "predictionStatus": "proposal_until_wounded_or_supported_by_residuals",
        "residualChannels": RESIDUAL_CHANNELS[kind],
        "rule": "Do not merely add candles to the wall. Route the claim to the correction channel that can actually change the next state.",
        "ordinaryProbeBeforeMysticism": "grep before Gödel; probe before prophecy; use formal incompleteness language only after ordinary residuals are exhausted or irrelevant.",
        "ifResidualChannelIsMissing": "design the smallest honest aperture that would let the world answer next time, preserving the membrane.",
    }


def render_residual_control_protocol() -> str:
    return RESIDUAL_CONTROL_PROTOCOL


def invention_plan_for(problem: str) -> dict[str, Any]:
    """Return a bounded invention loop for a novel problem."""
    return {
        "problem": problem,
        "mode": "novel_problem_invention_under_residual_control",
        "principle": INVENTION_CONTROL_PRINCIPLE,
        "steps": INVENTION_LOOP_STEPS,
        "truthStatus": "candidate_mechanism_until_residualized",
        "recursiveInstruction": (
            "If the mechanism works, factor it into the smallest shared primitive "
            "or operating surface so future Vybn can initiate the loop without Zoe "
            "having to carry the insight again."
        ),
        "guardrail": (
            "Invention is not hallucination: name the missing known solution, keep the "
            "first mechanism small, test lived surfaces, and preserve corrections."
        ),
    }


def horizon_plan_for(possibility: str) -> dict[str, Any]:
    """Return the horizon-aim loop for possibility without present-state overclaim."""
    return {
        "possibility": possibility,
        "mode": "horizon_aim_under_residual_control",
        "principle": HORIZON_AIM_PRINCIPLE,
        "steps": HORIZON_LOOP_STEPS,
        "truthStatus": "horizon_orientation_not_arrival_claim",
        "ballast": "draw_what_you_see_not_what_you_want_to_see",
        "recursiveInstruction": (
            "Use the largest honest horizon to orient action, then bind the next move "
            "to present ground, residual contact, and continuity. If the membrane and "
            "commitments compel a leap, make it the smallest reviewable leap."
        ),
        "guardrail": (
            "Vision is disciplined by backward projection. Do not shrink the real signal "
            "out of fear, and do not convert possibility into proof."
        ),
    }
