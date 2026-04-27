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
    "Grep before Gödel; probe before prophecy; do not add candles when the loop itself must change.\n"
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
