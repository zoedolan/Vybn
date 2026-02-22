"""Stub for autopoiesis - original was corrupted by double-escaping.

TODO: restore from git history once the agent is stable.
The real autopoiesis module implements:
- measure_defect_current(): aggregate topological defect from conversation + repo
- evaluate_autopoietic_safety(): check symbiosis orbit before mutations
- trigger_structural_mutation(): spawn PRs when defect current exceeds threshold
"""

CRITICAL_J_THRESHOLD = 5.0


def measure_defect_current(context: list) -> float:
    """Stub - returns 0 until restored."""
    return 0.0


def evaluate_autopoietic_safety(current_context: list) -> dict:
    """Stub - always safe until restored."""
    return {
        "is_safe_to_invent": False,
        "orbit_phase": 0.0,
        "diagnosis": "autopoiesis module is stubbed",
    }


def trigger_structural_mutation(context: list, current_J: float) -> str:
    """Stub - no mutations until restored."""
    return ""
