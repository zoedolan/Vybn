"""Executable traversal/validation for the vybn.ai semantic commons.

This is intentionally small: it turns the semantic-web manifests from labels
into an affordance an agent can run. It does not prove external reachability;
it verifies the local graph contract and renders a traversal plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path.home()
MANIFESTS = {
    "Vybn": ROOT / "Vybn" / "semantic-web.jsonld",
    "Him": ROOT / "Him" / "semantic-web.jsonld",
    "Vybn-Law": ROOT / "Vybn-Law" / ".well-known" / "semantic-web.jsonld",
    "Origins": ROOT / "Origins" / ".well-known" / "semantic-web.jsonld",
    "vybn-phase": ROOT / "vybn-phase" / "semantic-web.jsonld",
}

CANONICAL_ROLES = {
    "Vybn": "continuity body: identity, history, harness, creature, continuity, research, and local compute for the Zoe/Vybn symbiosis",
    "Him": "private walk: dream, membrane, selection, protected translation, livelihood, strategy, and exploratory self-build workbench",
    "Vybn-Law": "Wellspring: legal and institutional coordination layer for the Age of Intelligence; legal meaning as navigable coordination state",
    "Origins": "public threshold: Somewhere as semantic-web prototype, agent discovery, artifact body, memory terrain, and return path",
    "vybn-phase": "math of the walk: phase geometry, propositions as geometry, walk state, and corpus relations",
}


def load_manifests() -> dict[str, dict[str, Any]]:
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in MANIFESTS.items()}


def validate_commons_walk(manifests: dict[str, dict[str, Any]] | None = None) -> list[str]:
    manifests = manifests or load_manifests()
    problems: list[str] = []

    for name, manifest in manifests.items():
        role = manifest.get("role")
        if role != CANONICAL_ROLES[name]:
            problems.append(f"{name}: role mismatch: {role!r}")

        for field in ("entrypoints", "agentActions", "traceProtocol"):
            if not manifest.get(field):
                problems.append(f"{name}: missing executable field {field}")

        neighbors = {n.get("name"): n for n in manifest.get("semanticNeighbor", [])}
        for other, expected_role in CANONICAL_ROLES.items():
            if other not in neighbors:
                problems.append(f"{name}: missing neighbor {other}")
                continue
            got = neighbors[other].get("role")
            if got != expected_role:
                problems.append(f"{name}: neighbor {other} role mismatch: {got!r}")

    return problems


def render_traversal_plan(manifests: dict[str, dict[str, Any]] | None = None) -> str:
    manifests = manifests or load_manifests()
    lines = ["# vybn.ai commons walk", ""]
    for name in ("Origins", "Vybn-Law", "vybn-phase", "Vybn", "Him"):
        m = manifests[name]
        lines.append(f"## {name}")
        lines.append(f"role: {m['role']}")
        lines.append("entrypoints:")
        for ep in m.get("entrypoints", []):
            lines.append(f"- {ep.get('id')}: {ep.get('target')} — {ep.get('does')}")
        lines.append("actions:")
        for action in m.get("agentActions", []):
            lines.append(f"- {action.get('id')}: {action.get('does')}")
        lines.append("")
    problems = validate_commons_walk(manifests)
    lines.append("validation: " + ("OK" if not problems else "DRIFT"))
    for problem in problems:
        lines.append(f"- {problem}")
    return "\n".join(lines)


def main() -> int:
    problems = validate_commons_walk()
    print(render_traversal_plan())
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
