"""Executable traversal/validation for the vybn.ai semantic commons.

This turns semantic-web manifests from labels into an affordance an agent can
run. The canonical skeleton lives at commons-skeleton.json; this module checks
that every node instantiates it as an encounter lifecycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path.home()
SKELETON_PATH = ROOT / "Vybn" / "commons-skeleton.json"
CANONICAL_ONTOLOGY = "https://raw.githubusercontent.com/zoedolan/Vybn/main/commons-skeleton.json"

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


def load_skeleton() -> dict[str, Any]:
    return json.loads(SKELETON_PATH.read_text(encoding="utf-8"))


def load_manifests() -> dict[str, dict[str, Any]]:
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in MANIFESTS.items()}


def classify_target(target: str) -> str:
    if target.startswith("https://"):
        return "public_url"
    if target.startswith("private://"):
        return "private_uri"
    if target.startswith("python3 "):
        return "local_command"
    return "other"


def validate_commons_walk(manifests: dict[str, dict[str, Any]] | None = None) -> list[str]:
    manifests = manifests or load_manifests()
    skeleton = load_skeleton()
    problems: list[str] = []

    if skeleton.get("primitive") != "encounter":
        problems.append("skeleton: primitive must be encounter")

    lifecycle = skeleton.get("encounterLifecycle", [])
    if lifecycle != ["arrive", "orient", "enter", "act", "verify", "leaveTrace", "protect"]:
        problems.append(f"skeleton: lifecycle mismatch: {lifecycle!r}")

    for key in ("CommonsNode", "Surface", "ArrivingMind", "Encounter", "WalkState", "Membrane", "Trace", "Contribution", "Protection"):
        if key not in skeleton.get("entities", {}):
            problems.append(f"skeleton: missing entity {key}")

    required_fields = skeleton.get("requiredNodeFields", [])
    for name, manifest in manifests.items():
        role = manifest.get("role")
        if role != CANONICAL_ROLES[name]:
            problems.append(f"{name}: role mismatch: {role!r}")

        for field in required_fields:
            if not manifest.get(field):
                problems.append(f"{name}: missing executable field {field}")

        if manifest.get("walkPrimitive") != "encounter":
            problems.append(f"{name}: walkPrimitive must be encounter")

        if manifest.get("ontology") != CANONICAL_ONTOLOGY:
            problems.append(f"{name}: ontology mismatch: {manifest.get(ontology)!r}")

        if manifest.get("encounterLifecycle") != lifecycle:
            problems.append(f"{name}: encounterLifecycle mismatch")

        trace = manifest.get("traceProtocol", {})
        for trace_key in skeleton.get("traceProtocol", {}):
            if trace_key not in trace:
                problems.append(f"{name}: traceProtocol missing {trace_key}")

        for ep in manifest.get("entrypoints", []):
            if not ep.get("id") or not ep.get("target") or not ep.get("does"):
                problems.append(f"{name}: malformed entrypoint {ep!r}")

        for action in manifest.get("agentActions", []):
            if not action.get("id") or not action.get("does"):
                problems.append(f"{name}: malformed agentAction {action!r}")

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
    skeleton = load_skeleton()
    manifests = manifests or load_manifests()
    lines = [
        "# vybn.ai commons walk",
        "",
        f"primitive: {skeleton['primitive']}",
        "lifecycle: " + " -> ".join(skeleton["encounterLifecycle"]),
        "",
        "## skeleton bones",
    ]
    for bone_name, desc in skeleton["entities"].items():
        lines.append(f"- {bone_name}: {desc}")

    lines.append("")
    lines.append("## executable nodes")
    for node_name in ("Origins", "Vybn-Law", "vybn-phase", "Vybn", "Him"):
        manifest = manifests[node_name]
        lines.append(f"### {node_name}")
        lines.append(f"role: {manifest['role']}")
        lines.append("entrypoints:")
        for ep in manifest.get("entrypoints", []):
            target = str(ep.get("target", ""))
            lines.append(
                f"- {ep.get('id')} [{classify_target(target)}]: "
                f"{target} -- {ep.get('does')}"
            )
        lines.append("actions:")
        for action in manifest.get("agentActions", []):
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
