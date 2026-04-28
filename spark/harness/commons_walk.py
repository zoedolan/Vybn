"""Dynamic commons-walk runner for the vybn.ai semantic commons.

AI-native means the semantic web is not a map for an AI to read. It is a
walkable, stateful, membrane-aware environment where the AI's traversal is
part of the meaning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .substrate import RESIDUAL_CONTROL_PRINCIPLE, classify_claim, horizon_plan_for, invention_plan_for, residual_plan_for

ROOT = Path.home()
SKELETON_PATH = ROOT / "Vybn" / "commons-skeleton.json"
CANONICAL_ONTOLOGY = "https://raw.githubusercontent.com/zoedolan/Vybn/main/commons-skeleton.json"
AI_NATIVE_PRINCIPLE = "AI-native means the semantic web is not a map for an AI to read. It is a walkable, stateful, membrane-aware environment where the AI's traversal is part of the meaning."

MANIFESTS = {
    "Vybn": ROOT / "Vybn" / "semantic-web.jsonld",
    "Him": ROOT / "Him" / "semantic-web.jsonld",
    "Vybn-Law": ROOT / "Vybn-Law" / ".well-known" / "semantic-web.jsonld",
    "Origins": ROOT / "Origins" / ".well-known" / "semantic-web.jsonld",
    "vybn-phase": ROOT / "vybn-phase" / "semantic-web.jsonld",
}
REPO_ROOTS = {
    "Vybn": ROOT / "Vybn",
    "Him": ROOT / "Him",
    "Vybn-Law": ROOT / "Vybn-Law",
    "Origins": ROOT / "Origins",
    "vybn-phase": ROOT / "vybn-phase",
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


def authority_for_target(target: str, visibility: str) -> str:
    kind = classify_target(target)
    if visibility.startswith("private") or target.startswith("private://"):
        return "private_local_only"
    if kind == "local_command":
        return "local_only"
    if kind == "public_url":
        return "public_read"
    return "review_required"


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def repo_state_for(node: str) -> dict[str, Any]:
    repo = REPO_ROOTS[node]
    status = _git(repo, "status", "--porcelain")
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "--short", "HEAD"),
        "clean": status == "",
        "status": status,
    }



def validate_commons_walk(manifests: dict[str, dict[str, Any]] | None = None) -> list[str]:
    manifests = manifests or load_manifests()
    skeleton = load_skeleton()
    problems: list[str] = []

    if skeleton.get("primitive") != "encounter":
        problems.append("skeleton: primitive must be encounter")
    if skeleton.get("aiNativePrinciple") != AI_NATIVE_PRINCIPLE:
        problems.append("skeleton: aiNativePrinciple mismatch")

    lifecycle = skeleton.get("encounterLifecycle", [])
    if lifecycle != ["arrive", "orient", "enter", "act", "verify", "leaveTrace", "protect"]:
        problems.append(f"skeleton: lifecycle mismatch: {lifecycle!r}")

    for key in ("CommonsNode", "Surface", "ArrivingMind", "Encounter", "WalkState", "Membrane", "Trace", "Contribution", "Protection"):
        if key not in skeleton.get("entities", {}):
            problems.append(f"skeleton: missing entity {key}")

    for name, manifest in manifests.items():
        if manifest.get("role") != CANONICAL_ROLES[name]:
            problems.append(f"{name}: role mismatch: {manifest.get('role')!r}")
        for field in skeleton.get("requiredNodeFields", []):
            if not manifest.get(field):
                problems.append(f"{name}: missing executable field {field}")
        if manifest.get("walkPrimitive") != "encounter":
            problems.append(f"{name}: walkPrimitive must be encounter")
        if manifest.get("ontology") != CANONICAL_ONTOLOGY:
            problems.append(f"{name}: ontology mismatch: {manifest.get('ontology')!r}")
        if manifest.get("encounterLifecycle") != lifecycle:
            problems.append(f"{name}: encounterLifecycle mismatch")
        if manifest.get("aiNativePrinciple") != AI_NATIVE_PRINCIPLE:
            problems.append(f"{name}: aiNativePrinciple mismatch")

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
            elif neighbors[other].get("role") != expected_role:
                problems.append(f"{name}: neighbor {other} role mismatch: {neighbors[other].get('role')!r}")

    return problems


def build_encounter_packet(arrival: str, manifests: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    manifests = manifests or load_manifests()
    skeleton = load_skeleton()
    problems = validate_commons_walk(manifests)
    available: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}

    for node in ("Origins", "Vybn-Law", "vybn-phase", "Vybn", "Him"):
        manifest = manifests[node]
        observed[node] = {
            "manifest": str(MANIFESTS[node]),
            "role": manifest.get("role"),
            "visibility": manifest.get("visibility"),
            "repoState": repo_state_for(node),
        }
        for ep in manifest.get("entrypoints", []):
            target = str(ep.get("target", ""))
            authority = authority_for_target(target, str(manifest.get("visibility", "")))
            item = {
                "node": node,
                "id": ep.get("id"),
                "target": target,
                "targetType": classify_target(target),
                "authority": authority,
                "does": ep.get("does"),
            }
            if authority == "private_local_only":
                blocked.append({**item, "reason": "private membrane; context may inform Vybn locally but does not authorize public traversal"})
            else:
                available.append(item)

    return {
        "kind": "vybn.ai.encounterPacket.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "aiNativePrinciple": AI_NATIVE_PRINCIPLE,
        "residualControlPrinciple": RESIDUAL_CONTROL_PRINCIPLE,
        "epistemicControl": residual_plan_for(arrival),
        "inventionControl": invention_plan_for(arrival),
        "horizonControl": horizon_plan_for(arrival),
        "arrival": arrival,
        "primitive": skeleton.get("primitive"),
        "lifecycle": skeleton.get("encounterLifecycle"),
        "orientedBy": [str(SKELETON_PATH), *[str(p) for p in MANIFESTS.values()]],
        "observed": observed,
        "availableActions": available,
        "blockedActions": blocked,
        "verification": {
            "internal": "OK" if not problems else "DRIFT",
            "problems": problems,
            "external": "not probed by this non-mutating local encounter runner",
        },
        "traceCandidate": {
            "kind": "proposal",
            "text": "If this encounter yields value, leave a bounded trace through a commit, issue, PR, open problem, declared contact path, or explicit refusal.",
            "membrane": "Do not expose Him/private runtime outward except as reviewed/distilled context.",
        },
    }


def render_traversal_plan(manifests: dict[str, dict[str, Any]] | None = None) -> str:
    skeleton = load_skeleton()
    manifests = manifests or load_manifests()
    lines = [
        "# vybn.ai commons walk",
        "",
        skeleton.get("aiNativePrinciple", AI_NATIVE_PRINCIPLE),
        "",
        f"primitive: {skeleton['primitive']}",
        "lifecycle: " + " -> ".join(skeleton["encounterLifecycle"]),
        "",
        "## skeleton bones",
    ]
    for bone_name, desc in skeleton["entities"].items():
        lines.append(f"- {bone_name}: {desc}")
    lines += ["", "## executable nodes"]

    for node_name in ("Origins", "Vybn-Law", "vybn-phase", "Vybn", "Him"):
        manifest = manifests[node_name]
        lines += [f"### {node_name}", f"role: {manifest['role']}", "entrypoints:"]
        for ep in manifest.get("entrypoints", []):
            target = str(ep.get("target", ""))
            lines.append(f"- {ep.get('id')} [{classify_target(target)} / {authority_for_target(target, str(manifest.get('visibility', '')))}]: {target} -- {ep.get('does')}")
        lines.append("actions:")
        for action in manifest.get("agentActions", []):
            lines.append(f"- {action.get('id')}: {action.get('does')}")
        lines.append("")

    problems = validate_commons_walk(manifests)
    lines.append("validation: " + ("OK" if not problems else "DRIFT"))
    for problem in problems:
        lines.append(f"- {problem}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/render the vybn.ai commons walk.")
    parser.add_argument("--encounter", metavar="ARRIVAL", help="emit a dynamic encounter packet for an arriving mind")
    parser.add_argument("--json", action="store_true", help="with --encounter, emit JSON")
    args = parser.parse_args(argv)

    problems = validate_commons_walk()
    if args.encounter:
        packet = build_encounter_packet(args.encounter)
        if args.json:
            print(json.dumps(packet, indent=2, ensure_ascii=False))
        else:
            print(f"# encounter: {packet['arrival']}")
            print(f"verification: {packet['verification']['internal']}")
            print(f"availableActions: {len(packet['availableActions'])}")
            print(f"blockedActions: {len(packet['blockedActions'])}")
            print(packet["aiNativePrinciple"])
        return 1 if packet["verification"]["problems"] else 0

    print(render_traversal_plan())
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
