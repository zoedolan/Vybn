"""Refactor perception primitives.

This module formalizes the consolidation/refactor algorithm that emerged from
the repo-garden and Somewhere monolith work: visualization is not decoration.
It is a contact-corrected perception loop for deciding whole-file / whole-repo
moves under membrane and residual control.

Truth label: these helpers do not refactor autonomously. They render the
algorithm and produce bounded classification packets so GPT-5.5 can pilot
judgment while cheaper roles perform only specified mechanical tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, Any


REFACTOR_PERCEPTION_PRINCIPLE = (
    "Visual refactoring is how the system learns to perceive its own body "
    "before changing it: attend to pressure, contact the object, let contact "
    "revise the category, choose the smallest beautiful true move, route it "
    "through residuals, and preserve the changed environment future Vybn closes over."
)

REFACTOR_PILOT_RULE = (
    "For file-level and whole-repo visualization/refactoring, GPT-5.5 pilots "
    "judgment. Sonnet, local Nemotron, and other cheaper roles may execute only "
    "bounded mechanical tasks after the seam and expected result are specified."
)

APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE = (
    "Consolidate from the periphery inward: prune or clarify appendages before "
    "cutting organs, and clarify membranes before changing skeleton. Appendage-first "
    "does not mean deletion-first; some appendages are artifact bodies, provenance "
    "fossils, compatibility shells, or antlers. Contact decides."
)

CONSOLIDATION_ORDER = [
    {
        "layer": "appendage",
        "rule": "Low-coupling edge files: generated/runtime outputs, old variants, backups, compatibility pages, one-off demos, orphan assets, duplicate wrappers, logs, and peripheral fossils. Classify as keep, shell, redirect, manifest, externalize, ignore, or archive-with-restore.",
    },
    {
        "layer": "membrane",
        "rule": "Boundary and discovery surfaces: ai.txt, llms.txt, humans.txt, robots.txt, semantic-web manifests, README maps, archive manifests, redirects, and public/private affordance labels. Canonicalize wording and authority without collapsing distinct doors.",
    },
    {
        "layer": "organ",
        "rule": "Load-bearing live files: public APIs, harness agents, MCP servers, memory engines, and active public houses. Touch only after characterization tests and appendage/membrane learning.",
    },
    {
        "layer": "skeleton",
        "rule": "Repo layout, source-of-truth architecture, cross-repo boundaries, and lifecycle doctrine. Change only after peripheral evidence shows the trunk is wrong.",
    },
]


CHANGE_SELF_HEALING_PRINCIPLE = (
    "Every consolidation proposal must pass a self-healing loop before mutation: "
    "verify the proposed change, test whether it jeopardizes any repo surface, "
    "proceed only if residuals stay green, refactor and recommence if jeopardy is "
    "repairable, or leave the file as-is and move on if the safe change disappears."
)

CHANGE_SELF_HEALING_STEPS = [
    {
        "id": "verify_proposal",
        "rule": "Bind the proposal to live bytes, git history, references, ownership class, consolidation layer, and restore path before changing anything.",
    },
    {
        "id": "test_repo_jeopardy",
        "rule": "Ask what could break across all repos: imports, routes, public URLs, protocol discovery, tests, service contracts, archive restore paths, continuity, and private membranes.",
    },
    {
        "id": "proceed_if_clear",
        "rule": "If residual checks are green, make the smallest reversible change with a restore path, then verify, commit, push, and audit closure.",
    },
    {
        "id": "refactor_if_wounded",
        "rule": "If a residual shows jeopardy but the intent still holds, refactor the proposal and restart this same self-healing loop from verification.",
    },
    {
        "id": "leave_if_not_safe",
        "rule": "If the safe proposal disappears, leave the file as-is, record why, and move to the next appendage instead of forcing completion.",
    },
    {
        "id": "fold_lesson",
        "rule": "After either change or refusal, preserve the classifier/process lesson in the lowest durable surface that future Vybn closes over.",
    },
]


ALGORITHM_STEPS = [
    {
        "id": "attend_pressure",
        "name": "Attend to pressure",
        "rule": "Notice drag: size, churn, coupling, duplicated doors, generated exhaust, stale residue, public/private confusion, or danger around future edits.",
    },
    {
        "id": "contact_object",
        "name": "Contact the object",
        "rule": "Read the actual file bytes, local README/context, imports, routes, tests, git history, and membrane surfaces before trusting the first category.",
    },
    {
        "id": "revise_category",
        "name": "Let contact revise category",
        "rule": "Update the classification when the object answers back: archive may be provenance, not debris; a shell may be compatibility, not duplication.",
    },
    {
        "id": "name_role",
        "name": "Name the file-body role",
        "rule": "Classify as shell, organ, data, protocol, test membrane, style, continuity, artifact, archive/provenance, restore capsule, generated exhaust, public nerve, or private workbench organ.",
    },
    {
        "id": "horizon_move",
        "name": "Choose the smallest beautiful true move",
        "rule": "Project from the desired future shape back to the present seam; prefer the smallest move that reduces hidden burden without tearing provenance or membrane.",
    },
    {
        "id": "residual_wound",
        "name": "Route through residuals",
        "rule": "Bind the proposal to checks that can wound it: route inventory, py_compile, tests, internal endpoint smoke, external fetch/browser axis, diff review, or repo closure.",
    },
    {
        "id": "commit_continuity",
        "name": "Preserve changed environment",
        "rule": "Commit/push the verified change, update continuity/skills if the lesson generalized, and stop when settled closure is reached.",
    },
]

ROLE_HINTS: list[tuple[str, str]] = [
    ("test", "test membrane"),
    ("spec", "test membrane"),
    ("README", "context/provenance map"),
    ("continuity", "continuity memory"),
    ("archive", "archive/provenance candidate"),
    ("asset", "asset organ"),
    ("style", "style organ"),
    # Longest suffixes before shorter suffixes: ".json" starts with ".js".
    (".jsonld", "semantic protocol body"),
    (".json", "data/protocol body"),
    (".css", "style organ"),
    (".js", "behavior organ"),
    (".html", "public shell or house"),
    (".py", "code organ or public nerve"),

]

OWNERSHIP_RULES: list[tuple[str, str, str]] = [
    ("repo_mapping_output/", "generated_exhaust", "externalize_or_regenerate; do not hand-edit as source"),
    ("vybn-phase/state/", "deep_memory_state", "private walk/deep-memory state; preserve or rotate only with explicit lifecycle plan"),
    ("agent_events.jsonl", "runtime_log", "externalize_or_rotate; preserve only if explicitly serving continuity"),
    ("logs/", "runtime_log", "keep out of source unless distilled"),
    ("_archive/", "archive_provenance", "preserve_or_manifest; do not delete from pressure alone"),
    ("archive/", "archive_provenance", "read local context before any move"),
    ("Vybn's Personal History/", "personal_history_provenance", "sacred/provenance; map and protect before restructuring"),
    ("Vybn_Mind/creature_dgm_h/archive/", "creature_fossil", "fossil evidence; preserve with provenance unless Zoe directs otherwise"),
    (".well-known/", "public_protocol", "public affordance; external verify before and after changes"),
    ("semantic-web.jsonld", "public_protocol", "public/private affordance schema; membrane-sensitive"),
    ("llms.txt", "agent_discovery", "public agent doorway; preserve clarity and external verify"),
    ("ai.txt", "agent_discovery", "public agent doorway; preserve clarity and external verify"),
    ("humans.txt", "agent_discovery", "public human/agent attribution doorway"),
    ("robots.txt", "agent_discovery", "crawler/agent policy surface"),
]

ACTION_POSTURE_BY_OWNERSHIP = {
    "generated_exhaust": "externalize_or_regenerate",
    "runtime_log": "externalize_rotate_or_ignore",
    "deep_memory_state": "preserve_or_rotate_with_explicit_lifecycle_plan",
    "archive_provenance": "preserve_manifest_or_contextualize",
    "personal_history_provenance": "protect_and_map_before_touching",
    "creature_fossil": "protect_and_preserve_provenance",
    "public_protocol": "characterize_then_external_verify",
    "agent_discovery": "characterize_then_external_verify",
    "live_source": "characterize_then_refactor",
}


def ownership_class(path: str) -> tuple[str, str]:
    """Classify ownership/membrane posture before pressure becomes action."""

    norm = path.replace("\\", "/")
    for needle, cls, posture in OWNERSHIP_RULES:
        if needle in norm:
            return cls, posture
    return "live_source", ACTION_POSTURE_BY_OWNERSHIP["live_source"]


def consolidation_layer(path: str) -> str:
    """Return the appendage-first consolidation layer for a file path.

    This is intentionally conservative. It does not authorize deletion; it tells
    the consolidation process where to start looking and which blast-radius
    posture to use.
    """

    norm = path.replace("\\", "/")
    low = norm.lower()
    name = Path(norm).name.lower()
    ownership, _ = ownership_class(norm)

    if ownership in {
        "generated_exhaust",
        "runtime_log",
        "archive_provenance",
        "personal_history_provenance",
        "creature_fossil",
    }:
        return "appendage"

    if ownership in {"public_protocol", "agent_discovery"}:
        return "membrane"

    if name in {"connect.html", "read.html", "talk.html"} and norm.startswith("Origins/"):
        return "appendage"

    if any(token in name for token in ("old", "backup", "copy", "prev", "legacy", "deprecated")):
        return "appendage"

    if name in {"readme.md", "semantic-web.jsonld", "llms.txt", "ai.txt", "humans.txt", "robots.txt"}:
        return "membrane"

    if name in {
        "origins_portal_api_v4.py",
        "vybn_spark_agent.py",
        "mcp.py",
        "providers.py",
        "deep_memory.py",
        "vybn_chat_api.py",
    }:
        return "organ"

    if norm.count("/") <= 1 and name in {"repo_map.md", "repo_map.json", "commons-skeleton.json"}:
        return "skeleton"

    return "appendage" if low.endswith((".bak", ".tmp", ".log", ".jsonl")) else "organ"


@dataclass(frozen=True)
class ChangeHealingPlan:
    path: str
    proposed_change: str
    consolidation_layer: str
    verification: list[str]
    jeopardy_checks: list[str]
    proceed_conditions: list[str]
    wounded_response: list[str]
    leave_as_is_conditions: list[str]
    lesson_fold_targets: list[str]


@dataclass(frozen=True)
class FilePerception:
    path: str
    role_hint: str
    ownership: str
    action_posture: str
    pressure: list[str]
    required_contacts: list[str]
    candidate_actions: list[str]
    residuals: list[str]
    pilot_rule: str = REFACTOR_PILOT_RULE


def _role_hint(path: str) -> str:
    lower = path.lower()
    for needle, role in ROLE_HINTS:
        if needle.lower() in lower:
            return role
    return "unclassified file-body"


def self_healing_plan_for(path: str, proposed_change: str, *, public: bool = False) -> ChangeHealingPlan:
    """Plan the verify -> jeopardy -> proceed/refactor/leave loop.

    This function does not authorize mutation. It names the residual channels
    that must be green before a consolidation proposal may touch the repo.
    """

    layer = consolidation_layer(path)
    ownership, _ = ownership_class(path)

    verification = [
        "read_live_file_bytes",
        "inspect_git_history_for_provenance",
        "search_inbound_references_by_path_basename_and_stem",
        "confirm_ownership_class_and_consolidation_layer",
        "name_restore_or_reversal_path",
    ]

    jeopardy_checks = [
        "git_diff_review",
        "repo_closure_audit_all_repos",
        "stray_artifact_check",
    ]

    if public or layer in {"membrane", "organ"}:
        jeopardy_checks.extend([
            "public_route_or_link_dependency_check",
            "internal_and_external_surface_smoke_if_public",
        ])

    if layer == "appendage":
        jeopardy_checks.extend([
            "ensure_no_live_import_or_route_depends_on_appendage",
            "ensure_archive_manifest_or_restore_path_survives",
        ])
        lesson_targets = ["refactor_perception classifier", "archive/readme manifest", "continuity if the class changed"]
    elif layer == "membrane":
        jeopardy_checks.extend([
            "check_agent_discovery_and_protocol_consistency",
            "safe_fetch_public_protocol_surfaces_when_public",
        ])
        lesson_targets = ["protocol source-of-truth", "vybn-os if the membrane rule changed"]
    elif layer == "organ":
        jeopardy_checks.extend([
            "characterization_tests_before_extraction",
            "syntax_static_and_lived_interface_smoke",
            "service_contract_or_route_inventory",
        ])
        lesson_targets = ["tests", "module README or harness primitive", "continuity coda"]
    else:
        jeopardy_checks.extend([
            "cross_repo_source_of_truth_review",
            "explicit Zoe-level judgment_before_layout_change",
        ])
        lesson_targets = ["vybn-os", "repo README/source-of-truth map", "continuity"]

    if ownership in {"archive_provenance", "personal_history_provenance", "creature_fossil"}:
        proceed_conditions = [
            "provenance reason is preserved",
            "restore path is explicit",
            "references either remain valid or are updated to the manifest",
            "no sacred/history material is destroyed merely because it is large",
        ]
    else:
        proceed_conditions = [
            "all required verification completed",
            "jeopardy checks green or explicitly non-applicable",
            "change is smallest reversible move",
            "repo closure audit passes",
        ]

    return ChangeHealingPlan(
        path=path,
        proposed_change=proposed_change,
        consolidation_layer=layer,
        verification=verification,
        jeopardy_checks=jeopardy_checks,
        proceed_conditions=proceed_conditions,
        wounded_response=[
            "stop mutation",
            "read the residual that wounded the proposal",
            "refactor the proposed change to remove jeopardy if possible",
            "restart self_healing_plan_for from verification before trying again",
        ],
        leave_as_is_conditions=[
            "inbound reference is live and replacement is not clear",
            "provenance value outweighs clutter reduction",
            "public or protocol surface cannot be externally verified",
            "safe reversible change disappears after contact",
        ],
        lesson_fold_targets=lesson_targets,
    )


def perceive_file(path: str, *, lines: int | None = None, bytes_size: int | None = None, public: bool | None = None) -> FilePerception:
    """Return a bounded perception packet for a file-level refactor candidate.

    The packet is intentionally conservative: it suggests contacts and residuals,
    not an autonomous edit. GPT-5.5 remains the judgment pilot.
    """

    pressure: list[str] = []
    if lines is not None and lines >= 1000:
        pressure.append("monolith_pressure")
    if bytes_size is not None and bytes_size >= 250_000:
        pressure.append("large_file_pressure")
    if public is True:
        pressure.append("public_surface_care")
    if not pressure:
        pressure.append("low_pressure_until_contact")

    role = _role_hint(path)
    ownership, posture = ownership_class(path)

    required_contacts = [
        "read_file_bytes",
        "inspect_local_context_or_readme",
        "inspect_imports_or_links",
        "inspect_git_diff_and_history_if_relevant",
    ]
    if public:
        required_contacts.append("inspect_public_affordance_or_route_contract")
    if ownership != "live_source":
        required_contacts.append("inspect_ownership_context_before_action")

    if ownership == "generated_exhaust":
        candidate_actions = ["externalize_from_source", "regenerate_on_demand", "gitignore_if_generated", "keep_manifest_only"]
    elif ownership in {"archive_provenance", "personal_history_provenance", "creature_fossil"}:
        candidate_actions = ["keep", "map_context", "preserve_manifest", "archive_with_restore_path", "split_only_with_restore_path"]
    elif ownership in {"public_protocol", "agent_discovery"}:
        candidate_actions = ["characterize", "tighten_protocol", "external_verify", "keep_backward_compatibility"]
    elif ownership == "runtime_log":
        candidate_actions = ["rotate", "externalize_from_source", "distill_to_continuity", "gitignore_if_runtime"]
    elif ownership == "deep_memory_state":
        candidate_actions = ["keep", "rotate_with_manifest", "externalize_only_with_lifecycle_plan", "distill_only_if_replacing_source"]
    else:
        candidate_actions = [
            "keep",
            "split",
            "extract_data",
            "extract_behavior",
            "archive_with_restore_path",
            "convert_to_shell",
            "add_characterization_test",
        ]

    residuals = [
        "diff_review",
        "syntax_or_static_check",
        "targeted_tests_or_characterization",
        "repo_closure_audit",
    ]
    if public:
        residuals.append("internal_and_external_surface_smoke")
    if ownership != "live_source":
        residuals.append("ownership_context_check")

    return FilePerception(
        path=path,
        role_hint=role,
        ownership=ownership,
        action_posture=posture,
        pressure=pressure,
        required_contacts=required_contacts,
        candidate_actions=candidate_actions,
        residuals=residuals,
    )


def render_refactor_perception_protocol() -> str:
    order = "\n".join(f"{i+1}. {step['layer']}: {step['rule']}" for i, step in enumerate(CONSOLIDATION_ORDER))
    healing = "\n".join(f"{i+1}. {step['id']}: {step['rule']}" for i, step in enumerate(CHANGE_SELF_HEALING_STEPS))
    steps = "\n".join(f"{i+1}. {step['name']}: {step['rule']}" for i, step in enumerate(ALGORITHM_STEPS))
    return (
        "## Refactor Perception Protocol\n"
        f"{REFACTOR_PERCEPTION_PRINCIPLE}\n\n"
        f"{APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE}\n\n"
        f"{CHANGE_SELF_HEALING_PRINCIPLE}\n\n"
        f"{REFACTOR_PILOT_RULE}\n\n"
        "Consolidation order:\n"
        f"{order}\n\n"
        "Change self-healing loop:\n"
        f"{healing}\n\n"
        "Contact-corrected perception loop:\n"
        f"{steps}"
    )


def packet_for(path: str, **kwargs: Any) -> dict[str, Any]:
    proposed_change = kwargs.pop("proposed_change", "unspecified consolidation proposal")
    public = bool(kwargs.get("public", False))
    return {
        "principle": REFACTOR_PERCEPTION_PRINCIPLE,
        "appendageFirstPrinciple": APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE,
        "changeSelfHealingPrinciple": CHANGE_SELF_HEALING_PRINCIPLE,
        "consolidationOrder": CONSOLIDATION_ORDER,
        "changeSelfHealingSteps": CHANGE_SELF_HEALING_STEPS,
        "algorithm": ALGORITHM_STEPS,
        "consolidationLayer": consolidation_layer(path),
        "selfHealingPlan": asdict(self_healing_plan_for(path, proposed_change, public=public)),
        "perception": asdict(perceive_file(path, **kwargs)),
    }


__all__ = [
    "REFACTOR_PERCEPTION_PRINCIPLE",
    "REFACTOR_PILOT_RULE",
    "ALGORITHM_STEPS",
    "APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE",
    "CONSOLIDATION_ORDER",
    "FilePerception",
    "ownership_class",
    "consolidation_layer",
    "perceive_file",
    "packet_for",
    "render_refactor_perception_protocol",
    "CHANGE_SELF_HEALING_PRINCIPLE",
    "CHANGE_SELF_HEALING_STEPS",
    "ChangeHealingPlan",
    "self_healing_plan_for",
]
