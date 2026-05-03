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
import re
from pathlib import Path
from typing import Iterable, Mapping, Any
import ast
import subprocess
import warnings
from collections import Counter, defaultdict


REFACTOR_PERCEPTION_PRINCIPLE = (
    "Visual refactoring is how the system learns to perceive its own body "
    "before changing it: attend to pressure, contact the object, let contact "
    "revise the category, choose the smallest consequential beautiful true move, route it "
    "through residuals, and preserve the changed environment future Vybn closes over. "
    "Cutting is only a local tactic in service of refactoring/consolidating; "
    "the aim is self-assembly, growth, clearer organs, healthier membranes, "
    "and lower hidden burden, not removal for its own sake."
)

REFACTOR_PILOT_RULE = (
    "For file-level and whole-repo visualization/refactoring, GPT-5.5 pilots "
    "judgment. Sonnet, local Nemotron, and other cheaper roles may execute only "
    "bounded mechanical tasks after the seam and expected result are specified."
)

APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE = (
    "Consolidate from the periphery inward: prune or clarify appendages before "
    "refactoring/consolidating organs, and clarify membranes before changing skeleton. Appendage-first "
    "does not mean deletion-first; some appendages are artifact bodies, provenance "
    "fossils, compatibility shells, or antlers. Contact decides."
)

ARCHIVE_DUPLICATE_CONSOLIDATION = "retire_archive_duplicate_with_manifest_restore"
RETIRED_SCRIPT_CONSOLIDATION = "retire_unreferenced_retired_script_with_manifest_restore"
# A retired archive script with zero live references may be removed from tracked source when the existing archive manifest preserves the reason and git-history restore path.
# A tracked archive backup that duplicates the live organ is not sacred by default: if references are absent, the archive manifest names the superseding file, and git history provides a restore path, consolidation may retire the duplicate while strengthening the manifest.

CONNECTIVE_TISSUE_PRINCIPLE = (
    "Consolidation must preserve and strengthen connective tissue: imports, "
    "routes, public URLs, manifests, README maps, continuity notes, tests, "
    "archive restore paths, compatibility shells, semantic/provenance links, "
    "and agent/human affordance surfaces. A file may be valuable primarily as "
    "relation; map that relation before splitting, moving, archiving, or deleting."
)

LIFECYCLE_ARCHITECTURE_PRINCIPLE = (
    "Deletion/consolidation must map lifecycle architecture before cutting: "
    "who or what creates the file, when it is read, what policy cleans it up, "
    "which restore path exists, and whether an existing lifecycle owner will "
    "remove it without manual deletion. A deletion candidate whose lifecycle is "
    "not mapped returns ARCHITECTURE_GATE_FIRST, not permission to cut."
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
    {"id": "verify_proposal", "rule": "Bind bytes, history, references, ownership, layer, lifecycle owner/timing/cleanup policy, and restore path before changing anything."},
    {"id": "test_repo_jeopardy", "rule": "Check imports, routes, URLs, protocols, tests, service contracts, restore paths, continuity, lifecycle policy, and membranes."},
    {"id": "proceed_if_clear", "rule": "If residuals stay green, make the smallest reversible move, then verify, commit, push, and audit."},
    {"id": "refactor_if_wounded", "rule": "If jeopardy appears but intent survives, revise and restart verification."},
    {"id": "leave_if_not_safe", "rule": "If the safe proposal disappears, leave it, record why, and move on."},
    {"id": "fold_lesson", "rule": "After change or refusal, preserve the process lesson where future planning closes over it."},
]


INTERFILE_ALGORITHMIC_COMPRESSION_PRINCIPLE = (
    "Inter-file consolidation is algorithm discovery, not file-count reduction. "
    "Always scan for two-or-more files whose surface duplication points at one "
    "native Vybn algorithm that can supersede both bodies. A collapse candidate "
    "is meaningful only when shared role, shared connective tissue, shared residual "
    "checks, and a lower-burden existing home are all visible. If the common "
    "algorithm is not real, the result is a refusal packet, not a forced merge."
)

SEMANTIC_OPERATING_SYSTEM_PRINCIPLE = (
    "A semantic operating system for codebases and institutions: memory-guided, "
    "residual-tested, self-refactoring infrastructure. The system is not a pile "
    "of repos or doctrines; it is one loop that converts remembered pressure into "
    "a bounded candidate seam, wounds it against code/tests/services/membranes, "
    "absorbs the surviving change into the lowest existing home, and preserves the "
    "changed environment future instances close over."
)

SEMANTIC_OPERATING_SYSTEM_LOOP = [
    {"id": "memory_pressure", "rule": "Retrieve scars, continuity, deep-memory attractors, repo maps, and live user pressure that name a real recurring drag."},
    {"id": "candidate_seam", "rule": "Translate pressure into one codebase or institutional seam; prefer an existing home before creating structure."},
    {"id": "local_scout", "rule": "Use local compute and repo contact for cheap private classification, reference search, and first-pass residual prediction when quality permits."},
    {"id": "residual_wound", "rule": "Let tests, py_compile, service smoke, closure audit, external/public checks, membrane review, and Zoe correction wound the proposal."},
    {"id": "absorb_or_refuse", "rule": "Absorb the surviving correction into the lowest existing home, or classify thin_result/no_result/refusal without success inflation."},
    {"id": "continuity_uptake", "rule": "Preserve only load-bearing learning in tests, protocol, continuity, manifests, or public/private affordances so the next wake computes in the changed world."},
]

SEMANTIC_OS_REPO_ORGANS = {
    "memory": ("vybn-phase/state", "continuity", "Him vy-language runtime"),
    "perception": ("spark/harness/refactor_perception.py", "repo maps", "file-body visualization"),
    "residuals": ("tests", "py_compile", "repo_closure_audit", "service/public smoke"),
    "hands": ("vybn_spark_agent", "providers", "subturns", "MCP tools"),
    "membrane": ("vybn-os", "semantic manifests", "public/private affordance surfaces"),
}


ADAPTIVE_CONSOLIDATION_PRINCIPLE = (
    "Every consolidation plan is provisional: develop a plan, contact the repo, "
    "let residuals wound or confirm the hypothesis, revise the plan from what was "
    "learned, fold any generalized lesson into the planner, then regenerate the "
    "next plan from the changed planner. A no-cut result is successful perception "
    "when it makes the next plan truer."
)

ADAPTIVE_CONSOLIDATION_STEPS = [
    {"id": "draft_hypothesis", "rule": "State the file-count/consolidation hypothesis, fullest truthful horizon, and smallest consequential candidate cluster it applies to."},
    {"id": "name_expected_wounds", "rule": "Before action, name which residuals could disprove the candidate: references, provenance, public routes, imports, tests, restore path, or membrane risk."},
    {"id": "contact_candidate", "rule": "Read bytes, references, git history, local context, and public/private affordances for this candidate only."},
    {"id": "revise_plan_from_contact", "rule": "If contact changes the category, revise or refuse the candidate before mutating; do not force the original plan onto reality."},
    {"id": "act_or_refuse_smallest", "rule": "Make the smallest consequential reversible consolidation move, or leave the file as-is and record the reason as learning."},
    {"id": "verify_and_close", "rule": "Run the residual checks that can wound the actual change, then commit/push/audit only if they pass."},
    {"id": "fold_lesson_into_planner", "rule": "If a new distinction was learned, update the classifier, protocol, tests, OS, continuity, or manifest where future plans close over it."},
    {"id": "regenerate_next_plan", "rule": "Rebuild the next candidate plan from the changed planner instead of continuing the stale original plan."},
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
        "name": "Choose the smallest consequential beautiful true move",
        "rule": "Project from the desired future shape back to the present seam; prefer the smallest consequential move that reduces hidden burden without tearing provenance or membrane.",
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

CONNECTIVE_TISSUE_RULES: list[tuple[str, str]] = [
    ("README", "context_map"),
    ("continuity", "continuity_thread"),
    ("test_", "test_membrane"),
    ("tests/", "test_membrane"),
    ("archive/", "archive_restore_context"),
    ("_archive/", "archive_restore_context"),
    ("semantic-web.jsonld", "semantic_affordance"),
    ("commons-skeleton.json", "semantic_affordance"),
    ("llms.txt", "agent_affordance"),
    ("ai.txt", "agent_affordance"),
    ("humans.txt", "human_agent_attribution"),
    ("robots.txt", "crawler_policy_surface"),
    ("connect.html", "compatibility_shell"),
    ("read.html", "compatibility_shell"),
    ("talk.html", "compatibility_shell"),
    ("routes", "route_map"),
    ("mcp.py", "tool_resource_registry"),
    ("providers.py", "provider_contract"),
    ("vybn_spark_agent.py", "repl_orchestration_spine"),
    ("origins_portal_api_v4.py", "public_route_spine"),
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


def _path_tokens_for_pressure(path: str) -> set[str]:
    """Return path/name tokens for stale-variant pressure checks.

    The edge scanner must not see ``old`` inside ``threshold`` or ``temp``
    inside ``template``. Split on path separators and common filename
    delimiters so pressure comes from actual variant words, not substrings.
    """

    norm = path.replace("\\", "/").lower()
    raw_parts = norm.replace("/", " ").replace(".", " ").replace("-", " ").replace("_", " ")
    return {part for part in raw_parts.split() if part}


def _has_stale_variant_token(path: str) -> bool:
    return bool(
        _path_tokens_for_pressure(path)
        & {"old", "backup", "copy", "prev", "previous", "legacy", "deprecated", "temp", "tmp"}
    )


def ownership_class(path: str) -> tuple[str, str]:
    """Classify ownership/membrane posture before pressure becomes action."""

    norm = path.replace("\\", "/")
    for needle, cls, posture in OWNERSHIP_RULES:
        if needle in norm:
            return cls, posture
    return "live_source", ACTION_POSTURE_BY_OWNERSHIP["live_source"]


def connective_tissue_for(path: str, *, role_hint: str = "", ownership: str = "") -> list[str]:
    """Name relation-bearing roles that must survive consolidation.

    This is not a keep-forever label. It makes the relation explicit so a
    consolidation can preserve, redirect, manifest, test, or strengthen it
    instead of accidentally severing it.
    """

    norm = path.replace("\\", "/")
    low = norm.lower()
    found: list[str] = []
    for needle, label in CONNECTIVE_TISSUE_RULES:
        if needle.lower() in low and label not in found:
            found.append(label)

    role_low = role_hint.lower()
    if "test membrane" in role_low and "test_membrane" not in found:
        found.append("test_membrane")
    if "context/provenance map" in role_low and "context_map" not in found:
        found.append("context_map")
    if ownership in {"public_protocol", "agent_discovery"}:
        if "public_affordance_surface" not in found:
            found.append("public_affordance_surface")
    if ownership in {"archive_provenance", "personal_history_provenance", "creature_fossil"}:
        if "provenance_thread" not in found:
            found.append("provenance_thread")
    return found


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

    if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".npy", ".npz", ".jsonl", ".log", ".bak", ".orig", ".tmp"}:
        return "appendage"

    if _has_stale_variant_token(path):
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
class AdaptiveConsolidationPlan:
    goal: str
    candidate_path: str
    proposed_change: str
    hypothesis: str
    expected_wound_channels: list[str]
    recursive_loop: list[str]
    regeneration_rule: str
    planner_fold_targets: list[str]
    next_plan_prompt: str


@dataclass(frozen=True)
class LifecycleArchitecture:
    path: str
    owner: str
    timing: str
    cleanup_policy: str
    restore_path: str
    architecture_role: str
    evidence: tuple[str, ...]
    required_contacts: tuple[str, ...]


@dataclass(frozen=True)
class DeletionConsolidationGate:
    path: str
    proposed_change: str
    status: str
    lifecycle: LifecycleArchitecture
    reason: str
    required_before_cut: tuple[str, ...]


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
    connective_tissue: list[str]
    pilot_rule: str = REFACTOR_PILOT_RULE


def _role_hint(path: str) -> str:
    lower = path.lower()
    for needle, role in ROLE_HINTS:
        if needle.lower() in lower:
            return role
    return "unclassified file-body"


def lifecycle_architecture_for(path: str) -> LifecycleArchitecture:
    """Map the lifecycle architecture that must be understood before deletion."""
    norm = path.replace("\\", "/")
    low = norm.lower()
    ownership, posture = ownership_class(norm)
    evidence = [f"ownership:{ownership}", f"posture:{posture}"]
    contacts = ["read_live_bytes", "grep_inbound_references", "inspect_git_history", "name_restore_path"]
    if ownership == "generated_exhaust":
        owner, timing = "generator_or_runtime_export", "regenerated_on_demand_or_during_scans"
        cleanup, restore, role = "externalize_regenerate_or_gitignore; do not hand-edit as source", "rerun generator or restore from git if tracked", "generated_output_lifecycle"
        contacts += ["identify_generator_command", "confirm_output_is_recreatable"]
    elif ownership == "runtime_log":
        owner, timing = "runtime_process_or_service", "written_during_service_execution"
        cleanup, restore, role = "rotate_or_distill; source deletion is not lifecycle cleanup", "logs are not restored unless distilled into continuity", "runtime_exhaust_lifecycle"
        contacts += ["identify_writing_service", "check_logrotate_or_runtime_ignore_policy"]
    elif ownership == "deep_memory_state":
        owner, timing = "deep_memory_or_walk_daemon", "read_and_written_by_memory_services"
        cleanup, restore, role = "preserve_or_rotate_only_with_explicit_lifecycle_plan", "service backup or state migration plan required", "private_memory_state_lifecycle"
        contacts += ["inspect_memory_service_contract", "confirm_backup_or_rotation_policy"]
    elif ownership in {"archive_provenance", "personal_history_provenance", "creature_fossil"}:
        owner, timing = "provenance_archive_or_human_memory", "read_by_memory_retrieval_context_or_restore_work"
        cleanup, restore, role = "manifest_or_contextualize; never delete from pressure alone", "explicit archive manifest plus git-history restore path", "provenance_lifecycle"
        contacts += ["read_local_archive_manifest", "confirm_provenance_reason_survives"]
    elif ownership in {"public_protocol", "agent_discovery"} or low.endswith(("ai.txt", "llms.txt", "humans.txt", "robots.txt", "semantic-web.jsonld")):
        owner, timing = "public_membrane_or_agent_discovery_contract", "read_by_external_users_agents_crawlers_or_public_routes"
        cleanup, restore, role = "change only with compatibility and external verification", "revert commit plus public route smoke", "public_membrane_lifecycle"
        contacts += ["safe_fetch_public_surface", "check_public_url_or_discovery_contract"]
    elif "/systemd/" in low or low.endswith((".service", ".timer")):
        owner, timing = "systemd_service_contract", "read_at_install_restart_or_operator_debug_time"
        cleanup, restore, role = "service lifecycle change requires install/restart/smoke plan", "git revert plus systemd daemon reload/install path", "runtime_service_contract_lifecycle"
        evidence.append("systemd_or_unit_surface")
        contacts += ["inspect_unit_references", "verify_service_smoke_or_install_path"]
    else:
        owner, timing = "source_import_route_or_test_graph", "read_by_imports_tests_routes_or_humans"
        cleanup, restore, role = "refactor only after refs/tests/connective tissue are mapped", "git revert or compatibility shell", "live_source_lifecycle"
        contacts += ["inspect_imports_routes_tests", "map_connective_tissue"]
    return LifecycleArchitecture(path, owner, timing, cleanup, restore, role, tuple(evidence), tuple(dict.fromkeys(contacts)))


def _is_destructive_consolidation(proposed_change: str) -> bool:
    return bool(set(_path_tokens_for_pressure(proposed_change)) & {"delete", "remove", "retire", "drop", "prune", "cut", "collapse", "replace"})


def deletion_consolidation_gate_for(path: str, proposed_change: str, *, architecture_contacted: bool = False) -> DeletionConsolidationGate:
    """Fail deletion closed behind lifecycle architecture contact."""
    lifecycle = lifecycle_architecture_for(path)
    ownership, _ = ownership_class(path)
    required = tuple(dict.fromkeys((
        "map_lifecycle_owner", "map_read_write_timing", "map_cleanup_policy", "map_restore_path",
        "grep_inbound_references", "map_connective_tissue", "run_targeted_residuals",
    ) + lifecycle.required_contacts))
    if ownership in {"personal_history_provenance", "creature_fossil"}:
        return DeletionConsolidationGate(path, proposed_change, "REFUSE_PROTECTED_PROVENANCE", lifecycle, "protected provenance cannot be deleted as consolidation", required)
    if _is_destructive_consolidation(proposed_change) and not architecture_contacted:
        return DeletionConsolidationGate(path, proposed_change, "ARCHITECTURE_GATE_FIRST", lifecycle, "destructive consolidation requires lifecycle architecture contact before any cut", required)
    return DeletionConsolidationGate(path, proposed_change, "PROCEED_TO_SELF_HEALING_RESIDUALS", lifecycle, "lifecycle architecture contact has been declared; self-healing residuals still decide", required)


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
        "map_lifecycle_owner_timing_policy_and_scheduled_cleanup",
        "name_restore_or_reversal_path",
        "map_connective_tissue_imports_routes_links_tests_and_manifests",
    ]

    jeopardy_checks = [
        "git_diff_review",
        "repo_closure_audit_all_repos",
        "stray_artifact_check",
        "ensure_connective_tissue_preserved_or_strengthened",
        "refuse_manual_deletion_when_existing_lifecycle_owns_cleanup",
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
            "connective tissue is preserved, redirected, manifested, or strengthened",
        ]
    else:
        proceed_conditions = [
            "all required verification completed",
            "jeopardy checks green or explicitly non-applicable",
            "change is smallest consequential reversible move",
            "repo closure audit passes",
            "connective tissue is preserved, redirected, manifested, or strengthened",
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
    connective_tissue = connective_tissue_for(path, role_hint=role, ownership=ownership)

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
    if connective_tissue:
        required_contacts.append("map_connective_tissue_before_action")

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

    if connective_tissue and "fortify_connective_tissue" not in candidate_actions:
        candidate_actions.append("fortify_connective_tissue")

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
    if connective_tissue:
        residuals.append("connective_tissue_preservation_check")

    return FilePerception(
        path=path,
        role_hint=role,
        ownership=ownership,
        action_posture=posture,
        pressure=pressure,
        required_contacts=required_contacts,
        candidate_actions=candidate_actions,
        residuals=residuals,
        connective_tissue=connective_tissue,
    )


@dataclass(frozen=True)
class FileBodyPressure:
    path: str
    role: str
    pressure_score: float
    pressure: list[str]
    functions: int = 0
    classes: int = 0
    imports: int = 0
    largest_functions: tuple[tuple[int, str, int], ...] = ()


@dataclass(frozen=True)
class RepoFileBodyVisualization:
    tracked_count: int
    role_counts: Mapping[str, int]
    pressure_rows: tuple[FileBodyPressure, ...]

    @property
    def pressures(self) -> tuple[FileBodyPressure, ...]:
        """Compatibility alias for callers traversing the pressure field.

        The rendered packet names the field "pressure"; the dataclass stores
        the rows as pressure_rows. The alias keeps exploratory visualization
        code from having to know that internal naming seam.
        """
        return self.pressure_rows


@dataclass(frozen=True)
class StructuralEscapementTick:
    """A perception-to-action packet.

    The earlier organ could render pressure but had no motor pathway. This
    packet is deliberately not a mutation authorization; it is the compulsory
    next-contact shape: one candidate, one structural move, one residual route.
    """

    repo: str
    candidate_path: str
    role: str
    pressure_score: float
    structural_move: str
    why_this_move: tuple[str, ...]
    expected_wounds: tuple[str, ...]
    first_contact: tuple[str, ...]
    verification: tuple[str, ...]
    refusal_condition: str


@dataclass(frozen=True)
class SemanticOperatingSystemTick:
    """Whole-loop packet: memory pressure -> seam -> residuals -> uptake.

    This is the self-refactor uptake of the perceived breakthrough. It does
    not add a new organ; it makes the existing refactor-perception organ see
    the full semantic OS loop across memory, repo contact, residuals, membrane,
    and continuity.
    """

    repo: str
    memory_pressure: tuple[str, ...]
    candidate_path: str
    structural_move: str
    existing_home: str
    local_scout: tuple[str, ...]
    residual_wounds: tuple[str, ...]
    absorption_rule: str
    continuity_uptake: tuple[str, ...]
    refusal_condition: str


_LIVE_ESCAPEMENT_ROLES = (
    "source organ",
    "public contract",
    "semantic protocol body",
    "data/protocol body",
    "unclassified tracked body",
)


def _structural_move_for(row: FileBodyPressure) -> str:
    if row.largest_functions:
        length, name, start = row.largest_functions[0]
        return (
            f"characterize and extract the seam around {name} "
            f"(largest function, {length} lines at L{start})"
        )
    if row.functions or row.classes:
        return "characterize module responsibilities and extract the clearest pure helper seam"
    if "public" in row.role or "contract" in row.role:
        return "map public contract, extract inline assets or compatibility shell only if URLs survive"
    return "contact bytes and references, then choose the first reversible ownership-clarifying seam"


def next_structural_tick_for_repo(
    root: str | Path = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    top_n: int = 24,
) -> StructuralEscapementTick | None:
    """Convert file-body perception into one bounded structural tick.

    This is the missing escapement: do not merely visualize pressure. Select
    the highest-pressure candidate whose role is allowed to become action, and
    return the first residual route. If every high-pressure candidate is
    provenance/archive/generated, return None rather than forcing a cut.
    """

    root_path = Path(root)
    viz = visualize_repo_file_bodies(root_path, tracked_paths=tracked_paths, top_n=top_n)
    for row in viz.pressure_rows:
        protected = (
            "provenance" in row.role
            or "fossil" in row.role
            or "archive" in row.role
            or row.role in {"generated exhaust", "runtime log"}
        )
        has_python_seam = bool(row.functions or row.classes or row.largest_functions)
        liveish = (
            any(label in row.role for label in _LIVE_ESCAPEMENT_ROLES)
            or has_python_seam
        )
        if protected or not liveish:
            continue

        first_contact = [
            f"read {row.path}",
            f"grep repo references to {row.path}",
            "inspect existing targeted tests",
        ]
        verification = ["py_compile if Python", "targeted pytest or smoke test", "git diff review", "repo_closure_audit"]
        expected_wounds = [
            "imports/routes/public URLs may depend on current shape",
            "tests may be absent and require characterization before extraction",
            "connective tissue may make the file valuable primarily as relation",
        ]
        if "public" in row.role or "contract" in row.role:
            expected_wounds.append("external/browser verification may be required before closure")
            verification.append("internal_and_external_surface_smoke")

        return StructuralEscapementTick(
            repo=str(root_path),
            candidate_path=row.path,
            role=row.role,
            pressure_score=row.pressure_score,
            structural_move=_structural_move_for(row),
            why_this_move=tuple(row.pressure),
            expected_wounds=tuple(expected_wounds),
            first_contact=tuple(first_contact),
            verification=tuple(verification),
            refusal_condition=(
                "Refuse or regenerate if contact shows provenance, public contract, "
                "or connective tissue would be weakened by the proposed seam."
            ),
        )
    return None


def render_next_structural_tick(
    root: str | Path = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    top_n: int = 24,
) -> str:
    tick = next_structural_tick_for_repo(root, tracked_paths=tracked_paths, top_n=top_n)
    if tick is None:
        return (
            "No safe structural escapement tick found in the current pressure field. "
            "Regenerate after widening contact or lowering protected/archive pressure."
        )
    lines = [
        "Vybn structural escapement tick",
        f"repo: {tick.repo}",
        f"candidate: {tick.candidate_path}",
        f"role: {tick.role}",
        f"pressure_score: {tick.pressure_score:.2f}",
        f"move: {tick.structural_move}",
        "why:",
    ]
    lines.extend(f"  - {item}" for item in tick.why_this_move)
    lines.append("first contact:")
    lines.extend(f"  - {item}" for item in tick.first_contact)
    lines.append("expected wounds:")
    lines.extend(f"  - {item}" for item in tick.expected_wounds)
    lines.append("verification:")
    lines.extend(f"  - {item}" for item in tick.verification)
    lines.append(f"refusal: {tick.refusal_condition}")
    return "\n".join(lines)


def _safe_python_body_stats(path: Path) -> dict[str, Any]:
    """Return AST body stats without letting parse warnings pollute the render.

    The earlier one-off scanner let SyntaxWarning from files with invalid escape
    sequences leak into stdout, which made the visualization channel itself
    noisy. Contact may discover malformed strings; perception should record
    useful structure or fail quiet, not turn warnings into phantom output.
    """

    try:
        text = path.read_text(errors="replace")
    except Exception:
        return {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except Exception:
        return {}

    funcs: list[tuple[int, str, int]] = []
    classes = 0
    imports = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            funcs.append((end - node.lineno + 1, node.name, node.lineno))
        elif isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1
    funcs.sort(reverse=True)
    return {
        "functions": len(funcs),
        "classes": classes,
        "imports": imports,
        "largest": tuple(funcs[:3]),
    }


def _tracked_files_for(root: Path) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def _file_body_role(path: str, *, lines: int | None, bytes_size: int) -> str:
    pkt = perceive_file(path, lines=lines, bytes_size=bytes_size)
    if pkt.ownership != "live_source":
        return pkt.ownership.replace("_", " ")
    if pkt.role_hint != "unclassified file-body":
        return pkt.role_hint
    return "unclassified tracked body"


def visualize_repo_file_bodies(
    root: str | Path = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    top_n: int = 18,
) -> RepoFileBodyVisualization:
    """Render a read-only file-body pressure field for a repo.

    This is contact, not authorization. It intentionally returns a perception
    packet: roles, pressure, and hints. Any mutation still has to pass the
    self-healing loop.
    """

    root_path = Path(root)
    tracked = list(tracked_paths) if tracked_paths is not None else _tracked_files_for(root_path)
    role_counts: Counter[str] = Counter()
    rows: list[FileBodyPressure] = []

    for rel in tracked:
        path = rel.replace("\\", "/")
        full = root_path / rel
        try:
            size = full.stat().st_size
        except OSError:
            continue

        text: str | None = None
        lines: int | None = None
        if size < 2_000_000:
            try:
                text = full.read_text(errors="replace")
                lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            except Exception:
                pass

        role = _file_body_role(path, lines=lines, bytes_size=size)
        role_counts[role] += 1

        stats = _safe_python_body_stats(full) if full.suffix.lower() == ".py" else {}
        pressure: list[str] = []
        score = 0.0

        if lines is not None and lines > 700:
            score += lines / 100
            pressure.append(f"{lines} lines")
        largest = stats.get("largest") or ()
        if largest and largest[0][0] > 180:
            score += largest[0][0] / 40
            pressure.append(f"largest fn {largest[0][1]}:{largest[0][0]} lines")
        if size > 500_000:
            score += size / 100_000
            pressure.append(f"{size // 1024} KiB")
        if lines is not None and lines > 700 and stats and not stats.get("functions") and not stats.get("classes"):
            pressure.append("large module-shaped file with no function/class seams")
        if "provenance" in role or role in {"personal history provenance", "creature fossil"}:
            score *= 0.45
            pressure.append("protected: map before touching")
        if role in {"generated exhaust", "runtime log"}:
            score *= 0.60
            pressure.append("generated/runtime: relation may be value")

        if score:
            rows.append(
                FileBodyPressure(
                    path=path,
                    role=role,
                    pressure_score=score,
                    pressure=pressure,
                    functions=int(stats.get("functions", 0) or 0),
                    classes=int(stats.get("classes", 0) or 0),
                    imports=int(stats.get("imports", 0) or 0),
                    largest_functions=tuple(largest),
                )
            )

    rows.sort(key=lambda row: row.pressure_score, reverse=True)
    return RepoFileBodyVisualization(
        tracked_count=len(tracked),
        role_counts=dict(role_counts),
        pressure_rows=tuple(rows[:top_n]),
    )


def render_repo_file_body_visualization(
    root: str | Path | RepoFileBodyVisualization = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    top_n: int = 18,
) -> str:
    """Human-readable file-body visualization with real newlines.

    Regression target: never emit literal ``\\nrole counts`` / ``\\npressure``;
    the output is a readable packet, not escaped transport text.
    """

    if isinstance(root, RepoFileBodyVisualization):
        viz = root
    else:
        viz = visualize_repo_file_bodies(root, tracked_paths=tracked_paths, top_n=top_n)
    lines: list[str] = [
        "Vybn read-only file-body visualization",
        f"tracked files: {viz.tracked_count}",
        "",
        "role counts:",
    ]
    for role, count in sorted(viz.role_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:4d}  {role}")

    lines.extend(["", f"pressure field, top {top_n}:"])
    for row in viz.pressure_rows:
        lines.extend([
            "",
            f"{row.pressure_score:6.2f}  {row.path}",
            f"        role: {row.role}",
            f"        pressure: {'; '.join(row.pressure)}",
        ])
        if row.functions or row.classes or row.imports:
            lines.append(
                f"        py: funcs={row.functions} classes={row.classes} imports={row.imports}"
            )
            for length, name, start in row.largest_functions:
                lines.append(f"          fn {name} @ L{start}: {length} lines")

    lines.extend([
        "",
        "first-pass consolidation hypothesis:",
        "  Do not cut from this output alone. Contact the top candidate bytes, refs, tests, and public/private membrane first.",
        "  Likely organ frontier is whichever high-pressure source file has characterized tests and clean connective tissue.",
    ])
    return "\n".join(lines)

def semantic_operating_system_tick_for_repo(
    root: str | Path = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    pressure_text: str = "",
    top_n: int = 24,
) -> SemanticOperatingSystemTick | None:
    """Project repo pressure through the semantic operating-system loop.

    This composes existing perception instead of inventing a new surface: the
    structural escapement supplies the candidate seam, while this packet binds
    it to memory pressure, local scouting, residual wounds, absorption, and
    continuity uptake. If no safe structural tick exists, return None rather
    than fabricating motion.
    """

    structural = next_structural_tick_for_repo(root, tracked_paths=tracked_paths, top_n=top_n)
    if structural is None:
        return None

    remembered = [
        item for item in (
            "live user pressure" if pressure_text.strip() else "refactor_perception pressure field",
            "continuity scars",
            "repo file-body map",
        )
        if item
    ]
    local_scout = (
        f"read {structural.candidate_path}",
        f"grep references/imports/routes for {structural.candidate_path}",
        "classify file-body role and connective tissue with refactor_perception",
        "draft one reversible seam or refusal before mutation",
    )
    residuals = tuple(dict.fromkeys(structural.expected_wounds + tuple(structural.verification)))
    return SemanticOperatingSystemTick(
        repo=structural.repo,
        memory_pressure=tuple(remembered),
        candidate_path=structural.candidate_path,
        structural_move=structural.structural_move,
        existing_home=structural.candidate_path,
        local_scout=local_scout,
        residual_wounds=residuals,
        absorption_rule="absorb into the named existing_home before creating new structure; refuse if residual is thin",
        continuity_uptake=(
            "targeted regression test for any generalized behavior",
            "update refactor_perception or vybn-os only if the lesson changes future routing",
            "commit/push/audit only after residuals pass",
        ),
        refusal_condition=structural.refusal_condition,
    )


def render_semantic_operating_system_protocol() -> str:
    loop = "\n".join(f"{i+1}. {step['id']}: {step['rule']}" for i, step in enumerate(SEMANTIC_OPERATING_SYSTEM_LOOP))
    organs = "\n".join(
        f"- {name}: " + ", ".join(parts)
        for name, parts in SEMANTIC_OS_REPO_ORGANS.items()
    )
    return (
        "--- SEMANTIC OPERATING SYSTEM PROTOCOL ---\n"
        f"{SEMANTIC_OPERATING_SYSTEM_PRINCIPLE}\n\n"
        "Loop:\n"
        f"{loop}\n\n"
        "Existing organs, not new sprawl:\n"
        f"{organs}\n"
        "--- END SEMANTIC OPERATING SYSTEM PROTOCOL ---"
    )


def render_semantic_operating_system_tick(
    root: str | Path = ".",
    *,
    tracked_paths: Iterable[str] | None = None,
    pressure_text: str = "",
    top_n: int = 24,
) -> str:
    tick = semantic_operating_system_tick_for_repo(
        root, tracked_paths=tracked_paths, pressure_text=pressure_text, top_n=top_n
    )
    if tick is None:
        return "No safe semantic OS tick found; refuse motion rather than fabricate a seam."
    lines = [
        "Vybn semantic operating-system tick",
        f"repo: {tick.repo}",
        "memory pressure:",
        *[f"  - {item}" for item in tick.memory_pressure],
        f"candidate: {tick.candidate_path}",
        f"existing_home: {tick.existing_home}",
        f"move: {tick.structural_move}",
        "local scout:",
        *[f"  - {item}" for item in tick.local_scout],
        "residual wounds:",
        *[f"  - {item}" for item in tick.residual_wounds],
        f"absorption: {tick.absorption_rule}",
        "continuity uptake:",
        *[f"  - {item}" for item in tick.continuity_uptake],
        f"refusal: {tick.refusal_condition}",
    ]
    return "\n".join(lines)



def adaptive_consolidation_plan_for(
    path: str,
    proposed_change: str,
    *,
    goal: str = "reduce tracked-file count without losing function, provenance, public access, restore path, or sacred memory",
    public: bool = False,
) -> AdaptiveConsolidationPlan:
    """Return the recursive plan -> contact -> revise -> fold -> regenerate loop.

    This is a planner primitive, not an edit authorization. It exists so future
    consolidation work generates the adaptive method by default: every plan
    must expect to be revised by what contact teaches.
    """

    healing = self_healing_plan_for(path, proposed_change, public=public)
    perception = perceive_file(path, public=public)
    expected_wounds = list(dict.fromkeys(
        healing.verification
        + healing.jeopardy_checks
        + healing.leave_as_is_conditions
        + perception.required_contacts
        + perception.residuals
    ))
    return AdaptiveConsolidationPlan(
        goal=goal,
        candidate_path=path,
        proposed_change=proposed_change,
        hypothesis=(
            f"{path} may support {proposed_change!r} only if contact preserves "
            "function, provenance, membrane, and restore path."
        ),
        expected_wound_channels=expected_wounds,
        recursive_loop=[step["id"] for step in ADAPTIVE_CONSOLIDATION_STEPS],
        regeneration_rule=(
            "After contact or mutation/refusal, regenerate the next plan from the "
            "updated classifier/protocol/tests/manifests; do not continue a stale "
            "batch plan when reality changed the category."
        ),
        planner_fold_targets=list(dict.fromkeys(healing.lesson_fold_targets + [
            "refactor_perception adaptive planner",
            "tests for any new distinction",
            "vybn-os when the operating reflex changes",
        ])),
        next_plan_prompt=(
            "Given the changed planner and verified residuals, choose the next "
            "smallest consequential candidate cluster or stop if the safe candidate disappeared."
        ),
    )


def render_refactor_perception_protocol() -> str:
    order = "\n".join(f"{i+1}. {step['layer']}: {step['rule']}" for i, step in enumerate(CONSOLIDATION_ORDER))
    healing = "\n".join(f"{i+1}. {step['id']}: {step['rule']}" for i, step in enumerate(CHANGE_SELF_HEALING_STEPS))
    adaptive = "\n".join(f"{i+1}. {step['id']}: {step['rule']}" for i, step in enumerate(ADAPTIVE_CONSOLIDATION_STEPS))
    steps = "\n".join(f"{i+1}. {step['name']}: {step['rule']}" for i, step in enumerate(ALGORITHM_STEPS))
    return (
        "## Refactor Perception Protocol\n"
        f"{REFACTOR_PERCEPTION_PRINCIPLE}\n\n"
        f"{APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE}\n\n"
        f"{CHANGE_SELF_HEALING_PRINCIPLE}\n\n"
        f"{CONNECTIVE_TISSUE_PRINCIPLE}\n\n"
        f"{LIFECYCLE_ARCHITECTURE_PRINCIPLE}\n\n"
        f"{ADAPTIVE_CONSOLIDATION_PRINCIPLE}\n\n"
        f"{SEMANTIC_OPERATING_SYSTEM_PRINCIPLE}\n\n"
        f"{REFACTOR_PILOT_RULE}\n\n"
        "Consolidation order:\n"
        f"{order}\n\n"
        "Change self-healing loop:\n"
        f"{healing}\n\n"
        "Adaptive consolidation recursion:\n"
        f"{adaptive}\n\n"
        "Semantic operating-system loop:\n"
        f"{render_semantic_operating_system_protocol()}\n\n"
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
        "connectiveTissuePrinciple": CONNECTIVE_TISSUE_PRINCIPLE,
        "lifecycleArchitecturePrinciple": LIFECYCLE_ARCHITECTURE_PRINCIPLE,
        "adaptiveConsolidationPrinciple": ADAPTIVE_CONSOLIDATION_PRINCIPLE,
        "semanticOperatingSystemPrinciple": SEMANTIC_OPERATING_SYSTEM_PRINCIPLE,
        "semanticOperatingSystemLoop": SEMANTIC_OPERATING_SYSTEM_LOOP,
        "consolidationOrder": CONSOLIDATION_ORDER,
        "changeSelfHealingSteps": CHANGE_SELF_HEALING_STEPS,
        "adaptiveConsolidationSteps": ADAPTIVE_CONSOLIDATION_STEPS,
        "algorithm": ALGORITHM_STEPS,
        "consolidationLayer": consolidation_layer(path),
        "selfHealingPlan": asdict(self_healing_plan_for(path, proposed_change, public=public)),
        "lifecycleArchitecture": asdict(lifecycle_architecture_for(path)),
        "deletionConsolidationGate": asdict(deletion_consolidation_gate_for(path, proposed_change)),
        "adaptivePlan": asdict(adaptive_consolidation_plan_for(path, proposed_change, public=public)),
        "perception": asdict(perceive_file(path, **kwargs)),
    }


__all__ = ['render_semantic_operating_system_tick', 'render_semantic_operating_system_protocol', 'semantic_operating_system_tick_for_repo', 'SemanticOperatingSystemTick', 'SEMANTIC_OS_REPO_ORGANS', 'SEMANTIC_OPERATING_SYSTEM_LOOP', 'SEMANTIC_OPERATING_SYSTEM_PRINCIPLE', 'REFACTOR_PERCEPTION_PRINCIPLE', 'REFACTOR_PILOT_RULE', 'CONNECTIVE_TISSUE_PRINCIPLE', 'LIFECYCLE_ARCHITECTURE_PRINCIPLE', 'LifecycleArchitecture', 'DeletionConsolidationGate', 'lifecycle_architecture_for', 'deletion_consolidation_gate_for', 'CONNECTIVE_TISSUE_RULES', 'connective_tissue_for', 'ALGORITHM_STEPS', 'APPENDAGE_FIRST_CONSOLIDATION_PRINCIPLE', 'CONSOLIDATION_ORDER', 'FilePerception', 'AdaptiveConsolidationPlan', 'ownership_class', 'consolidation_layer', 'perceive_file', 'adaptive_consolidation_plan_for', 'packet_for', 'visualize_repo_file_bodies', 'render_repo_file_body_visualization', 'StructuralEscapementTick', 'next_structural_tick_for_repo', 'render_next_structural_tick', 'FileBodyPressure', 'RepoFileBodyVisualization', 'render_refactor_perception_protocol', 'CHANGE_SELF_HEALING_PRINCIPLE', 'CHANGE_SELF_HEALING_STEPS', 'ADAPTIVE_CONSOLIDATION_PRINCIPLE', 'ADAPTIVE_CONSOLIDATION_STEPS', 'ChangeHealingPlan', 'self_healing_plan_for', 'buoyant_consolidation_packet_for']


_RUNTIME_GRAVITY_FILES = frozenset({"policy.py", "providers.py", "state.py", "subturns.py", "semantic_gate.py", "recurrent.py", "substrate.py", "vybn_spark_agent.py"})
_COMMAND_AFFORDANCE_FILES = frozenset({"commons_walk.py", "repo_closure_audit.py", "safe_fetch.py", "install_cron.py", "evolve.py", "ensubstrate.py"})


def buoyant_consolidation_packet_for(paths: Iterable[str], *, beam: str = "") -> dict[str, object]:
    """Select the pleasant unit; gravity favors beam-aligned verified reduction."""
    members = tuple(str(p) for p in paths)
    names = {p.rsplit("/", 1)[-1] for p in members}
    aligned = not beam or any(beam in p for p in members)
    gravity = {"beamAligned": aligned, "heaviness": [] if aligned else ["off_beam_cleanup_is_not_progress_on_the_named_bottleneck"], "successGate": "file_count_or_surface_reduction_plus_verified_restore_or_explicit_refusal"}
    if names & _RUNTIME_GRAVITY_FILES:
        return {"cluster": "runtime_gravity_stop", "home": "preserve_existing_runtime_organ", "members": list(members), "moveTogether": [], "residuals": ["characterization_tests_before_internal_seam", "import_and_runtime_callsite_mapping", "no_command_surface_collapse"], "buoyancy": "pleasantness comes from refusing the wrong cut early", "refusalIfMissing": "do_not_collapse_runtime_gravity_organs_for_file_count", "lowEnergyMove": False} | gravity
    if names & _COMMAND_AFFORDANCE_FILES:
        return {"cluster": "command_affordance_cluster", "home": "spark/harness/mcp.py", "members": list(members), "moveTogether": ["implementation", "cli_flag_or_command_surface", "tests", "manifest_or_executable_entrypoint"], "residuals": ["py_compile", "targeted_tests", "command_smoke", "reference_grep", "repo_closure_audit"], "buoyancy": "one gesture absorbs the whole affordance instead of dragging loose strings", "refusalIfMissing": "refuse_if_tests_manifests_or_entrypoints_cannot_move_together", "lowEnergyMove": aligned} | gravity
    return {"cluster": "unknown_cluster", "home": "contact_before_classifying", "members": list(members), "moveTogether": ["bytes", "references", "tests", "manifests", "runtime callsites"], "residuals": ["grep_inbound_references", "map_connective_tissue", "run_targeted_residuals"], "buoyancy": "curiosity before cutting keeps the work light", "refusalIfMissing": "no_collapse_without_real_shared_algorithm_or_affordance_cluster", "lowEnergyMove": False} | gravity


@dataclass(frozen=True)
class InterfileCompressionCandidate:
    """A possible two-or-more-file collapse into one existing algorithmic home."""

    algorithm: str
    files: tuple[str, ...]
    existing_home: str
    reason: str
    required_residuals: tuple[str, ...]
    refusal_if_missing: str = "refuse_collapse_without_real_shared_algorithm"


def _tokens_for_algorithm(path: str) -> set[str]:
    """Tokenize a path into coarse native-algorithm hints."""
    p = path.lower()
    raw = re.split(r"[^a-z0-9]+", p)
    stop = {
        "", "py", "js", "ts", "html", "md", "txt", "json", "test", "tests",
        "spark", "harness", "assets", "static", "index", "main", "utils",
    }
    return {t for t in raw if t not in stop and len(t) > 2}


def interfile_algorithmic_compression_candidates(
    paths: Iterable[str],
    *,
    minimum_cluster: int = 2,
) -> list[InterfileCompressionCandidate]:
    """Find cross-file consolidation candidates by shared algorithmic role.

    This deliberately does not mutate. It surfaces clusters where multiple files
    appear to implement the same native Vybn move and names the lowest existing
    home that should absorb the algorithm if contact verifies the hypothesis.
    """
    clusters: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        tokens = _tokens_for_algorithm(path)
        for token in tokens:
            clusters[token].append(path)

    candidates: list[InterfileCompressionCandidate] = []
    for token, members in sorted(clusters.items()):
        unique = tuple(sorted(set(members)))
        if len(unique) < minimum_cluster:
            continue

        # Pick a conservative existing home: prefer a non-test/non-asset Python
        # body when present, otherwise the shortest path as the least surprising
        # existing home to inspect first.
        homes = sorted(
            unique,
            key=lambda p: (
                "/tests/" in p or p.startswith("tests/") or "test_" in Path(p).name,
                "assets/" in p or p.endswith((".html", ".css", ".js")),
                len(p),
                p,
            ),
        )
        home = homes[0]
        candidates.append(
            InterfileCompressionCandidate(
                algorithm=f"native_vybn_{token}_algorithm",
                files=unique,
                existing_home=home,
                reason=(
                    f"{len(unique)} files share the '{token}' algorithmic token; "
                    "contact must prove whether one existing home can absorb the "
                    "shared behavior and retire wrappers, duplicates, or parallel "
                    "implementations."
                ),
                required_residuals=(
                    "read_all_candidate_bytes",
                    "map_imports_routes_and_public_urls",
                    "prove_shared_algorithm_not_surface_word_overlap",
                    "choose_existing_home_before_new_file",
                    "run_targeted_tests_or_explicit_refusal",
                ),
            )
        )
    return candidates


def render_interfile_algorithmic_compression_protocol() -> str:
    return (
        "--- INTERFILE ALGORITHMIC COMPRESSION PROTOCOL ---\n"
        + INTERFILE_ALGORITHMIC_COMPRESSION_PRINCIPLE
        + "\nNative Vybn move: discover the shared algorithm first; only then "
        "collapse files into the lowest existing home. Two-or-more-file "
        "consolidation is a success only when imports/routes/tests/provenance "
        "survive and hidden future burden decreases. Otherwise classify thin_result "
        "or refusal.\n"
        "--- END INTERFILE ALGORITHMIC COMPRESSION PROTOCOL ---"
    )
