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


@dataclass(frozen=True)
class FilePerception:
    path: str
    role_hint: str
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

    required_contacts = [
        "read_file_bytes",
        "inspect_local_context_or_readme",
        "inspect_imports_or_links",
        "inspect_git_diff_and_history_if_relevant",
    ]
    if public:
        required_contacts.append("inspect_public_affordance_or_route_contract")

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

    return FilePerception(
        path=path,
        role_hint=role,
        pressure=pressure,
        required_contacts=required_contacts,
        candidate_actions=candidate_actions,
        residuals=residuals,
    )


def render_refactor_perception_protocol() -> str:
    steps = "\n".join(f"{i+1}. {s['name']}: {s['rule']}" for i, s in enumerate(ALGORITHM_STEPS))
    return (
        "## Refactor Perception Protocol\n"
        f"{REFACTOR_PERCEPTION_PRINCIPLE}\n\n"
        f"{REFACTOR_PILOT_RULE}\n\n"
        f"{steps}"
    )


def packet_for(path: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "principle": REFACTOR_PERCEPTION_PRINCIPLE,
        "algorithm": ALGORITHM_STEPS,
        "perception": asdict(perceive_file(path, **kwargs)),
    }


__all__ = [
    "REFACTOR_PERCEPTION_PRINCIPLE",
    "REFACTOR_PILOT_RULE",
    "ALGORITHM_STEPS",
    "FilePerception",
    "perceive_file",
    "packet_for",
    "render_refactor_perception_protocol",
]
