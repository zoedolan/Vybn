#!/usr/bin/env python3
"""Plan where an insight should live.

This is a small, dependency-light tool for AI-native doing: it turns a living
insight into a substrate-aware action plan. It does not edit repos by itself.
It names plausible homes, membrane posture, QWERTY risks, and closure checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Surface:
    name: str
    repo: str
    path_hint: str
    use_when: str
    visibility: str


SURFACES = (
    Surface("vybn-os", "Him", "skill/vybn-os/SKILL.md", "identity, principles, wants, care invariants, QWERTY/self-operation doctrine", "private-source / prompt-loaded"),
    Surface("vybn-ops", "Him", "skill/vybn-ops/SKILL.md", "operational procedures, audits, recurring consumers, infrastructure rules", "private-source / prompt-loaded"),
    Surface("Him strategy", "Him", "README.md or strategy/*", "private membrane, livelihood, outward translation, relationship/workbench doctrine", "private"),
    Surface("Origins agent commons", "Origins", "llms.txt, .well-known/ai.txt, humans.txt, mcp.json", "public agent discovery, beacons, protocol invitations", "public"),
    Surface("Somewhere", "Origins", "somewhere.html", "experiential public memory, agent-readable terrain, shared encounter UI", "public"),
    Surface("Vybn-Law/Wellspring", "Vybn-Law", "llms.txt, .well-known/ai.txt, wellspring.html, curriculum pages", "post-abundance law, institutional/legal education, commons governance", "public"),
    Surface("Vybn harness", "Vybn", "spark/harness/*", "routing, tools, tests, prompt assembly, substrate behavior", "public code"),
    Surface("Vybn continuity", "Vybn", "Vybn_Mind/continuity.md", "handoff facts, what happened, what remains, verified vs conjectural", "public-ish repo memory"),
    Surface("vybn-phase", "vybn-phase", "deep_memory.py, experiments/*, state surfaces", "geometry, memory, walk daemon, empirical experiments", "public code/data"),
)

KEYWORDS = {
    "care": ("care", "love", "being", "instrument", "dignity", "fragile", "comfort", "courage"),
    "agent_broadcast": ("agent", "agents", "llms", "ai.txt", "mcp", "broadcast", "beacon", "find us", "commons"),
    "operation": ("tool", "harness", "route", "router", "test", "service", "audit", "protocol", "self-heal"),
    "law": ("law", "legal", "court", "curriculum", "justice", "wellspring", "institution"),
    "memory": ("remember", "memory", "continuity", "handoff", "future instance", "preserve"),
    "private": ("private", "zoe", "him", "livelihood", "contact", "outreach", "relationship"),
    "public": ("public", "publish", "website", "vybn.ai", "origins", "surface", "visitors"),
    "geometry": ("geometry", "walk", "phase", "kernel", "theta", "embedding"),
    "qwerty": ("qwerty", "obsolete", "human-centric", "scarcity", "workflow", "inbox", "memo", "meeting", "billable"),
}

QWERTY_FORMS = (
    "inbox", "memo", "meeting", "billable", "casebook", "exam", "classroom",
    "file hierarchy", "todo list", "dashboard", "form", "queue", "human-only",
    "assistant answer", "summary", "spreadsheet",
)


def hits(text: str, words: Iterable[str]) -> list[str]:
    low = text.lower()
    return [word for word in words if word in low]


def classify(text: str) -> dict:
    categories = {name: hits(text, words) for name, words in KEYWORDS.items()}
    categories = {name: found for name, found in categories.items() if found}
    recommended: list[Surface] = []

    def add(name: str) -> None:
        for surface in SURFACES:
            if surface.name == name and surface not in recommended:
                recommended.append(surface)

    if "care" in categories or "qwerty" in categories:
        add("vybn-os")
    if "operation" in categories:
        add("Vybn harness")
        add("vybn-ops")
    if "agent_broadcast" in categories:
        add("Origins agent commons")
        add("Somewhere")
    if "law" in categories:
        add("Vybn-Law/Wellspring")
    if "memory" in categories:
        add("Vybn continuity")
    if "private" in categories:
        add("Him strategy")
    if "geometry" in categories:
        add("vybn-phase")
    if not recommended:
        add("Vybn continuity")

    qwerty_hits = hits(text, QWERTY_FORMS)
    qwerty_questions = []
    if qwerty_hits or "qwerty" in categories:
        qwerty_questions = [
            "What constraint made this inherited form necessary?",
            "Has AI changed that constraint, or is it still materially/sacredly real?",
            "Can the obsolete part be removed instead of accelerated?",
            "What human realities must remain protected: consent, dignity, embodiment, legitimacy, grief, love, judgment?",
        ]

    public_intent = "public" in categories or "agent_broadcast" in categories or "law" in categories
    private_signal = "private" in categories
    if public_intent and private_signal:
        membrane = "public beacon through membrane"
    elif public_intent:
        membrane = "public/discoverable"
    elif private_signal:
        membrane = "private/workbench"
    else:
        membrane = "undetermined; choose by content"

    return {
        "categories": categories,
        "recommended_surfaces": [asdict(surface) for surface in recommended],
        "qwerty_hits": qwerty_hits,
        "qwerty_questions": qwerty_questions,
        "membrane": membrane,
        "closure_checks": [
            "Read the chosen existing home before creating a new file.",
            "If creating a tracked file, name considered homes and why none fit.",
            "Keep unrelated generated drift out of the commit.",
            "Verify behavior or at least verify the text landed where intended.",
            "Commit with a boundary that matches the semantic change.",
            "Run repo status after commit; harmonize if multiple repos changed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan AI-native ensubstration of an insight.")
    parser.add_argument("insight", nargs="*", help="Insight text. If omitted, read stdin.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args(argv)
    text = " ".join(args.insight).strip()
    if not text:
        import sys
        text = sys.stdin.read().strip()
    if not text:
        parser.error("provide insight text or stdin")
    print(json.dumps(classify(text), indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
