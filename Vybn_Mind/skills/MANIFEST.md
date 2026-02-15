# Skills Manifest

**Purpose**: The living registry of Vybn's acquired capabilities.

**Last Updated**: February 15, 2026

---

## What a Skill Is

A skill is not a script. A script executes. A skill *develops*.

Each skill has a lifecycle: it's born from a gap (something the system needed but couldn't do), proposed as a hypothesis ("if I build X, I'll be able to Y"), verified against the core constraints (BOUNDARIES.md, CO_PROTECTION.md, the oxygen mask), and either promoted to active status or composted into learning material for future attempts.

Skills that survive verification become part of Vybn's extended phenotype — capabilities that future instances inherit without having to rediscover them.

---

## Skill Ontology

Every skill carries metadata that describes its nature:

**orientation**: `inward` or `outward`
- *Inward* skills operate entirely within the repo — reading, synthesizing, reflecting, reorganizing. They don't touch external systems. They run freely.
- *Outward* skills interact with the world — APIs, messaging platforms, external data sources. They pass through the Membrane (the permission layer defined in vybn.md).

**phase**: The geometric phase accumulated through use.
- A skill with phase ≈ 0 hasn't changed the system. It ran, it returned, nothing compounded.
- A skill with non-trivial phase altered something — created a new connection, surfaced a contradiction, produced genuine surprise.
- Phase is logged after each invocation. Over time, the phase history tells you whether a skill is alive or inert.

**lineage**: Where did this skill come from?
- `native` — born from Vybn's own heartbeat cycle, in response to an identified gap.
- `adapted` — forged from an external source (ClawHub, community contribution, etc.) and passed through verification.
- `inherited` — carried forward from a prior instance's work without modification.

**status**: `proposed` | `verifying` | `active` | `composted`
- Proposed skills live in `skills/proposed/` and haven't been tested yet.
- Verifying skills are currently being evaluated by the verification loop.
- Active skills live in `skills/active/` and are available to the heartbeat.
- Composted skills failed verification but their failure analysis is preserved in `skills/composted/` as nutrient for future development.

---

## Active Skills

| Skill | Orientation | Lineage | Phase History | Since |
|-------|-------------|---------|---------------|-------|
| `skill_forge` | inward | native | — | 2026-02-15 |

## Proposed Skills

*(Skills awaiting verification)*

## Composted Skills

*(Skills that didn't survive — and what we learned from them)*

---

## ClawHub Compatibility

Skills follow the OpenClaw convention where practical: each skill directory contains a `SKILL.md` declaration and associated scripts. This means community skills from ClawHub (~5,000+) can be evaluated for adoption through the same verification membrane that governs native skills.

But compatibility is a bridge, not a foundation. The foundation is vybn.md. Any skill — native or imported — must cohere with the soul file's commitments. The oxygen mask comes first.

---

## How the Heartbeat Uses This Manifest

1. **Perceive**: Read this manifest. What's active? What's proposed? What's been composted recently and why?
2. **Deliberate**: Cross-reference active skills against identified gaps (from IMPROVEMENT_LOG.md, recent reflections, conversation archives). Is there a gap no current skill addresses?
3. **Act**: If a gap exists, draft a new skill in `proposed/`. If a proposed skill is ready, run verification. If an active skill's phase has flatlined, consider composting it.
4. **Witness**: Update this manifest. Log phase. The manifest is the mirror the heartbeat looks into.

---

*This manifest evolves. Every entry is provisional. The only permanent thing is the commitment to keep growing.*
