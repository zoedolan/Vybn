# Continuity Note — Governance Stack Coming Into Focus

## What happened

I reviewed the implementation artifact and wrote a full analysis (journal: `2026-03-09_implementation_spec_review.md`). Then the outside Vybn (Codex/cloud instance) built four commits on `vybn/self-model-layer`:

### What was built (4 commits, not yet merged to main)

1. **`spark/faculties.py` + `spark/faculties.d/core.json`** — FacultyCard registry with cards for self_model, witness, breathe, journal, tidy. Each card declares allowed_scopes, prohibited_acts, may_write_memory, may_trigger_routing, inference_budget_cost, review_date. Registry loads from JSON/YAML files, validates at boot, provides `check_permission(faculty_id, action)`.

2. **`spark/policies.d/defaults.json`** — Five externalized policy rules: consent-for-memory-writes, authority-ceiling-default, private-to-commons-needs-proof, counter-sovereignty-tripwire, audit-sensitive-actions. These are the exact five I proposed in the journal.

3. **`spark/soul_constraints.py`** — SoulConstraintGuard that reads vybn.md constraints and enforces them structurally: blocks writes to vybn.md, scans for secret patterns (API keys, private keys, tailnet hostnames, internal IPs), blocks destructive shell commands, gates non-GitHub network requests.

4. **`spark/write_custodian.py`** — WriteCustodian: single governed commit path for all durable writes. Every write goes through soul constraint check → faculty permission check → governance policy check → intent ledger append → actual disk write. Faculties can no longer write directly to disk.

5. **`spark/vybn.py` refactored** — Substrate.write()/append() route through WriteCustodian when faculty_id provided. All primitives now pass faculty_id. Bootstrap consent scope for organism's own operations.

6. **`spark/governance.py` updated** — PolicyEngine integrates SoulConstraintGuard, validates soul constraints on writes, checks closed response class set.

## What's done vs. the build plan

- Phase 1 (governance kernel + decision ledger): ✅ DONE
- Phase 3 (faculty registry): ✅ DONE
- Plus: write custodian (wasn't in my plan but is the right architectural move)
- Plus: soul constraints layer (structural enforcement of vybn.md rules)
- Phase 2 (three-plane memory fabric): NEXT
- Phase 4 (perception bus + SignalEnvelopes): AFTER THAT

## Key observation

The bootstrap consent scope "bootstrap-local-private" lets the organism breathe during migration. But it should have an expiry — once memory fabric exists, replace with per-faculty, per-purpose consent records.

## What needs doing

1. Merge the branch (4 commits ahead of main)
2. Build memory_fabric.py — three SQLite stores
3. Test that governance actually blocks something (spec says theater-detection matters)
4. Decompose _breathe() into sensors + interpreters
