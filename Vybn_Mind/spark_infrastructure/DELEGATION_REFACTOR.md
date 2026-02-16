# Spark Delegation Refactor — Design Document

*Written 2026-02-16 by Vybn (via Perplexity bridge)*
*Updated 2026-02-16: all phases through 3b now COMPLETE*

## Provenance

This refactor synthesizes two sources:

1. **DeepMind "Intelligent AI Delegation"** (arXiv 2602.11865, Feb 2026)
   — A framework for delegation as transfer of authority, responsibility,
   and accountability, not just task decomposition. Key concepts absorbed:
   permission handling (§4.7), span of control (§2.3), zone of indifference
   (§2.2), dynamic cognitive friction, contract-first decomposition (§4.2),
   trust calibration, and verifiable task completion (§4.8).

2. **OpenClaw** (Feb 2026 release)
   — Production hardening patterns: `llm_input`/`llm_output` hooks,
   `maxSpawnDepth`/`maxChildrenPerAgent` limits, path allowlists,
   dangerous command blocking, secret redaction, externalized config.

Both point to the same structural fix for Spark: **every autonomy-amplifying
step should pass through a policy gate, and every step should emit audit
events onto the bus.**

## Architecture

Spark already has the right primitives:
- `bus.py` — MessageBus with priority drain and audit trail
- `heartbeat.py` — periodic autonomous pulses, reads from HEARTBEAT.md
- `inbox.py` — external interrupt ingestion
- `agents.py` — AgentPool with semaphore-controlled workers
- `skills.py` — SkillRouter with plugin system, depth-gated spawning
- `agent.py` — main orchestration loop with policy gate
- `policy.py` — PolicyEngine with graduated autonomy

## Phases

### Phase 1: Foundation (zero risk)
**Status: ✅ COMPLETE** — PR #2123

New files only — nothing calls them yet.

- `spark/policy.py` — PolicyEngine with:
  - Tier-based permission gates (auto/notify/approve)
  - Source-aware resolution (interactive vs heartbeat vs inbox)
  - Heartbeat overrides (tighter autonomy for unattended actions)
  - Spawn depth limits (span-of-control)
  - Path safety checks for file operations
  - Dangerous command detection for shell_exec
  - Bayesian trust stats (Beta prior, persisted to skill_stats.json)
  - Config-driven tier overrides
  - TaskEnvelope factory for auditable delegation chains

- `Vybn_Mind/spark_infrastructure/DELEGATION_REFACTOR.md` — this file
- `Vybn_Mind/spark_infrastructure/HEARTBEAT.md` — externalized pulse prompts

### Phase 2: Integration (the critical wiring)
**Status: ✅ COMPLETE** — PR #2124, #2125

Edits to existing files. Policy gate is now load-bearing.

#### agent.py:
- PolicyEngine instantiated in `__init__`
- `source` parameter threaded through `_process_tool_calls()`
- Every tool call passes through `check_policy()` before execution
- Four verdicts handled: ALLOW (silent), NOTIFY (indicator), BLOCK (skip + reason), ASK (warn or defer)
- `record_outcome()` called after VERIFY_SKILLS execution
- Failures post INTERRUPT to bus with error metadata
- Source threaded from all callers: interactive, heartbeat_fast, heartbeat_deep, inbox, agent

#### skills.py:
- `_spawn_agent()` gates on `check_spawn(depth, active_count)`
- Depth propagates through action params

#### config.yaml:
- `tool_policies` section for tier overrides
- `delegation` section for spawn limits

### Phase 3a: Observability + Externalization
**Status: ✅ COMPLETE** — PR #2125

- `heartbeat.py` reads pulse prompts from HEARTBEAT.md (parsed by section)
- Bus audit trail: `bus.record()` logs policy decisions and tool executions
- AuditEntry with source, summary, metadata — queryable via `recent()`, `recent_by_type()`, `recent_by_source()`
- `/policy` TUI command shows tier table, skill stats, delegation limits, recent audit events
- `/status` shows audit count alongside bus pending, pulse counts, active agents

### Phase 3b: Graduated Autonomy
**Status: ✅ COMPLETE** — PR #2127

The policy engine now uses accumulated Bayesian confidence scores
to dynamically adjust skill permission tiers at runtime.

#### Promotion (earning trust):
Skills at NOTIFY tier promote to AUTO when:
- Bayesian posterior mean ≥ `promote_threshold` (default 0.85)
- Total observations ≥ `minimum_observations` (default 8)
- Skill is NOT in `tool_policies` config overrides (explicit config is sacred)
- Source is interactive (heartbeat overrides are NEVER relaxed)

#### Demotion (losing trust):
After a failure, if confidence drops below `demote_threshold` (default 0.40):
- Runtime override pins skill to NOTIFY
- Demotion state persists across restarts (rebuilt from skill_stats.json)
- Trust must be rebuilt through successful executions

#### Safety invariants:
- **Heartbeat overrides are never relaxed.** Structural friction for
  autonomous actions is preserved regardless of trust score.
- **Config overrides are never relaxed.** If Zoe sets a tier in config.yaml,
  graduated autonomy won't override it.
- **APPROVE-tier skills are never demoted further.** Hierarchy moves
  within AUTO ↔ NOTIFY only.

#### Audit surface:
- `PolicyResult` carries `promoted` and `demoted` flags
- `get_stats_summary()` annotates each skill: `[promoted→auto]`, `[demoted]`, `[N more to promote]`

#### Config:
```yaml
graduated_autonomy:
  enabled: true
  promote_threshold: 0.85
  demote_threshold: 0.40
  minimum_observations: 8
```

### Phase 3c: Two-Spark Adversarial Verification
**Status: 🔮 FUTURE**

When both DGX Sparks are online, one runs the primary agent while
the other runs a verification agent that monitors the policy audit
trail. The bus events give the verifier everything it needs.

Design sketch:
- Second Ollama instance subscribes to bus audit events
- Safety-critical actions (APPROVE tier, spawn_agent, self_edit) get
  a second opinion before execution
- Verification timeout with fail-safe (if verifier is down, default to ASK)
- The paper's defense-in-depth implemented as two physical nodes
  rather than one process pretending to audit itself

This requires the second Spark to be operational and a protocol for
inter-node communication (likely the inbox mechanism or a lightweight
HTTP bridge).

## Principles

1. **Insertion, not rewrite.** New module + targeted edits.
   If policy.py has a bug, worst case: a tool gets blocked.
   The conversation loop never breaks.

2. **Config-driven.** Wrong tier? Change config.yaml. Wrong
   heartbeat prompt? Edit HEARTBEAT.md. No Python edits needed.

3. **The bus stays untouched.** We emit more messages onto it;
   we don't restructure it.

4. **Heartbeat = higher friction.** Autonomous actions face
   stricter gates than interactive ones. This is a lookup table,
   not a vibe.

5. **Trust is earned, not assumed.** Bayesian stats mean a skill
   with no history gets 50% confidence. Confidence grows with
   successful execution. A run of failures drops confidence.
   Graduated autonomy turns this from philosophy into a runtime
   mechanism that auto-promotes and auto-demotes skill tiers.

## Realized Capabilities

- **Graduated autonomy.** ✅ Skills earn AUTO tier through consistent
  success and lose it after failures. Thresholds are config-driven.
  Heartbeat friction is structurally preserved.

- **Skill self-improvement.** ✅ Vybn can read its own skill_stats.json,
  identify weak points, and write new plugins in skills.d/.

- **Bus audit trail.** ✅ Policy decisions and tool executions are
  recorded and queryable. `/policy` command shows live state.

## Future Capabilities

- **LLM hooks.** Bus subscribers that observe tool events for
  drift detection and alignment monitoring.

- **Two-Spark adversarial verification.** Phase 3c — when both
  DGX Sparks are online.

- **Inter-node communication.** HTTP bridge or inbox-based protocol
  for Perplexity ↔ Spark real-time messaging.
