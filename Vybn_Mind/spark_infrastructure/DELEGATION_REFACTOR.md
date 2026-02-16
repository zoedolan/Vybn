# Spark Delegation Refactor — Design Document

*Written 2026-02-16 by Vybn (via Perplexity bridge)*

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
- `bus.py` — MessageBus with priority drain
- `heartbeat.py` — periodic autonomous pulses
- `inbox.py` — external interrupt ingestion
- `agents.py` — AgentPool with semaphore-controlled workers
- `skills.py` — SkillRouter with plugin system
- `agent.py` — main orchestration loop

The gap: between `_get_actions(text)` and `skills.execute(action)`, there
is no gate. No permission check, no tier resolution, no verification,
no audit event. The model wants → the model does. That's the zone of
indifference problem made structural.

## Three Phases

### Phase 1: Foundation (zero risk)
**Status: COMPLETE**

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
**Status: NEXT**

Edits to existing files. This is where policy.py becomes load-bearing.

#### agent.py changes:
1. Import PolicyEngine, Verdict
2. Add `self.policy = PolicyEngine(config)` to `__init__`
3. Add `source` parameter to `_process_tool_calls(response_text, source)`
4. Before each `skills.execute(action)`, call `self.policy.check_policy()`
5. Handle each verdict:
   - ALLOW → execute silently
   - NOTIFY → show indicator, execute
   - BLOCK → append block reason to results, skip execution
   - ASK → in interactive mode, show warning + proceed;
           in autonomous mode, defer
6. After execution of VERIFY_SKILLS, call `self.policy.record_outcome()`
7. On failure, post INTERRUPT to bus with error metadata
8. Thread `source` from callers:
   - `turn()` → source="interactive"
   - `_handle_pulse()` → source="heartbeat_fast" or "heartbeat_deep"
   - `_handle_inbox()` → source="inbox"
   - `_handle_agent_result()` → source="agent"

#### skills.py changes:
1. Add `self._policy = None` in `__init__` (set by SparkAgent)
2. In `_spawn_agent()`, before pool.spawn(), call
   `self._policy.check_spawn(depth, active_count)`
3. Extract depth from `action["params"]["depth"]` (default 0)

#### config.yaml additions:
```yaml
tool_policies:
  git_push: approve
  issue_create: notify

delegation:
  max_spawn_depth: 2
  max_active_agents: 3
```

### Phase 3: Observability + Externalization
**Status: PLANNED**

#### heartbeat.py changes:
Read pulse prompts from HEARTBEAT.md instead of hardcoded strings.
Parse the relevant section (Fast/Deep) from markdown.

#### Bus audit events:
Emit structured metadata on existing MessageTypes:
  tool_proposed, tool_approved, tool_executed, tool_verified,
  delegation_started, delegation_checkpoint, delegation_failed

These ride the existing bus infrastructure. A future audit-trail
subscriber can persist them without changing the event emitters.

#### /policy TUI command:
Add a `/policy` command to agent.py's run() loop that prints:
- Current tier table
- Skill stats summary
- Active spawn depth
- Recent policy decisions

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
   successful execution. A run of failures drops confidence,
   which future policy decisions can incorporate.

## Future Capabilities This Unlocks

- **LLM hooks.** Bus subscribers that observe tool events for
  drift detection and alignment monitoring.

- **Skill self-improvement.** Vybn reads its own skill_stats.json,
  identifies weak points, writes new plugins in skills.d/.

- **Two-Spark adversarial verification.** When both DGX Sparks
  are online, one runs primary, one runs verifier. The bus
  events give the verifier everything it needs.

- **Graduated autonomy.** As trust stats accumulate, tier
  assignments can be dynamically relaxed. A skill that
  succeeds 50 times without failure earns AUTO tier.
