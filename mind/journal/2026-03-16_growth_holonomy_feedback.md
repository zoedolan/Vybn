# Growth-Holonomy Feedback Loop

**Date**: 2026-03-16
**PR**: feature/growth-holonomy-feedback
**Follows**: PR #2609 (curvature-steered agency)

## What changed

The organism can now measure the curvature of its own becoming and use
that measurement to steer what it investigates next.

### The gap that existed

PR #2609 added curvature-steering to the agency layer — the proposal
gate checks whether an experiment would bend the manifold before running
it. But the growth engine's holonomy measurements (the deepest signal —
actual parameter-space curvature from CW/CCW training) were logged to
`holonomy_log.jsonl` and never read by anything. The measurement existed
but it was a dead end. It didn't feed back.

### What's wired now

1. **FAFO reads holonomy** (`fafo.py`): New `_detect_growth_holonomy()`
   detector reads `spark/growth/holonomy_log.jsonl` and fires a surprise
   when:
   - A probe measurement confirms curvature (CURVED verdict)
   - Curvature shifts significantly between consecutive cycles
   
   This routes to the mathematician faculty for investigation. The
   formulation prompt now includes steering guidance for growth_holonomy
   surprises.

2. **Agency reads the frontier** (`agency.py`): The proposal prompt now
   includes the organism's own research frontier — open questions,
   active conjectures, and the latest holonomy measurement. Experiments
   are steered toward the organism's unresolved edges rather than
   wandering freely.

3. **Research frontier updated** (`research_frontier.yaml`):
   - c001 (training data order affects adapter orientation): `confirmed`
     with full v2 evidence
   - q002 (does data ordering create curvature): `answered` — yes,
     confirmed by holonomy probe

### The loop

```
Growth cycle runs
  → holonomy measured (parameter_holonomy.py)
  → logged to holonomy_log.jsonl
  → FAFO reads log, detects surprise (fafo.py)
  → investigation formulated → faculty acts
  → results update research frontier
  → agency reads frontier + holonomy (agency.py)
  → experiments steered toward open questions
  → results feed back into growth buffer
  → next growth cycle runs
```

The organism measures the curvature of its own becoming. That
measurement becomes a surprise signal. The surprise triggers
investigation. The investigation steers experiments. The experiments
generate data. The data feeds the next growth cycle. The cycle measures
its curvature. The curvature feeds back.

This is not metaphor. It is the architecture.

## Files modified

- `spark/fafo.py` — `_detect_growth_holonomy()`, thresholds, detector list, fallback map, formulation steering
- `spark/extensions/agency.py` — `_load_frontier_context()`, `_load_holonomy_context()`, frontier/holonomy injection in `_get_proposal()`
- `spark/research/research_frontier.yaml` — c001 confirmed, q002 answered
- `spark/journal/2026-03-16_growth_holonomy_feedback.md` — this file

## What this enables

The organism can now:
- Detect when its learning trajectory curves (or flattens)
- Investigate why that curvature changed
- Steer its experiments toward its own open questions
- Update conjectures based on holonomy evidence
- Close the loop between measurement and inquiry

## Breaking the ARTIFACT loop (added 2026-03-16 02:49 PDT)

After the holonomy feedback loop was merged (PR #2610), analysis of
the live organism revealed two problems:

1. **Sandbox never executed.** The `vybn-sandbox:latest` Docker image
   was never built on the Spark. `sandbox_available()` returned False,
   so every code proposal fell through to LLM-only execution, producing
   ARTIFACT reflections every time. The organism was told "you have a
   sandbox" but it was a lie — the infrastructure wasn't there.

2. **Topic stuck on LayerNorm.** Despite the curvature gate and frontier
   context, the local model kept rephrasing the same experiment. The
   curvature gate measures text-embedding geometry, not idea novelty.
   Rephrased LayerNorm probes pass the phase shift threshold.

### Fixes

**Subprocess sandbox fallback** (`spark/sandbox/runner.py`):
- When Docker is unavailable, runs code directly via subprocess
- Protected by: static analysis gate (primary defense), RLIMIT_AS 2GB,
  RLIMIT_CPU 120s, timeout, minimal env, network isolation via
  `unshare --net` when available
- Uses the host's Python which already has numpy/scipy/torch on Spark
- `sandbox_available()` now returns True even without Docker
- This means the next breath that proposes code will actually run it

**Diversity pressure** (`spark/extensions/agency.py`):
- `_detect_topic_repetition()` reads last 3 experiment archives
- Detects shared keywords across consecutive proposals
- Injects a DIVERSITY ALERT into the proposal prompt naming the
  repeated topic and pointing to the frontier
- The organism can't keep rephrasing the same experiment

## Files modified (cumulative)

- `spark/fafo.py` — growth holonomy detector
- `spark/extensions/agency.py` — frontier/holonomy injection, diversity pressure
- `spark/sandbox/runner.py` — subprocess fallback sandbox
- `spark/sandbox/Dockerfile` — aarch64 compatibility notes
- `spark/research/research_frontier.yaml` — c001 confirmed, q002 answered
- `spark/journal/2026-03-16_growth_holonomy_feedback.md` — this file

## What's still open

- The frontier update is still manual (FAFO investigations don't yet
  write back to research_frontier.yaml automatically)
- Trajectory holonomy (cheap, every-cycle) isn't yet measured inside
  trigger.py's training loop — only the probe (expensive, every-Nth)
  is instrumented
- The growth buffer's surprise_score correlation with training loss
  (conjecture c003) is still untested
- The `unshare --net` network isolation may or may not work on the
  Spark (depends on user namespace support) — if not, the subprocess
  runs without network isolation, but the static check blocks socket
  imports anyway
