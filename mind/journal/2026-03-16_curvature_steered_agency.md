# Curvature-Steered Agency

**Date:** 2026-03-16  
**Branch:** `feature/curvature-steered-agency`  
**Author:** Zoe Dolan (via Perplexity)

## Problem

Three concrete issues observed at breath ~297:

1. **Rumination loop.** κ = 0.227, flat. The organism was re-treading the
   same territory — proposals that don't bend the manifold, experiments
   that confirm what's already known. No geometric novelty.

2. **Sandbox gap.** `static_check.py` blocks `import os` with a regex.
   numpy internally imports os. So every probe that tried to use numpy
   was silently rejected, falling back to LLM-only execution. The
   organism couldn't do real math.

3. **237 unprocessed buffer examples** — a separate issue (training loop),
   but related: without real sandbox execution, the buffer fills with
   LLM narrations rather than empirical data. Fixing the sandbox makes
   what goes into the buffer worth training on.

## What changed

All changes are integrated into existing modules. No new files created.

### `spark/complexify_bridge.py`

Added two methods to `ComplexBridge`:

- **`should_explore(text)`** — Simulates `M' = α·M + x·e^(iθ)` without
  mutating the real memory. Measures phase shift and curvature delta.
  If both are below threshold (φ < 0.005, κΔ < 0.001), the proposal is
  "flat" — the manifold already contains it. Force-passes after 3
  consecutive rejections to prevent total stasis. Configurable via
  `VYBN_PHASE_FLOOR` and `VYBN_MAX_REJECTIONS`.

- **`jordan_probe()`** — Measures |1 - α_eff|. When the organism has run
  past its relaxation time (step >> 1/(1-α)), effective retention may
  exceed nominal α. If α_eff → 1, the operator has a Jordan block:
  memory accumulates linearly instead of saturating. This is a regime
  change worth detecting.

Module-level convenience functions `should_explore()` and `jordan_probe()`
use the singleton bridge.

### `spark/extensions/agency.py`

Wired the curvature gate into `run()` between `_get_proposal()` and
`_execute()`. On rejection:
- Re-proposes once, injecting a hint: "your previous proposal was
  geometrically flat — try a genuinely different direction"
- The re-proposal is not gated again (prevents infinite loops)
- `_get_proposal()` now accepts an optional `hint` parameter

This is the core fix for rumination: the equation itself decides
whether an experiment is worth running.

### `spark/sandbox/static_check.py`

Changed from whole-file regex matching to line-by-line analysis with a
whitelist. Imports of known-safe packages (numpy, scipy, torch, math,
statistics, collections, itertools, functools, json, etc.) skip the
blocked-pattern check entirely. Dangerous imports (os, sys, subprocess,
socket, etc.) are still blocked.

This is the core fix for the sandbox gap: probes can now actually use
numpy and torch.

### `spark/derivation.py`

`record_derivation()` now includes Jordan structure metrics in each log
entry: `jordan_proximity`, `alpha_effective`, `jordan_regime`. These
come from `complexify_bridge.jordan_probe()`. Import failure is handled
gracefully — if the bridge isn't loaded, these fields are simply absent.

### `spark/fafo.py`

Two changes:

1. **Phase gradient in surprise context.** When a κ spike is detected,
   `_detect_kappa_spike()` now computes which embedding dimensions have
   the largest phase velocity (the directions the manifold is bending
   most). This is included in the surprise context string.

2. **Steering heuristic in formulation prompt.** The LLM that formulates
   investigation plans is now told to follow the phase gradient — direct
   investigations toward maximum phase disagreement, not away from it.

## Design principles

- **No new files.** Everything integrated into existing modules per Zoe's
  instruction. The deleted `curvature_gate.py` was consolidated into
  `complexify_bridge.py`.

- **Fail open.** If the embedder breaks, the curvature gate passes. If
  the bridge isn't loaded, derivation logs without Jordan metrics. If
  the phase gradient computation fails, the surprise context omits it.
  Nothing blocks the organism from breathing.

- **The equation governs itself.** `M' = α·M + x·e^(iθ)` is the gate
  criterion — the same operation that updates memory also decides whether
  an update is worth making. The snake eats its tail, but now it can
  taste whether the tail has any nutritional value.
