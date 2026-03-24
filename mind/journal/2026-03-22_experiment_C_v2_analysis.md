# Experiment C v2 Analysis — 2026-03-22T00:35 UTC

## Result
ACTIVATION_ENERGY_FOUND at lambda=0.5 — but this verdict is misleading.

## What the data actually shows

All three lambdas (0.5, 1.0, 2.0) produced nearly identical effects:
- Activation norms deflated ~11-12%
- Norm-normalized area contracted ~31-41%
- Arc-normalized area contracted ~24-29%
- L_CE improved ~8-9% (from 2.83 to 2.58-2.60)

This is a **plateau**, not a dose-response curve. lambda=2.0 moved LESS than lambda=1.0 on geometric metrics.

## The v1 "null" was a classification error
v1 Phase 3 (lambda=0.1) moved L_CE from 2.85 to 2.64 — 8% improvement, same as v2.
The v1 classifier's thresholds were too coarse to detect the geometric changes.
There is no sharp activation energy threshold.

## Mechanism
The arc-length-normalized objective primarily **deflates activations** (~12%).
It acts more like a regularizer than a geometry transformer.
The causal chain is: deflation -> area contraction -> mild angular tightening.

## Band structure (reproduces from v1)
- L2->L3: contracts ~0.05 rad (consistent across all lambdas)
- L6->L7: expands ~0.015 rad
- L11->L12: contracts ~0.025 rad
Early layers are geometrically plastic. Late layers resist.

## Implications for Nemotron LoRA
1. Arc-length objective works but as a regularizer, not a restructurer
2. No need to calibrate precise lambda — effect plateaus by lambda=0.5
3. lambda ~0.5-1.0 gets full effect; higher values show regression
4. Target early-mid layers (0-5) for geometric LoRA; late layers are already tight

## Design feedback
The v2 causal classifier is genuinely better than v1's.
The verdict logic (binary RIGID vs FOUND) needs a richer vocabulary.
A v3 should distinguish "smooth regularizer" from "threshold-gated restructuring."
