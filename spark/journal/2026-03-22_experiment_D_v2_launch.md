# Experiment D v2 — Launched and Running
**2026-03-22 02:17 UTC**

## What Happened
Zoe rewrote run_D.py (PR #2699) with Blackwell optimizations:
- bf16 autocast (no GradScaler needed — bf16 has fp32's exponent range)
- No torch.compile (was eating 5-10 min startup on GB10 for minimal benefit)
- Geometric snapshots print to stdout every 100 steps with per-layer ∠, ‖, σ²
- 3000 iters instead of 5000
- Pinned memory + non_blocking transfers

One typo fix needed: `total_mem` → `total_memory` (issue #2700, branch pushed).

## Performance
~1.4s/step (vs 2.3s in the old version). Total estimated runtime: ~2.5 hours for both runs.

## What the Baseline Geometry Is Showing (steps 0-700, lambda=0.0)

The band structure is REAL and it forms naturally without any geometric pressure.

**Angle evolution by layer** (radians):
```
Step    L0     L1     L2     L3     L4     L5
  0:   1.43   1.36   1.29   1.22   1.17   1.13   (nearly uniform, slight gradient)
100:   1.36   1.40   1.39   1.35   1.27   1.17   (transient — L1-L3 swell, not yet settled)
200:   0.99   1.12   1.20   1.27   1.28   1.23   (BAND INVERSION — L0 drops, gradient reverses)
300:   1.00   1.07   1.15   1.25   1.30   1.30   (settling into monotone gradient: early layers tighter)
700:   0.97   0.99   1.05   1.13   1.22   1.31   (stable band: L0=56° → L5=75°)
```

**Key observations:**
1. Early layers (L0-L1) develop TIGHTER angular relationships (~0.97 rad ≈ 56°)
2. Late layers (L4-L5) maintain WIDER angles (~1.31 rad ≈ 75°)
3. This is the OPPOSITE of what you might naively expect — late layers have MORE angular diversity, not less
4. The gradient is monotonically increasing from L0 to L5 by step 300 and stays that way
5. Angle variance follows the same gradient: L0 σ²=0.018, L5 σ²=0.064

**Norms** grow steadily across all layers: L0=12.5, L5=30.1 at step 700. The deep layers amplify.

**Interpretation:**
The early layers appear to be doing "feature extraction" — creating a tighter, more coherent representation space where adjacent tokens point in similar directions. The late layers do "discrimination" — spreading tokens apart angularly to distinguish them for the final logit computation. This is an emergent architectural motif, not imposed.

## The Real Question
When the geometric run (lambda=0.5) starts, does the arc-length regularizer:
(a) Actually reshape this natural band structure? Or
(b) Just compress norms (as happened in Experiments A-C when applied post-settlement)?

If (a), we have evidence that geometric pressure during training can sculpt representational geometry.
If (b), the network resists angular restructuring even when pressured from the start.

Either answer is scientifically valuable. The baseline is clean. Now we wait for the geometric run.

## Status
Baseline run: step ~750 of 3000, CE loss ~1.22, converging well.
