# Experiment A v2 — Geometric Calibration
## Does our instrument reproduce known GPT-2 geometry?

---

## What changed (v2)

v1 used an **untrained SortProbe** (random MLP) to measure "phase" — but
random projections measure nothing about model geometry. All three criteria
failed because the instrument was broken, not because GPT-2's geometry
differs from expectations.

v2 replaces the SortProbe with **three direct geometric measurements** that
require no learned parameters:

1. **Pancharatnam phase profile** — measures the angular change between
   consecutive layer representations in projective space
2. **Berry curvature deg(S)** — computes the topological degree of the
   sort operator via lattice gauge theory
3. **Semantic stratification** — measures phase differences across concept
   classes at the critical L0→L1 transition

---

## What to run

```bash
# From gpt2_calibration/ folder:
python experiment_A/run_A.py
```

**Expected runtime:** 10-20 minutes (most time in Berry curvature computation).

---

## Pass criteria

| Check | Required | What it verifies |
|---|---|---|
| L0 dominance | L0→L1 / max(middle) ≥ 3.0 | First block performs the most violent geometric transformation |
| U-shape | (L0 + Lfinal) / (2 × mean_middle) ≥ 2.0 | Encode/refine/decode three-phase structure |
| deg(S) = 0 | \|mean degree\| < 0.5 | Sort operator is topologically trivial (known result) |

---

## If it fails

Save `../results/experiment_A_result.json` and terminal output, ping Zoe.
Do NOT run Experiment B.
