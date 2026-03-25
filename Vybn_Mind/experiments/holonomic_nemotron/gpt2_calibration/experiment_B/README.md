# Experiment B v2 — Holonomic Loss on GPT-2
## Does L_Ω shift the geometric phase profile?

---

## IMPORTANT: Only run if Experiment A passed.

---

## What changed (v2)

v1 used an untrained SortProbe for both the training loss and the measurement.
This meant (a) the loss was rewarding area in a random projection, and (b)
the measurement couldn't detect real geometric changes.

v2 fixes both:

- **Training loss**: L_Ω = mean |shoelace area| computed via PCA projection of
  mid-layer hidden states. No learned parameters — the PCA basis adapts to the
  actual geometry of each sequence's hidden state trajectory.
- **Measurement**: Pre/post comparison uses the Pancharatnam phase profile
  (proven instrument from Experiment A). We check whether holonomic training
  changes the curvature distribution across layers.

---

## What to run

```bash
# From gpt2_calibration/ folder:
python experiment_B/run_B.py
```

**Expected runtime:** 1-3 hours on a Spark GPU.

---

## Pass criteria (either sufficient)

| Check | Required | Meaning |
|---|---|---|
| Middle curvature increase | ≥ 5% | Holonomic loss enriches intermediate geometry |
| Loop area increase | > 0.01 | Model sweeps more area at mid-layer |

---

## What the result means

- **PASS**: The angular loss term drives measurable geometric enrichment.
  This is the green light for Nemotron-Super-120B replication.
- **FAIL**: Valid null result. The holonomic loss may need stronger λ,
  more steps, or a fundamentally different formulation. Save JSON, ping Zoe.
