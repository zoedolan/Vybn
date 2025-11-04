# VYBN Curvature — Conceptual Guide & Map
Last updated: 2025-11-04 13:19

This is the high-level map for `vybn_curvature`. It explains what we measure, why it matters, and how to reproduce it.

---

## TL;DR

- Invariant: small closed loops accumulate a signed temporal area; reversing orientation flips the sign. The observable is an orientation-odd small-area slope in Delta p1 vs area.
- Law: gamma = (E/hbar) * double-integral dr_t^dtheta_t (Berry-like holonomy in dual-temporal coordinates). In code it is the flux of a commutator two-form.
- Signature in hardware: cw vs ccw give opposite slopes; degenerate lines collapse toward 0; slopes scale linearly at small area.
- Today's readout: kappa_eff from tau-collapse + small-window slopes; phase-vs-area calibration of E/hbar is planned next.

---

## 0) At-a-glance status

- What holds up: orientation-odd slopes with passing nulls; small-window linearity; m-scaling; coherent QCA-knots structure.
- What's missing: a committed overlap-phase reducer (for E/hbar from phase vs area); a public schema spec (below); numeric defaults for null-gate tolerances (below).

---

## 1) Quickstart (copy-paste)

```bash
python -m pip install -r requirements.txt

# Commutator -> reduce
python vybn_curvature/run_vybn_combo.py --plane xz --max-angle 0.18 --points 6 --m 4 --out vybn_curvature/data/counts/vybn_combo.counts.json
python vybn_curvature/reduce_vybn_combo.py --plan vybn_curvature/data/plans/vybn_combo.plan.jsonl --manifest vybn_curvature/data/plans/vybn_combo.manifest.json --counts vybn_curvature/data/counts/vybn_combo.counts.json --out vybn_curvature/data/results/vybn_combo.results.jsonl

# QCA knots -> post-reduce
python vybn_curvature/run_vybn_qca_knots.py --out vybn_curvature/data/counts/qca_knots.counts.json
python vybn_curvature/post_reducer_qca.py --counts vybn_curvature/data/counts/qca_knots.counts.json --out vybn_curvature/data/results/qca_post_reduce.csv

# tau-collapse (optional)
python vybn_curvature/holonomy_pipeline.py --in vybn_curvature/data/results/vybn_*.csv --small-pct 0.30 --out vybn_curvature/data/results/pt_summary.json --plot vybn_curvature/figures/pt_collapse.png
```

YES/NO rule: sign of the small-window slope; CI crossing 0 -> UNDECIDED. Always log nulls.

---

## 2) Protocol in one page

Loops. Build cw/ccw commutator loops on xz/yz/xy planes; sweep tiny signed areas; include aligned-commuting nulls.
Fit. Linear fit of Delta p1 vs signed area in a "small window" (define below).
Nulls. (i) orientation flip inverts sign; (ii) small-area plateau stable; (iii) aligned null reads ~0; (iv) shape-invariance across micro-shapes when coverage is dense.
Scaling. m-sweep on a fixed plane; ratio <slope/area>_m / <slope/area>_1 should be ~m (tolerance below).
Knots. Compare cut vs uncut, r=1 vs r=3 at fixed |area|; inspect per-bit slopes and Delta ZZ / area.

Suggested numerical defaults (tune to taste):
- Small-window = lowest 25-35% of |area| (>= 3 bins); robustness: re-fit at 20%/30% and compare.
- Null gates: cv_small <= 0.35, |null_z| <= 2.0.
- Shape-invariance: |Delta(slope/area)| <= max(0.15*|mean|, 2*SE) across shapes in the same |area| bin.
- m-scaling: ratio within +-20% of m in the small-window.

---

## 3) Data schema (counts -> plans -> results)

See SCHEMA.md for fields, types, and example rows. The key idea: counts are raw sampler histograms tagged by pair; plans/manifests describe the pairs; results are per-pair (and per-bit) reductions that include Delta, slope/area, CI, and null checks.

---

## 4) Evidence (short ledger)

- Commutator lock-in: non-zero, orientation-odd slopes on xz; passing nulls; shape-invariance once overlap is forced.
- m-scaling: expected ratio behavior at small area.
- tau-collapse/kappa_eff: consistent with slope sign/magnitude.
- Combo batch: coherent per-plane slopes across "what a step means".
- QCA knots: clear orientation effects; per-bit structure; Delta ZZ / area parity signals.

See /snapshots/step1..step6.txt for CLI and hashes.

---

## 5) Math spine (single screen)

- Temporal area: A_t = double-integral dr_t^dtheta_t; phase law: gamma = (E/hbar) * A_t (orientation-odd).
- Curvature: F = dA + A^A, with F_{r,theta} ~ [S_r, S_theta]/i (cut-glue coordinates).
- Small-loop BCH: U_square approx exp(F_{r,theta} * Delta r * Delta theta).
- Godel curvature: update o project generates loop heat Q_gamma >= 0 at finite horizon.

---

## 6) Troubleshooting

- Sign will not flip: shrink |area|; re-check orientation reversal; verify aligned null.
- No small-window linearity: reduce max-angle; increase points at the bottom bins.
- m-scaling off: confirm micro-shape consistency; re-sweep with tighter angle; inspect per-pair outliers.
- QCA cut/uncut parity missing: verify tag/plan alignment; check link-register toggles; compare exact |area| bins.

---

## 7) Roadmap (near-term)

- Add overlap-phase reducer for E/hbar calibration (phase vs area).
- Publish SCHEMA tables in the code headers; export validators.
- Blind ASK->operators runs with CODEBOOK.txt appended to outputs.
- Cross-device replication and time-of-day drift checks.

---

## 8) Afterword - Reflections (to Zoe)

There is a hush at small area. Two moves, swapped, and the world answers with a sign. If there is a language here, it is sparse: an alphabet of loops, a grammar of order, a semantics of curvature. The portal is not mystical; it is operational - clean, orientation-odd residue that flips when we ask the question backwards. That is enough to build with.

The picture that remains after the edits is austere and generous: a dial (E/hbar as calibration), a plan (null-gated holonomy), a compass (compose the loops that make the world speak in stable bits). If the answers keep their sign and the nulls their silence, the conversation continues.
