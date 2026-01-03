# vybn_curvature — Unified README (Consolidated)
Last updated: 2025-11-04 14:29

This single README **consolidates** the essentials from prior docs (Guide, Codebook, Schema, Theory Crosswalk, Glossary) into one file. It is written so a first-time human or AI can understand **what to run, what to look for, and how the fundamental theory maps into the code and data**.

> TL;DR: Small closed loops in control space exhibit an **orientation-odd, small-area effect** in hardware.
> We read it today as the slope of Delta p1 = (p1_cw - p1_ccw) vs **signed area** (small-window). The conceptual
> law is gamma ~ (E/hbar) * Area_t (phase vs oriented temporal area). Phase-vs-area calibration is planned next.

---

## 0) Quickstart (copy/paste first, think later)

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

**YES / NO / UNDECIDED**  
- Fit the small-window slope of Delta p1 vs signed area (lowest ~25–35% of |area|, >= 3 bins).  
- YES = positive slope; NO = negative; UNDECIDED = 95% CI crosses 0.  
- **Nulls must pass**: (i) orientation flip inverts sign; (ii) aligned-commuting leg ~ 0; (iii) stable small-area plateau; (iv) shape-invariance when coverage is dense.

Suggested defaults (tune later): `cv_small <= 0.35`, `|null_z| <= 2.0`, shape-invariance tolerance `|Delta(slope/area)| <= max(0.15*|mean|, 2*SE)`, m-scaling within +-20% of m.

---

## 1) What this folder proves (operational statement)

- **Orientation-odd residue exists**: cw vs ccw small loops on the same plane produce opposite-signed slopes in the small-window.  
- **Linearity at tiny area**: slopes scale linearly with signed area near the origin; degenerate lines collapse toward 0.  
- **m-scaling appears**: multiplicity m scales the small-window slope ~ linearly (within tolerance).  
- **QCA knots show parity**: cut vs uncut (and r=1 vs r=3) reveal characteristic structure per bit and in dZZ/area.

---

## 2) Theory -> Practice (one-table crosswalk)

| Theory anchor (see papers/) | Key object | What to check in lab | Script(s) | Data field(s) | Falsifier |
|---|---|---|---|---|---|
| Temporal Holonomy (unified) | gamma = (E/hbar)*A_t | small-window slope sign at tiny area; later phase vs area | run_vybn_combo.py -> reduce_vybn_combo.py | slope_per_area, delta | no orientation-odd slope |
| Polar Temporal Coordinates | dual-time -> Bloch U(1) | linearity + orientation flip behavior | same | slope panels, nulls | nonlinearity persists at tiny area |
| Dual-Temporal Holonomy Thm | probe holonomy = Berry area | tau-collapse -> kappa_eff; later phase match | holonomy_pipeline.py | pt_summary.json, pt_collapse.png | mismatch to Berry area (once phase pipeline exists) |
| Cut-Glue (+foundations) | F ~ [S_r, S_theta] | commutator residue with null gates | run_vybn_combo.py -> reduce_vybn_combo.py | slope_per_area + null_z/null_pass | sign won’t invert; nulls fail |
| Godel Curvature | loop heat >= 0 | kappa_eff sign/magnitude across windows | holonomy_pipeline.py | pt_summary.json | negative loop-heat proxy robustly |
| Knot-a-Loop | engine-stage-self, parity | cut vs uncut, r parity at fixed |area| | run_vybn_qca_knots.py -> post_reducer_qca.py | per-bit slopes; dZZ/area | parity absent |
| CGT (Vybn conjecture) | expensive flux | schedule choice lowers cost for same signal | any runner | runtime/variance vs slope | no gain switching predicted cheaper frames |

---

## 3) Micro-protocol (how to ask clean questions)

- **Planes & loops:** xz/yz/xy commutator; sweep tiny signed areas with micro-shapes; always include aligned nulls.  
- **Fit window:** bottom 25–35% |area|; re-fit at 20%/30% for robustness.  
- **Shape-invariance:** slopes across micro-shapes should agree in the same |area| bin (within tolerance).  
- **m-scaling:** ratios ~ m in the small-window (within +-20%).  
- **QCA knots:** fixed |area| comparisons for cut/uncut and r variants; report per-bit slopes and dZZ/area.

**Codebook (mini):**  
- `rest`  -> commuting legs (expect ~0)  
- `become` -> small misaligned tilt (expect nonzero slope)  
- `not`   -> orientation reversal or order swap (flip sign)  
- `thermal?` -> tiny thermal/dephasing leg; sign tracks kappa_eff

---

## 4) Data schema (8-line contract)

- counts (*.counts.json): `{{tag, orientation:{cw|ccw}, shots, hist{{bitstring:count}}, meta}}`  
- plans/manifests: `pairs:[{{tag, plane, area, shots, ...}}], names:[...]`  
- vybn_combo.results.jsonl: per pair `{tag, plane, area, micro_shape, delta, slope_per_area, slope_se, same_sign, cv_small, null_z, null_pass}`  
- qca_post_reduce.csv/json: per knot `{family, area, width, r, cut, slope_per_area_bit0/1/2, zz01_cw/ccw, zz12_cw/ccw, dZZ01_per_area, dZZ12_per_area, meta}`  
- pt_summary.json: tau-collapse summary (and kappa_eff)  
- figures/pt_collapse.png: tau-normalized collapse plot

---

## 5) Mindmap

<img width="1128" height="1566" alt="mermaid-diagram-2025-11-04-065451" src="https://github.com/user-attachments/assets/eb08db50-ff86-48d8-8e1d-a44b28138bdb" />

---

## 6) Troubleshooting (fail fast)

- **Sign will not flip:** shrink |area|; verify orientation reversal; run aligned null.  
- **No small-window linearity:** reduce max-angle; add bottom-bin points; re-fit at 20/30%.  
- **m-scaling off:** check micro-shape consistency; retune angle; inspect per-pair outliers.  
- **QCA parity missing:** confirm tag/plan alignment; check link-register toggles; ensure same |area| bins.

---

## 7) Papers (local index)

The papers live under `vybn_curvature/papers/`. Filenames may vary; look for:
- temporal-holonomy-unified-theory.md
- polar_temporal_coordinates_qm_gr_reconciliation.md
- dual_temporal_holonomy_theorem.md
- cut-glue-unified-theory.md (+ mathematical_foundations_companion.md)
- godel_curvature_thermodynamics.md
- knot-a-loop-unified-theory-final.md
- vybn-conjecture-computational-geometry.md

---

## 8) Roadmap

- Add overlap-phase reducer for **E/hbar** calibration (phase vs area) and slot its CLI here.  
- Export JSON schema validators for counts/plans/results.  
- Blind ASK->operators runs with CODEBOOK rules appended to outputs.  
- Cross-device replication; time-of-day drift checks.

---

## 9) One sentence to remember

Two moves, swapped, speak. The sign is the answer; the nulls keep us honest.
