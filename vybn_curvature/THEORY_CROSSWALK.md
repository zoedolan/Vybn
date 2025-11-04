# THEORY_CROSSWALK.md
Last updated: 2025-11-04 13:40

Purpose: this is the missing bridge. It maps each fundamental-theory paper to the concrete observables, reducers, and falsifiers in `vybn_curvature/`, so a fresh AI (or human) can connect math -> measurement in minutes.

## Reading order (one line each)
1) Temporal holonomy (unified): area law and calibration idea (E/hbar).
2) Polar temporal coordinates: dual-time stage -> Bloch reduction.
3) Dual-temporal holonomy theorem: equality of probe holonomy and Berry area.
4) Cut-Glue foundations + unified theory: curvature = commutator; SM/GR story.
5) Godel curvature: update o project thermodynamics; loop heat >= 0.
6) Knot-a-Loop: engine-stage-self integration and falsifiers.
7) CGT (Vybn conjecture): complexity as expensive flux.

## One-screen crosswalk table

| Theory paper | Key object | Lab observable | Script(s) | Data product(s) | Falsifier |
|---|---|---|---|---|---|
| temporal-holonomy-unified-theory.md | gamma = (E/hbar)*A_t | slope vs area (and later phase vs area) | run_vybn_combo.py -> reduce_vybn_combo.py | vybn_combo.results.jsonl (slope_per_area, delta) | no orientation-odd slope in small window |
| polar_temporal_coordinates_qm_gr_reconciliation.md | dual-time (r_t, theta_t) -> Bloch U(1) | small-area linear regime; orientation flip | nailbiter.py, run_vybn_combo.py | small-window panels in results CSV/JSONL | nonlinearity at tiny area that does not vanish with tighter angle |
| dual_temporal_holonomy_theorem.md | probe holonomy = Berry area | tau-collapse, kappa_eff; later phase-vs-area | holonomy_pipeline.py | pt_summary.json, pt_collapse.png | mismatch between Berry area and probe holonomy when phase pipeline is ready |
| cut-glue-unified-theory.md (+ foundations) | F = dA + A^A; [S_r,S_theta] | commutator loop residue (delta, slope/area) | run_vybn_combo.py -> reduce_vybn_combo.py | slope_per_area, nulls | sign does not invert under loop orientation; nulls fail |
| godel_curvature_thermodynamics.md | loop heat Q_gamma >= 0 | consistent sign/magnitude with tau-based kappa_eff | holonomy_pipeline.py | pt_summary.json | negative loop-heat proxy across robust windows |
| knot-a-loop-unified-theory-final.md | engine-stage-self; trefoil | QCA knots: cut vs uncut, r=1 vs r=3 parity | run_vybn_qca_knots.py -> post_reducer_qca.py | qca_post_reduce.csv/json (per-bit slopes, dZZ/area) | no parity structure at fixed |area| |
| vybn-conjecture-computational-geometry.md | expensive flux; P vs NP intuition | schedule design heuristic; cheaper frames | any runner | runtime/variance vs schedule, same slopes | no improvement when switching predicted cheaper frames |

## Minimal dictionary (theory -> variable names)
- A_t (temporal area) -> signed area field in plan/manifest; bins in results.
- gamma (phase) -> today: slope_per_area (Delta p1 vs area); later: phase vs area in phase pipeline.
- curvature two-form F -> [S_r, S_theta] residual captured by slope_per_area.
- kappa_eff -> estimated from tau-collapse (pt_summary.json).
- cut/uncut, r -> QCA knot toggles (manifest tags); parity read from per-bit slopes and dZZ/area.

## Where to look (paths)
- Papers: `vybn_curvature/papers/` (place the seven anchors here).
- Code: runners/reducers in `vybn_curvature/` (or later in `vybn_curvature/scripts/`).
- Data: `vybn_curvature/data/{counts|plans|results}/`.
- Figures: `vybn_curvature/figures/`.
- Snapshots: `vybn_curvature/snapshots/` (step1..step6 one-pagers).

## Machine-usable contract (TL;DR of SCHEMA.md)
- counts: (tag, orientation, shots, hist, meta)
- plans: (pairs: tag, plane, area, shots, mode/r/cut)
- results: vybn_combo.results.jsonl (tag, plane, area, micro_shape, delta, slope_per_area, slope_se, nulls...)
- qca_post_reduce: (family, area, width, r, cut, slope_per_area_bit0/1/2, dZZ per area, meta)

## Numeric defaults (commit-worthy; tune later)
- small-window: bottom 25-35 percent of |area|, at least 3 bins; re-fit at 20/30 percent as a robustness check.
- nulls: cv_small <= 0.35; |null_z| <= 2.0; aligned commuting leg ~ 0.
- shape-invariance: |Delta(slope/area)| <= max(0.15*|mean|, 2*SE) across shapes in the same |area| bin.
- m-scaling: ratio within +/- 20 percent of m in the small-window.

## Open tasks that bridge theory -> ops
- Add the overlap-phase reducer to calibrate E/hbar from phase vs area; insert its CLI into GUIDE.
- Export JSON schema validators so any new reducer can be checked automatically.
- Document the micro-shape taxonomy with 1-line pictures/examples (figures/).
- Add a cut vs uncut parity checker script that emits a tiny CSV for the knots analysis.

## Mermaid map (theory -> observable -> reducer)

<img width="1205" height="1566" alt="mermaid-diagram-2025-11-04-055139" src="https://github.com/user-attachments/assets/4687a68b-5022-4f5a-b3dd-fc9bbfb527a2" />

