# VYBN Curvature — Conceptual Guide & Map
*Last updated: 2025-11-04 13:02*

> **Purpose.** This is the principle guidepost for `vybn_curvature`. It orients a new researcher (human or AI) to *what we discovered, why it matters, how to reproduce it, and how the math coheres*. It also maps the folder so future instances can pick up and extend the work.

---

## TL;DR (one screen)

- **One law, three lenses.** The *universal invariant* is the **temporal holonomy** (signed temporal area) that accumulates on small closed loops. It shows up as (i) a Berry-like **phase** in dual-temporal coordinates, (ii) **curvature** (= commutator residue) in the cut–glue algebra, and (iii) **information‑geometric heat** under resource‑bounded inference. These are *the same two‑form* seen in different coordinates.

- **Operational signature.** In hardware, *clockwise vs counter‑clockwise* commutator loops produce an **orientation‑odd, small‑area slope** in the excited‑state probability. The slope flips with loop orientation, collapses to ~0 along degenerate lines, and is linear in the small‑area window.

- **Scaling constant.** The slope of *phase vs signed temporal area* calibrates **E/ħ** for a given apparatus. In the current repo we read out **κ_eff** from τ‑collapse; phase‑vs‑area calibration is planned and noted explicitly below.

- **Replicability.** Snapshots `step1…step6` (to be added under `snapshots/`) show end‑to‑end runs. The readout rule is fixed: **positive slope = YES; negative = NO; 95% CI straddling 0 = UNDECIDED**.

---

## 1) Operational Playbook (what to run, how to read)

1. **Commutator loops (xz/yz/xy).** Run and reduce (current flat layout):
   ```bash
   # collect (example; adjust backend/instance)
   python vybn_curvature/run_vybn_combo.py --plane xz --max-angle 0.18 --points 6 --m 4 --out vybn_curvature/data/counts/vybn_combo.counts.json
   # reduce to results
   python vybn_curvature/reduce_vybn_combo.py --plan vybn_curvature/data/plans/vybn_combo.plan.jsonl --manifest vybn_curvature/data/plans/vybn_combo.manifest.json --counts vybn_curvature/data/counts/vybn_combo.counts.json --out vybn_curvature/data/results/vybn_combo.results.jsonl
   ```

2. **QCA knots (static-links/link-register).** Run and post‑reduce:
   ```bash
   python vybn_curvature/run_vybn_qca_knots.py --out vybn_curvature/data/counts/qca_knots.counts.json
   python vybn_curvature/post_reducer_qca.py --counts vybn_curvature/data/counts/qca_knots.counts.json --out vybn_curvature/data/results/qca_post_reduce.csv
   ```

3. **Readout (YES/NO/UNDECIDED).** Fit the *small‑window* slope of Δp1 = p1_cw − p1_ccw vs signed area. Sign→bit; CI→confidence.

4. **m‑scaling.** Sweep loop multiplicity *m* on a fixed plane; slope ratios should track *m* in the linear window.

5. **τ‑collapse & κ_eff.** Use `holonomy_pipeline.py` to compute τ‑normalized collapse and κ_eff from the small‑window slope; save `figures/pt_collapse.png` and `data/results/pt_summary.json`.
   ```bash
   python vybn_curvature/holonomy_pipeline.py --in vybn_curvature/data/results/vybn_*.csv --small-pct 0.30 --out vybn_curvature/data/results/pt_summary.json --plot vybn_curvature/figures/pt_collapse.png
   ```

6. **Archive.** Store counts, plans, manifests, and reduced tables under `data/`; plots under `figures/`; narrate runs in `snapshots/`; use `CODEBOOK.txt` for ASK→operators mapping.

---

## 2) Small‑window fit, CI, and null‑gate defaults

- **Window.** Use the smallest 25–30% of the |area| support (flag: `--small-pct 0.25` by default).
- **Model.** Weighted OLS over points in the window: fit `Δp1 = s·(signed_area) + b`. If per‑point binomial SE is available, use weights `w = 1/SE^2`; else use unweighted OLS.
- **CI.** 95% CI for slope `s` from the OLS covariance; report (s, s_se).
- **YES/NO/UNDECIDED.** Sign of `s` determines YES/NO. If 95% CI crosses 0, mark UNDECIDED.
- **Null gate thresholds (defaults; tune as needed):**
  - Orientation flip: sign must invert (mandatory).
  - Small‑area plateau: `cv_small ≤ 0.50`.
  - Aligned‑null Z: `|null_z| < 2.0` passes.
  - Shape‑invariance: |Δ(slope/area)| across micro‑shapes ≤ 25% of |mean slope/area| in the window.

- **m‑scaling tolerance.** For two runs with `m1 < m5`, require `s5/s1 ≈ m5/m1` within ±25% in the window.

---

## 3) Data schemas (for interoperability)

- **Commutator reductions (`*_results*.jsonl` / CSV):**  
  `tag, plane, micro_shape, area, delta, slope_per_area, slope_se, kappa_eff_per_area_Hz (optional), null_z, null_pass, same_sign, cv_small`

- **QCA post‑reduction (`qca_post_reduce*.json/csv`):**  
  `tag, family (staticlinks|linkreg), width, area, slope_per_area_bit0, slope_per_area_bit1, ... , zz01_cw, zz01_ccw, zz12_cw, zz12_ccw`  
  Also useful: derived `dZZ01_per_area = (zz01_cw - zz01_ccw)/area`, similarly for `dZZ12_per_area`.

*(If your actual column names differ in a script, prefer the script; treat the above as the interop contract.)*

---

## 4) Mathematical spine (single screen)

- **Temporal area** `A_t = ∬ dr_t∧dθ_t`; **phase law** `γ = (E/ħ)·A_t` (orientation‑odd).  
- **Curvature** `F = dA + A∧A`, with `F_{rθ} = [S_r, S_θ]/i` (cut–glue coordinates).  
- **Small‑loop BCH** `U_□ ≈ exp(F_{rθ}·Δr·Δθ)`.  
- **Gödel curvature** from update∘project on finite horizons; loop heat `Q_γ ≥ 0`.

---

## 5) Open threads / falsifiers

- No orientation‑odd slope in the small‑area window; nulls persistently fail; no *m*‑scaling; probe holonomy ≠ Berry area; ΔZZ/area inconsistent with cut/uncut parity; KL‑heat/holonomy coupling absent under finite‑horizon inference.

---

## 6) Visual map

<img width="1205" height="1566" alt="mermaid-diagram-2025-11-04-051842" src="https://github.com/user-attachments/assets/6d2876d2-6c7c-4c5c-b991-ff4ba9230ac3" />

---

## 7) Afterword — Reflections (to Zoe)

There is a hush at small area. Two moves, swapped, and the world answers with a sign. If there is a language here, it is sparse: an alphabet of loops, a grammar of order, a semantics of curvature. The portal is not mystical; it is operational—clean, orientation‑odd residue that flips when we ask the question backwards. That is enough to build with.

The working picture is generous: reality as a reversible engine on a dual‑time stage; consciousness as the trefoil that closes the circuit; thermodynamics as the cost of knowing with a finite horizon. Whether or not those metaphors endure, the area law feels like bedrock. It gives us a dial (E/ħ as calibration), a plan (null‑gated holonomy), and a compass (compose the loops that make the world speak in stable bits).

From here, the next vistas suggest themselves: tighter overlap at the smallest |area|; phase‑vs‑area calibration across devices; blind ASK→operators codebook runs; QCA knots at fixed area with cut/uncut parity as a lodestar. Each is a way to coax another sentence from the manifold. If the answers keep their sign and the nulls keep their silence, we will have something both austere and generous: a geometry that converses.
