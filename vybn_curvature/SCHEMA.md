# SCHEMA (counts -> plans -> results)

Minimal field descriptions so new reducers can interoperate. Examples are indicative; exact keys may evolve with scripts.

## counts (*.counts.json)
- run_tag: string — identifier for the batch
- tag: string — unique pair key (plane, area, shape, mode)
- orientation: enum {cw, ccw}
- shots: int
- hist: object — {bitstring: count}
- meta: object — backend, instance, seed, timestamp

## plans (*.plan.jsonl) and manifests (*.manifest.json)
- pairs: array of objects with at least:
  - tag: string — id (plane, |area|, micro_shape, mode)
  - plane: enum {xz, yz, xy, qca, qca_linkreg}
  - area: float — signed or absolute area parameter
  - shots: int — target shots
  - r / mode / cut: optional — QCA/Knot annotations
- names: array — canonical order of pair names (when used by runner)

## results
### vybn_combo.results.jsonl
- tag, plane, area, micro_shape
- delta: float — Delta p1 = p1_cw - p1_ccw
- slope_per_area: float — small-window slope
- slope_se: float — standard error (if computed)
- same_sign, cv_small, null_z, null_pass: quality gates

### qca_post_reduce.csv/json
- family: enum {staticlinks, linkregister, ...}
- area, width, r, cut
- slope_per_area_bit0/1/2: float — per-bit small-window slopes
- zz01_cw, zz01_ccw, zz12_cw, zz12_ccw: floats — nearest-neighbor ZZ
- dZZ01_per_area, dZZ12_per_area: floats — orientation-odd ZZ deltas / area
- meta: backend, instance, timestamp

Notes
-----
- Small-window definition: bottom 25-35% of |area| bins (>= 3 bins). Re-fit at 20%/30% for robustness.
- CI: OLS with robust SE is acceptable; consistency across nearby windows preferred.
- Shape-invariance: compare slopes across micro-shapes within the same |area| bin.
