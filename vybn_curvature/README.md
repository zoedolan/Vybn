# vybn_curvature

**Status:** active, WIP • **Last updated:** 2025-11-04 13:19

This folder hosts the Vybn curvature program: experiments and reducers that expose an orientation-odd small-area residue (temporal holonomy) in commutator/QCA loops, plus documents that explain the math and show how to reproduce the results.

Read first: -> [GUIDE.md](./GUIDE.md)

---

## Quickstart (5 minutes)

1. Install
   ```bash
   python -m pip install -r requirements.txt
   ```

2. Sanity check the runners
   ```bash
   python vybn_curvature/run_vybn_combo.py --help
   python vybn_curvature/post_reducer_qca.py --help
   python vybn_curvature/reduce_vybn_combo.py --help
   python vybn_curvature/holonomy_pipeline.py --help
   ```

3. Minimal end-to-end (commutator)
   ```bash
   # collect counts (example flags; tune for your backend/instance)
   python vybn_curvature/run_vybn_combo.py --plane xz --max-angle 0.18 --points 6 --m 4 --out vybn_curvature/data/counts/vybn_combo.counts.json

   # reduce -> per-pair rows with delta and slope/area
   python vybn_curvature/reduce_vybn_combo.py --plan vybn_curvature/data/plans/vybn_combo.plan.jsonl --manifest vybn_curvature/data/plans/vybn_combo.manifest.json --counts vybn_curvature/data/counts/vybn_combo.counts.json --out vybn_curvature/data/results/vybn_combo.results.jsonl
   ```

4. Post-reduce (QCA knots)
   ```bash
   python vybn_curvature/run_vybn_qca_knots.py --out vybn_curvature/data/counts/qca_knots.counts.json
   python vybn_curvature/post_reducer_qca.py --counts vybn_curvature/data/counts/qca_knots.counts.json --out vybn_curvature/data/results/qca_post_reduce.csv
   ```

5. tau-collapse (optional)
   ```bash
   python vybn_curvature/holonomy_pipeline.py --in vybn_curvature/data/results/vybn_*.csv --small-pct 0.30 --out vybn_curvature/data/results/pt_summary.json --plot vybn_curvature/figures/pt_collapse.png
   ```

---

## What is a YES/NO here?

- Fit the small-window slope of Delta p1 = (p1_cw - p1_ccw) vs signed area.
- Positive slope = YES; negative = NO; 95% CI straddling 0 = UNDECIDED.
- Enforce nulls: orientation flip inverts sign; aligned-commuting leg -> ~0; stable small-area plateau; shape-invariance in dense coverage.

See details in GUIDE.md and CODEBOOK.txt.

---

## Folder map

```
vybn_curvature/
  README.md, GUIDE.md, CODEBOOK.txt, SCHEMA.md
  run_vybn_combo.py           # driver (commutator + variants)
  reduce_vybn_combo.py        # reducer for vybn_combo counts
  run_vybn_qca_knots.py       # driver (QCA knots family)
  post_reducer_qca.py         # reducer for QCA counts
  holonomy_pipeline.py        # tau-collapse, kappa_eff
  holonomy_fixedpoint.py      # helper
  data/ {counts|plans|results}
  figures/ {pt_collapse.png, ...}
  snapshots/ {step1..step6 one-pagers}
  papers/ {seven conceptual anchors}
```

If something fails or looks off, jump to Troubleshooting in GUIDE.md.
