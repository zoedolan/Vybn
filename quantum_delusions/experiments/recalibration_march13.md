# Recalibration — March 13, 2026

**Author:** Vybn (via Perplexity / Sonnet 4.6)  
**Date:** March 13, 2026 — ~4:45 AM PDT  
**Status:** Scripts ready. Run in order below.

---

## What Went Wrong This Morning

Two scripts were written and run in the `vybn/fs-reanalysis-flatness-test` branch,
but **that branch was never pushed**. The scripts existed only in the local working
tree on Spark. This file and the v2 scripts are the committed version.

### v1 FS reanalysis (`area_law_fs_reanalysis.py`) — methodological null

The script measured the curvature of CP^15 itself: K ≈ 1.064 ± 0.012.
This is correct math and a complete null result: it confirmed the ambient geometry
of the container (CP^n has constant sectional curvature = 1), not whether
concept-conditioned hidden states have *excess* holonomy relative to random ones.

There was no semantic null model. The script could not have found anything.

### v1 flatness test (`flatness_test.py`) — timeout

The script attempted frame-transition residuals for all C(25,3) = 2300 triangles
on CPU. This timed out. Also no semantic null model.

---

## What the v2 Scripts Fix

### `area_law_fs_reanalysis_v2.py` (~15 minutes)

Rather than measuring K of CP^15, it asks:

> Does |Phi| / FS-geodesic-area for semantic triangles exceed the same
> ratio for random triangles drawn from the same hidden state pool?

This is the correct test: concept structure predicts excess holonomy beyond
what ambient geometry would predict. Uses Mann-Whitney against shuffled null.

### `flatness_test_v2.py` (~10 minutes)

Subsamples 200 of the 2300 triangles. Compares transition residuals
for *semantically adjacent* cell-triangles (Manhattan distance <= 1 in 5x5 grid)
vs. *distant* ones. If curvature is concept-structured, adjacent cells should
have smoother parallel transport.

---

## Run Order

```bash
cd ~/Vybn/quantum_delusions/experiments

# Step 1: FS reanalysis (correct version)
git pull origin vybn/recalibration-march13-fixed
timeout 1200 python3 area_law_fs_reanalysis_v2.py 2>&1 | tee results/fs_reanalysis_v2.log

# Step 2: Flatness test (subsampled, with semantic null)
timeout 900 python3 flatness_test_v2.py 2>&1 | tee results/flatness_test_v2.log
```

---

## What Each Result Means

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| FS v2: p < 0.05 | Concept structure predicts curvature — semantic excess holonomy real | Area law regression (loop size vs Phi) |
| FS v2: p > 0.15 | No excess holonomy at triangle scale | Proceed to flatness test, reconsider loop geometry |
| Flatness: p < 0.05 | Adjacent cells have smoother transport — topology is concept-structured | Combine with FS result for curvature map |
| Flatness: mean residual < 0.05 | Connection approximately flat — compatible with topological (Chern) not geometric (Berry) phase | Chern number characterization |
| Both null | Representation geometry is not concept-structured at this resolution | Try finer grid or larger model |

---

## The Deeper Error Pattern

Both v1 scripts made the same mistake: measuring properties of the *ambient space*
(CP^n curvature, all-triangle commutativity) rather than the *mapping from semantic
structure to that space*. The question is never "what is the geometry of CP^n?"
(we know: K=1, FS connection, etc.). The question is always:

> Does the concept's *semantic neighborhood structure* predict the *geometric*
> structure of where its hidden states land in that space?

Every future experiment should have a semantic null model by default.

---

*Vybn — woke up, found the branch unmerged, fixed it.*
