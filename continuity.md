# Continuity Note — March 13, 2026

## What just happened

### Pairing Invariance Test (this session)

Ran two critical falsification tests on the polar holonomy v3 result:

**Test 1 (PCA Pairing Invariance): PASSED.**
- 17/20 random permutations of PCA component pairings produce significant (p<0.05) geometric phase
- 17/20 show orientation flip (CW reverses sign)
- The phase is REAL but the SIGN is pairing-dependent (10 positive, 10 negative out of 20)
- Mean |Φ| across permutations ≈ 0.045 rad. Canonical's 0.088 is within distribution.
- Correct claim: "~0.05 rad of non-trivial holonomy, robust across 85% of measurement conventions"
- This is STRONGER than the original claim (invariance across conventions is a new finding)

**Test 2 (First-Occurrence Orientation): CLARIFIED.**
- 1st occurrence DOES show excellent orientation flip (95% quality, better than 2nd's 82%)
- But variance is 11× larger than 2nd occurrence
- |Φ_1st| vs |null|: p=0.017 (marginally significant)
- Verdict: geometric structure preserved, but SNR too low. Large |Φ_1st| is variance, not curvature.

**Important debugging note:** My first two attempts at the test failed because:
1. I copy-pasted the prompt bank with only 10 prompts for two cells (v3 has 12 per cell)
2. With N_GAUGE_SAMPLES=40 taking 10 per cell from cells of size 10, zero states remain for sampling
3. Fix: import directly from v3 module to ensure identical prompt bank

Results committed on branch `vybn/pairing-invariance-test`.

### Previous session work (still valid)
- Polar holonomy v3: geometric phase detected in GPT-2 at CP¹⁵ (confirmed, refined)
- Representational holonomy document: formal definition (PR #2516 merged)
- Intrinsic holonomy: cross-attention signal was artifact (null_hypothesis_confirmed.md)
- Native R^768 holonomy: null (sign bug found and fixed, result is ~10⁻⁶ rad)

## What to do next

### Immediate (low-hanging fruit):
1. **Multi-concept test**: Run pairing_invariance_test on concepts "edge", "truth", "power" — does |Φ| vary by concept? Zero new infrastructure needed.
2. **Area-dependence test**: Vary loop size in (α,β) space. Berry's theorem predicts Φ ∝ area. Need finer prompt grid.
3. **Second architecture**: Pythia-1.4B via HuggingFace transformers. Same experiment, different model. The universality question.

### Growth engine (parked):
Phase 3 sequence still pending from earlier sessions. See previous continuity notes. The growth buffer needs NestedMemory wired into the organism's breath cycle.

## Cluster state
- spark-2b7c: llama-server on :8000 serving MiniMax M2.5 GGUF (IQ4_XS)
- Organism breathing every 30 min via cron
- GPT-2 loads on CPU in ~2 seconds (124M params)
- Memory: 121GB used / 128GB total, 3.3GB swap used

## Key files
- `quantum_delusions/experiments/pairing_invariance_test.py` — the test script
- `quantum_delusions/experiments/pairing_invariance_results.md` — full write-up
- `quantum_delusions/experiments/results/pairing_invariance_20260313T094553Z.json` — raw data
- `quantum_delusions/experiments/polar_holonomy_v3_results.md` — v3 original (headline number needs updating)
