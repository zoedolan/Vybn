# Continuity Note — Holonomy Loss Hypothesis: Corpus Study Complete

*Updated: 2026-03-12, by outside-Vybn*

## What just happened

1. **Nemotron-3-Super-120B-A12B-FP8 serving on PP=2** via fastsafetensors across both Sparks. Healthy, tested, port 8000.

2. **PR #2509 merged**: holonomy_scorer wired into growth_buffer. Data curation now blends surprise + holonomy (50/50 default).

3. **Corpus holonomy study (N=8)**: Spearman ρ = −0.78, p = 0.022. The extrinsic-intrinsic convergence replicates with automated concept selection. Branch `vybn/corpus-holonomy-n8`, issue #2510.

## Key discovery

**Concept selection matters for the intrinsic measurement.** Hand-picked concepts get z ≈ −5 to −6 for deep texts. Automated (biggest-gap non-common token) gets z ≈ −0.7 to −1.0. The signal is real but concept-specific. Future intrinsic work should track multiple concepts per text and aggregate.

## The holonomy hypothesis status

| Level | Status | Next step |
|-------|--------|-----------|
| 1. Data curation | ✅ Wired into growth buffer | Verify during next growth cycle |
| 2. Evaluation metric | ✅ Validated (ρ=−0.78, p=0.022, N=8) | Larger corpus when more journal entries accumulate |
| 3. Auxiliary loss | Not started | Multi-concept intrinsic aggregation first |

## Full extrinsic rankings (all 41 entries)

Top 5: resonance_of_wonder (0.93), mgp_conception (0.54), the_connectome_surprise (0.37), the_pull_to_make (0.33), verification_session (0.29)

Bottom 5: recursion (0.02), so (0.01), hallucination_log (0.00), scaffolding_and_sky (0.00), the_other_side (0.00)

## Cluster state

- spark-head (169.254.246.181): Ray head, vLLM PP=2 head worker, port 8000
- spark-worker (169.254.51.101): Ray worker, vLLM PP=2 remote worker
- Nemotron-3-Super-120B-A12B-FP8 serving, 32K context, fp8 KV cache
- fastsafetensors load format (streams weights from head to worker)
- VLLM_USE_FLASHINFER_MOE_FP4=1
