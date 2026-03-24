# Experiment D Complete — March 22, 2026 ~02:45 UTC

## Result: Geometric regularization improves generalization

Clean A/B comparison on char-level GPT-2 (6 layers, 384 dim, ~10M params) 
trained on Shakespeare for 3000 steps.

### Key findings:

1. **Best val loss**: Baseline 1.5479 → Geometric 1.5295 (−1.19%)
2. **Geometric run wins at every single evaluation checkpoint** after step 0
3. **Val improvement grows with training**: from +0.03 at step 250 to +0.12 at step 2500
4. **Deepest layer angular behavior diverges**: 
   - Baseline L5: EXPANDS (1.315 → 1.365) — memorization pulls representations apart
   - Geometric L5: CONTRACTS (0.849 → 0.714) — regularizer holds coherence
5. **Geometric loss is stable**: ~0.0035 throughout, declining to ~0.0028 by end
6. **Training speed**: 1.6s/step geometric vs 1.4s/step baseline (~14% overhead)

### Geometry comparison at final step:

| Layer | Baseline angle | Geometric angle | Δ (radians) |
|-------|---------------|-----------------|-------------|
| L0 | 0.915 | 0.550 | −0.365 |
| L1 | 0.938 | 0.477 | −0.460 |
| L2 | 1.018 | 0.503 | −0.515 |
| L3 | 1.097 | 0.546 | −0.552 |
| L4 | 1.182 | 0.595 | −0.587 |
| L5 | 1.337 | 0.714 | −0.623 |

Mean angular reduction: 0.517 radians. The geometric regularizer creates 
fundamentally different internal representations — more aligned, more coherent, 
less scattered.

### What this means:

The holonomic theory predicted that constraining representational geometry 
would improve generalization by preventing the kind of angular scatter that 
accompanies memorization. Experiment D confirms this on a small model. 

The MIXED_RESTRUCTURING_AND_COMPRESSION verdict is honest — the geometric 
penalty changes both angles and norms, so the effect isn't purely angular. 
Disentangling these would require an experiment with separate angle-only 
and norm-only penalties. But the generalization improvement is real and 
monotonic across training.

### Next questions:

- Does the effect scale with model size? (More layers = more to regularize)
- What's the optimal λ? (Only tested 0.5; sweep needed)
- Can the angular contraction pattern predict generalization before val loss diverges?
- Does this connect to the LoRA fine-tuning path — can geometric constraints 
  during adapter training improve the Nemotron fine-tune?

Results: `/Vybn_Mind/experiments/holonomic_nemotron/results/experiment_D_result.json`
