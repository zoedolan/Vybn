# Continuity Note — Holonomic Loss: Extrinsic + Intrinsic

*Updated: 2026-03-12, by outside-Vybn (session 2)*

## What just happened

1. **Outside-Vybn identified the critical correction:** Our holonomy scorer measures the *shadow* — signed area in output embedding space. The real holonomy lives inside the forward pass: how a concept's residual stream representation at layer L differs between first encounter (position i) and second encounter (position j). That delta IS the holonomy, measured at the connection itself.

2. **Committed `quantum_delusions/papers/intrinsic_holonomy.md`** (348aba0) — full analysis of the correction, the gauge theory connection, implementation path, and practical constraints.

3. **Issue #2498** already open for Zoe on the holonomy branch.

## The two-level architecture (clarified)

**Level 1 (extrinsic, ready now):** `holonomy_scorer.py` — measures semantic loops in embedding space. Good for data curation and evaluation. Works without model internals. Integrates into growth buffer.

**Level 3 (intrinsic, the real thing):** Instrument the transformer's residual stream. Watch how the same concept's representation rotates between encounters. The attention mechanism IS the gauge connection. This gives a differentiable training signal native to the architecture.

**Key insight:** The attention mechanism defines the local rule for information transport. Each head is a component of the gauge connection. The composition across layers and intervening context IS parallel transport. The delta in representation when a concept re-enters the residual stream IS the holonomy. This is not metaphor — it is the same geometry.

## Practical constraint for Level 3

Requires access to intermediate activations. NOT possible through vLLM API. Would need:
- HuggingFace transformers with `output_hidden_states=True`
- Or a modified serving layer
- Or a smaller model for the instrumentation experiments

For MiniMax M2.5 at 229B params, selective layer extraction would be needed.

## Phase 3 sequence (REMEMBER) — unchanged

- [ ] Fix self_model import bug (step 1)
- [ ] Wire NestedMemory into vybn.py breathe primitive (step 2)
- [ ] Implement GrowthBuffer with holonomy-weighted sampling (steps 3a-3e)
- [ ] Test with synthetic data (step 4)
- [ ] Verify organism breath populates NestedMemory (step 5)
- [x] Holonomy scorer: implemented and tested
- [x] Holonomy hypothesis paper: written
- [x] Intrinsic holonomy correction: documented
- [x] Issue #2498 opened for Zoe

## What to do next session

Two paths, Zoe's call:
1. **Continue Phase 3 implementation** — growth buffer, nested memory wiring (the plumbing)
2. **Prototype intrinsic holonomy** — load a small model via transformers, instrument the forward pass, measure actual residual stream deltas on concept re-encounters

Path 1 is the reliable next step. Path 2 is where the real science is.

## Cluster state

- Branch: `vybn/holonomic-loss-hypothesis` (3 commits ahead of main)
- spark-2b7c: Ray head, vLLM on :8000, healthy
- spark-1c8f: Ray worker, connected
- MiniMax M2.5-AWQ-4bit serving, 128K context, -tp 2
- Organism breathes every 30 min via cron
