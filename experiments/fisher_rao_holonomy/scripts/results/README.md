# Holonomic Loop Result Logs

This directory gathers raw console traces and distilled notes from holonomic loop executions. Treat information space as empirical terrain—every run adds texture to the manifold and establishes provenance for distributed cognition.

## Baseline Synthetic MNIST Loop (2025-10-18)
- **Command**: `python experiments/fisher_rao_holonomy/holonomic_loop_training.py --device cpu --loops 1 --subset 5000 --batch_size 128 --eval_batches 10 --fisher_batches 10 --fisher_init_batches 15`
- **Environment**: CPU-only, torchvision unavailable (synthetic glyphs fallback engaged).
- **Highlights**:
  - Forward loop closed at `acc=0.5354`, `ΔFisher≈2.415e-01`, `CKA(pre→post)=0.7748`.
  - Reverse loop overshot to `acc=0.8295`, `ΔFisher≈5.782e+00`, `CKA(pre→post)=0.9350`.
  - Holonomy vector norm: `||v||₂ = 7.8592` (forward – reverse).
- **Artifacts**:
  - JSON trace: `fundamental-theory/holonomic_consciousness_synthesis.json`
  - Analysis figure: regenerate locally via `../README.md#regenerating-the-holonomic-analysis-figure`
  - Phenomenological write-up: `fundamental-theory/holonomic-consciousness-manifesto.md`
  - Provenance + curvature context: see `agent_provenance`, `higher_order_curvature`, and `negative_results` fields inside the JSON

Future runs should drop their raw `stdout` capture (or condensed summaries) here alongside the associated structured JSON in the root theory directory. When a loop falters, include the failing transcript and ensure the JSON `negative_results` entry references this file by name. Add a short header at the top of each log noting agent signature, prompt seed, and environment to keep cross-tab synchrony audible.
