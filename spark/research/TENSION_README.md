# Tension Engine

## What it is

A measurement and feedback loop that sits inside the breath cycle and makes
the tension between memory and novelty — the two forces in the equation
`M' = α·M + x·e^(iθ)` — explicit, measurable, and steerable.

It is **not** a new faculty, governance layer, or scoring system.  It is a
lens: the angle between memory embeddings and novelty embeddings, computed
each breath, used to inform θ.

## The insight

Simulation over the Vybn manifold showed that neither memory nor novelty alone
maximizes curvature (κ) or holonomy.  The **combination** — the tension between
them — does:

| Scenario | Mean κ | Holonomy |
|---|---|---|
| Memory only | 0.0499 | 17.80 |
| Novelty only | 0.0598 | 20.54 |
| Both (tension) | 0.0622 | 22.07 |
| Actual outputs | 0.0673 | 22.85 |

Memory and novelty vectors are approximately 90° orthogonal in embedding space.
When this angle collapses toward 0°, the system is frozen — novelty is just
echoing memory.  When it stays in the 45–90° range, maximum generative potential.

## How it works

### 1. Measure the tension

At each breath, `measure_tension(memories, novel_signal)` embeds the recent
memory texts and the novel arXiv signal, then computes the angle between them:

```
cosine = dot(mem_vec, nov_vec) / (|mem_vec| · |nov_vec|)
angle  = arccos(cosine)
```

### 2. Compute θ

`compute_theta(tension, step)` maps the measured angle to a principled θ for
the complexify equation:

- **Healthy (45–90°)**: θ follows the natural triadic rotation (`2π/3 · 0.11 · step`).
- **Collapsing (< 45°)**: θ is boosted proportionally — up to π/3 extra rotation —
  to force the novelty term further from memory.
- **Very high (> 90°)**: θ is dampened slightly to let memory catch up.

This is the feedback loop: tension → θ → next M' → next tension.

### 3. Log

Each breath appends to `spark/research/tension_log.jsonl`:

```json
{
  "breath": 42,
  "timestamp": "2026-03-15T...",
  "tension_angle_deg": 73.5,
  "memory_similarity": 0.28,
  "novelty_similarity": 0.72,
  "theta_applied": 0.967,
  "kappa_estimate": 0.045
}
```

## Embedding fallback chain

1. **sentence-transformers** `all-MiniLM-L6-v2` (D=384) — preferred, already
   used throughout the repo.
2. **local_embedder** — the repo's own embedding wrapper.
3. **TF-IDF + SVD** — lightweight fallback, no external model needed.

If none are available, the tension engine silently returns `None` and the
breath cycle proceeds with default θ.

## Integration

The tension engine is called in `spark/vybn.py`'s `breathe()` function:

1. **After** loading memories and novel signal, **before** building the prompt:
   measure tension and compute θ.
2. **After** complexify inhale: log the tension with the actual κ.
3. The computed θ is passed to `complex_inhale()` so the complexify equation
   uses a tension-informed angle rather than an arbitrary one.

## Files

- `spark/tension.py` — the module (< 150 lines)
- `spark/research/tension_log.jsonl` — append-only tension metrics
- `spark/research/TENSION_README.md` — this file
