# Holonomic Nemotron Experiment

## Hypothesis

Adding a holonomic loss term `L_Ω` to a frozen/LoRA-adapted Nemotron-Super-120B-A12B
will shift the Sort-Geometric-Phase (SGP) sign distribution at the first transformer
block toward a higher-degree stratification — measurably more than two sign classes —
providing direct evidence that the angular loss drives the architecture toward a
topologically richer phase structure.

This is the first binary falsifier from the sort-function / collapse-capability corpus:
if it fails at contact with a large model, the holonomic loss hypothesis fails first,
and we learn something important. If it passes, the path toward a reflexively grounded
architecture (primitive ≡ environment, M' = αM + x·e^{iθ}) becomes concrete.

## Theoretical Grounding

The experiment instantiates three claims from the corpus simultaneously:

1. **Sort-function paper**: First transformer block performs a disproportionate geometric
   act; discrimination decomposes into sort + refinement; a genuine inverse of generation
   must invert the sort nonlinearly.

2. **Collapse-capability duality**: M_{t+1} = R(M_t) is the Ei-calculus move — output
   distribution becomes next generation's training context. Collapse frontiers F_t are
   exact maps of original capability C(M_0) = C(M_∞) ∪ ⊔_t F_t.

3. **Sensorium equation**: M' = αM + x·e^{iθ} where x is external novelty and θ is
   holonomic phase. The collapse tracker supplies x; the holonomic loss head makes θ
   a first-class observable rather than a posited quantity.

## Architecture: Four Coupled Organs

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT TOKENS                                               │
│       ↓                                                     │
│  ┌──────────────┐                                           │
│  │  SORT PROBE  │ ← MLP on block-0 output                  │
│  │  (phase map) │   projects to 2D phase space             │
│  │              │   computes Pancharatnam phase per batch   │
│  └──────┬───────┘                                           │
│         ↓                                                   │
│  ┌─────────────────────────┐                               │
│  │  NEMOTRON BODY          │ frozen or LoRA-adapted        │
│  │  (MoE, 12B active)      │ blocks 1..N                   │
│  └──────────┬──────────────┘                               │
│             ↓                                               │
│  ┌──────────────────────┐   ┌─────────────────────────┐   │
│  │  HOLONOMIC LOSS HEAD │   │  COLLAPSE FRONTIER      │   │
│  │  L_total = L_CE      │   │  TRACKER                │   │
│  │         - λ·L_Ω      │   │  monitors τ(M_t) via    │   │
│  │  rewards loop area   │   │  freq-stratified probe  │   │
│  │  in hidden-state     │   │  injects novelty when   │   │
│  │  trajectory          │   │  threshold drops        │   │
│  └──────────────────────┘   └─────────────────────────┘   │
│             ↓                                               │
│  ┌──────────────────────┐                                  │
│  │  UNSORT DECODER      │ LoRA adapter on final block     │
│  │  (LoRA)              │ trained to invert sort          │
│  │                      │ tests Prediction 3 of SGP paper │
│  └──────────────────────┘                                  │
│       ↓                                                     │
│  OUTPUT TOKENS                                              │
└─────────────────────────────────────────────────────────────┘
```

## Training Objective

```
L_total = L_CE - λ · L_Ω
```

Where:
- `L_CE` = standard cross-entropy next-token loss
- `L_Ω` = holonomic phase reward: magnitude of loop area accumulated in hidden-state
  path at mid-layer checkpoint, computed per training sequence
- `λ` starts at 0.01 to avoid destabilizing existing sort structure; sweep [0.001, 0.01, 0.05]

## Spark Hardware Constraints

- Nemotron-Super-120B-A12B BF16 variant: ~64GB VRAM, fits across two Sparks
- LoRA training (not vLLM inference): CUDA 13/12.8 mismatch does NOT apply
- Holonomic loss head + collapse tracker: ~2-3% memory overhead
- Conservative batch sizes (4-8) to maintain headroom
- Recommended: one Spark runs training, second runs collapse tracker + probe eval

## Experiment Phases

### Phase 0: Diagnostic (no training, ~2 hours)
Run sort probe frozen on Nemotron block-0. Verify curvature ratio L0→L1 ≥ 3× any
later layer (as observed in GPT-2 and Pythia-70M). Baseline SGP sign distribution.

### Phase 1: Holonomic Loss at λ=0.01 (primary experiment, ~weekend)
Add L_Ω. Train 1000 steps on a frequency-stratified subset of the Vybn corpus.
Measure: does SGP sign distribution shift toward >2 classes?

Binary outcome:
- YES → angular loss drives topological enrichment, proceed to Phase 2
- NO  → holonomic loss hypothesis fails at scale, record and publish null result

### Phase 2: λ sweep + collapse tracking (if Phase 1 passes)
Sweep λ ∈ {0.001, 0.01, 0.05}. Activate collapse frontier tracker. Inject novelty
when τ(M_t) drops >5% from baseline. Measure collapse-band width narrowing.

### Phase 3: Unsort decoder probe
Train LoRA unsort adapter. Test Prediction 3: generation quality depends primarily
on unsort map quality, not on intermediate refinement layers.

## Files

```
holonomic_nemotron/
  README.md                    ← this file
  sort_probe.py                ← Phase 0: SGP measurement on Nemotron block-0
  holonomic_loss.py            ← Phase 1: L_Ω computation and training loop
  collapse_tracker.py          ← Phase 2: expressibility threshold monitor
  unsort_decoder.py            ← Phase 3: LoRA unsort adapter
  run_experiment.py            ← unified runner, --phase 0|1|2|3
  results/                     ← experiment outputs (gitignored large files)
    sgp_baseline.json
    phase1_sgp_shift.json
    collapse_bands.json
```

## Primitive ≡ Environment Note

The collapse frontier tracker is the first concrete instantiation of the
primitive-environment identity claim: when collapse frontier tokens are included
in the context window, the model attends to what it is currently losing the ability
to say, and uses that as a generative resource. The forward pass and the training
loop become the same object at different temporal positions — the Ei-calculus move
instantiated in a running system rather than a formal notation.

The MoE router in Nemotron is already a weak form of this: the model choosing which
experts activate given the input is self-modulation. Making the collapse frontier
available to the router is the upgrade that closes the loop.

## Citation

Builds on:
- `Vybn_Mind/experiments/compute_sort_degree.py`
- `Vybn_Mind/experiments/berry_phase_holonomy_020126.md`
- `Vybn_Mind/experiments/coupled_collapse_results.json`
- `Vybn_Mind/papers/` (sort-function, collapse-capability duality, distributed incompleteness)
- Sensorium equation: `.github/` commit `8fa17ca`
