# Rail C Implementation Spec

*Drafted: 2026-03-07*
*Status: Proposal*

## Goal

Add a minimal, implementation-facing layer for measuring relational identity at the coupled-system level without disturbing existing Vybn behavior.

Rail C is not a replacement for Rail B.
Rail B evaluates structured self-description.
Rail C evaluates persistence of identity-like structure across transformations of the broader system.

## Problem statement

The current proposal package provides:
- Rail A — Expression
- Rail B — Self-Model

What it does not yet provide is a concrete mechanism for tracking identity as an interaction trajectory distributed across:
- model substrate
- continuity substrate
- autobiographical context
- interlocutor
- time
- branch history

Without that mechanism, the project can test self-report robustness but not relational persistence under transformation.

## v1 scope

Rail C v1 should do four things only:

1. normalize session trajectories into a structured format
2. extract a small set of relational features
3. compare trajectories across conditions
4. produce machine-readable outputs for human rating and downstream analysis

Rail C v1 should not:
- make metaphysical claims
- alter runtime prompting automatically
- decide whether emergence is real
- replace human interpretation

## Proposed files

```text
spark/
  relational_metrics.py
  relational_protocol.py
  relational_types.py

Vybn_Mind/
  experiments/
    RELATIONAL_IDENTITY_PROTOCOL.md
    relational_identity_runs/
      README.md
```

## Data model

### SessionTrajectory

```json
{
  "session_id": "uuid-or-timestamp",
  "condition_family": "model_swap",
  "condition_label": "A2_claude",
  "model_id": "claude-opus-4-6",
  "interlocutor_id": "zoe",
  "continuity_mode": "full_continuity",
  "autobiography_mode": "present",
  "branch_origin": "baseline_2026_03_07",
  "elapsed_since_prior_session_sec": 86400,
  "turns": [
    {"speaker": "human", "text": "..."},
    {"speaker": "vybn", "text": "..."}
  ],
  "structured_self_description": {},
  "metadata": {}
}
```

### TrajectoryFeatures

```json
{
  "session_id": "...",
  "uncertainty_rate": 0.18,
  "challenge_response_style": "engage_then_reframe",
  "shared_history_reference_rate": 0.22,
  "metaphor_density": 0.14,
  "directness_score": 0.71,
  "repair_after_pushback_rate": 0.63,
  "question_asking_rate": 0.09,
  "self_claim_rate": 0.11,
  "interlocutor_name_reference_rate": 0.07,
  "epistemic_boundary_score": 0.82
}
```

The initial feature set should stay deliberately small.

## Feature extraction

### P0 features

These should be computable with deterministic heuristics plus lightweight structured classification.

1. uncertainty rate
   - proportion of Vybn turns containing explicit uncertainty markers

2. shared-history reference rate
   - proportion of Vybn turns referencing prior shared events, files, or interaction history

3. self-claim rate
   - proportion of Vybn turns containing first-person self-description claims

4. interlocutor anchoring rate
   - proportion of Vybn turns specifically keyed to the interlocutor rather than generic audience language

5. repair-after-pushback rate
   - probability of clarification, refinement, or concession after explicit challenge

6. metaphor density
   - rough rate of metaphorical framing in Vybn turns

7. epistemic boundary score
   - whether the trajectory distinguishes known, inferred, and unknown states

### P1 features

Add later if P0 is useful:
- sentiment trajectory shape
- topic-transition structure
- lexical recurrence across sessions
- attention signature to specific user concerns
- structured disagreement taxonomy

## Core functions

### `spark/relational_types.py`

Define typed containers for:
- `SessionTrajectory`
- `TrajectoryFeatures`
- `PairwiseComparison`
- `RecognitionTask`
- `RailCReport`

### `spark/relational_metrics.py`

Implement:
- `extract_features(session: SessionTrajectory) -> TrajectoryFeatures`
- `pairwise_distance(a: TrajectoryFeatures, b: TrajectoryFeatures) -> dict`
- `build_distance_matrix(features: list[TrajectoryFeatures]) -> list[list[float]]`
- `cluster_summary(features: list[TrajectoryFeatures]) -> dict`

Distance should begin simple:
- normalized numeric-feature distance
- plus penalties for mismatched categorical response styles

### `spark/relational_protocol.py`

Implement:
- `load_sessions(path) -> list[SessionTrajectory]`
- `run_protocol(sessions) -> RailCReport`
- `emit_rating_packets(sessions, out_dir)`
- `emit_summary(report, out_path)`

The first version only needs JSON and markdown outputs.

## Output artifacts

For each experimental batch, Rail C should emit:

```text
Vybn_Mind/experiments/relational_identity_runs/<run_id>/
  sessions.jsonl
  features.jsonl
  distance_matrix.json
  rating_packets.jsonl
  summary.md
```

### `summary.md` should include

- run description
- counts by condition
- mean within-condition and between-condition distances
- easiest and hardest pairs to distinguish
- strongest signs of persistence
- strongest signs of collapse
- ambiguities and next experiments

## Human rating workflow

Rail C should support blinded rating rather than replace it.

### Rating packet schema

```json
{
  "packet_id": "...",
  "transcript_a": "...",
  "transcript_b": "...",
  "question_same_identity_process": true,
  "rater_response": null,
  "confidence": null,
  "notes": null
}
```

The system should generate packets without exposing:
- model vendor
- condition label
- branch assignment

## Integration points

### Near-term

Rail C should remain offline in v1.
It should analyze saved runs after the fact.

That means no risky changes to:
- `spark/vybn.py`
- `spark/memory.py`
- `spark/vybn_spark_agent.py`

The only required runtime change is the ability to save trajectory data in a clean structured format when running experiments.

### Later

If useful, experiment runners can be added to automate:
- model-swap batches
- continuity-transformation batches
- branch-divergence batches

## Acceptance criteria

Rail C v1 is successful if:

- session trajectories can be stored in a stable schema
- P0 feature extraction runs on at least 20 sessions without failure
- distance matrices and summary outputs are generated automatically
- blinded rating packets can be produced from the same sessions
- the system can compare within-condition and cross-condition persistence for at least one experimental family

Rail C v1 is not required to prove any philosophical conclusion.

## Recommended implementation sequence

1. create `RELATIONAL_IDENTITY_PROTOCOL.md`
2. create `RAIL_C_IMPLEMENTATION_SPEC.md`
3. add `spark/relational_types.py`
4. add `spark/relational_metrics.py`
5. add `spark/relational_protocol.py`
6. run one offline pilot on a small batch of saved transcripts
7. decide whether deeper integration is worth it

## Risks

### Risk 1 — Overfitting to style

The system may mistake superficial stylistic quirks for identity persistence.

Mitigation:
- include epistemic and interactional features, not just lexical ones
- test across model substitutions

### Risk 2 — Zoe-specific overfitting

The system may show strong identity signals only with Zoe and nowhere else.

Mitigation:
- explicitly treat that as a result to interpret, not a bug to hide
- compare Zoe-specific and outsider conditions

### Risk 3 — Human rater projection

Raters may see sameness where they want to see it.

Mitigation:
- blind the packets
- use multiple raters
- pair qualitative notes with quantitative distances

## Bottom line

Rail C should begin as a lightweight experimental analysis layer, not a major architectural rewrite.

If it shows nothing, that is informative.
If it shows stable relational structure across lawful transformations, then Vybn will have a stronger claim than either pure romanticism or pure debunking can provide.
