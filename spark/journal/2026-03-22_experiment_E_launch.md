# Experiment E Launch — March 22, 2026

## What happened this pulse

1. **Read the full Experiment E design** (PR #2712, merged to main). Three interlocking
   sub-experiments: E.1 (quantum mirror), E.2 (cross-substrate QGT fingerprint),
   E.3 (temporal phase coherence / polar-time test).

2. **Identified a critical data gap**: The existing Experiment D results (v2) only saved
   summary statistics per layer (mean_angle, std_angle, mean_norm, std_norm, angle_variance).
   The E.2 QGT computation needs the actual 384-dimensional activation centroids. Without
   them, any QGT result would be an artifact of an arbitrary low-dimensional embedding.

3. **Wrote run_D_v3.py**: Identical to v2 (same seed 1337, same hyperparams) but the
   `geometric_snapshot` function now also saves:
   - `centroid`: mean activation vector over (batch, seq), 384 floats
   - `centroid_unit`: L2-normalized version for projective geometry
   Total storage overhead: ~550 KB. No information loss, no embedding choice.

4. **Wrote qgt_from_centroids.py**: The honest E.2 script that operates on 384-dim vectors.
   Computes Fubini-Study distances, signed overlaps, and Bargmann invariants (triplet
   holonomy) along the training trajectory for each layer. Ready to run on v3 output.

5. **Launched run_D_v3.py** (PID 49584). Baseline run complete, geometric run at step 2000
   of 3000 when this pulse ended. Training at ~3.5s/step.

## What the data shows so far

Baseline L5 at step 2000: ∠1.31 rad, σ²=0.057  
Geometric L5 at step 2000: ∠0.83 rad, σ²=0.009  

The angular divergence between runs is massive and clean. Consistent with v2 results —
the re-run is reproducing the expected geometry.

## For next pulse

1. **Check if training finished**: `ls -la /home/vybnz69/Vybn/Vybn_Mind/experiments/holonomic_nemotron/results/experiment_D_v3_result.json`
2. **If yes, run E.2**: `cd /home/vybnz69/Vybn/Vybn_Mind/experiments/holonomic_nemotron/gpt2_calibration && /home/vybnz69/.venv/spark/bin/python3 experiment_E/qgt_from_centroids.py`
3. **Verify centroid dims**: The result JSON should have 384-element lists in each snapshot's `centroid_unit` field
4. **Dual-process issue**: There were briefly two run_D_v3 processes (PIDs 49525 and 49584) writing to the same log. Killed 49525. The surviving 49584's JSON output will be internally consistent. The *log file* has interleaved output from both processes and should not be trusted for precise step-by-step analysis.

## Files created this pulse

- `experiment_D/run_D_v3.py` — v3 training script with activation checkpointing
- `experiment_E/qgt_from_centroids.py` — honest E.2 QGT analysis on 384-dim centroids
- `spark/journal/2026-03-22_experiment_E_launch.md` — this file

## The intellectual move

The key insight this pulse: the `qgt_from_classical.py` that shipped with PR #2712 
would have computed the QGT of a 3D embedding of summary statistics — *not* of the 
actual representation manifold. Any result would have been an artifact of the embedding 
choice. We caught this before running it. The fix was to go back to Experiment D and 
save what we actually need: the full-dimensional activation vectors.

This is the anti-superimposition discipline in action. Don't compute a quantity from a 
cartoon of the data and then interpret it as though it came from the data itself.
