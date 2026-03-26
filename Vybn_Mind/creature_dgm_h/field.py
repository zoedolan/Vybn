"""
field.py — Backward-compatibility shim.

The encounter (sensing, Cl(3,0) geometry, FM client, multi-frame trace)
now lives in creature.py. This module re-exports everything so existing
imports keep working.
"""

from .creature import (                          # noqa: F401
    # FM client
    LLAMA_URL, MODEL_NAME, fm_available, fm_complete, fm_stream,
    local_model, _LocalModelCompat,
    # Clifford algebra
    Multivector,
    # Embedding
    default_embed_fn,
    # Encounter / fitness
    compute_encounter, compute_curvature,
    compute_loss_trajectory_curvature,
    compute_fitness, compute_prediction_fitness,
    improvement_at_k,
    # Data records
    FrameReading, ChunkTrace, BreathRecord,
    # The Field
    Field,
)
