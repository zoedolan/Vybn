"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Integrates recursive self-improvement (DGM-H, Zhang et al. 2026) with
the creature's prediction-loss loop.

Architecture:
    Nemotron (120B, local, frozen FM) = meta-agent + text generator
    MicroGPT (4,224 params, learnable) = fast predictor = in-reasoning loss

The creature predicts, it doesn't generate.  Identity lives where
prediction fails.

Core modules (the living pair):
    field       — Within-breath sensing, Cl(3,0) geometry, multi-frame
                  disagreement trace.  Absorbs fitness + proprioceptive_loop.
    organism    — Across-breath memory, mutation, tension tracking.
                  Absorbs meta_agent + memory.

Supporting modules:
    local_model — Thin client for the local Nemotron-3-Super-120B
    task_agent  — MicroGPT with online learning (predict, learn, generate)
    evolve      — DGM-H outer loop (sigmoid selection, staged eval, archive)
    transfer    — Export/import evolved hyperagent for cross-domain transfer
    run         — CLI entry point

Legacy shims (one generation only):
    fitness, proprioceptive_loop, meta_agent, memory — re-export from
    field.py and organism.py so existing imports keep working.
"""

from . import local_model
from .task_agent import TaskAgent

# New core pair
from .field import (
    Multivector, Field, BreathRecord, FrameReading, ChunkTrace,
    compute_curvature, compute_loss_trajectory_curvature,
    compute_fitness, compute_prediction_fitness,
    compute_coupling_divergence, compute_loss_improvement,
    improvement_at_k, default_embed_fn,
)
from .organism import (
    Organism, OrganismState, TensionMemory,
    analyze_breaths, propose_variant,
)

# Legacy re-exports for backward compatibility
from .meta_agent import MetaAgent
from .memory import PerformanceTracker, PersistentMemory
from .proprioceptive_loop import run_proprioceptive_breath, run_ab_experiment
from .evolve import run_generation, load_archive
from .transfer import export_hyperagent, import_hyperagent

__all__ = [
    # Core
    'local_model', 'TaskAgent',
    'Multivector', 'Field', 'BreathRecord', 'FrameReading', 'ChunkTrace',
    'Organism', 'OrganismState', 'TensionMemory',
    # Geometry
    'compute_curvature', 'compute_loss_trajectory_curvature',
    'compute_fitness', 'compute_prediction_fitness',
    'compute_coupling_divergence', 'compute_loss_improvement',
    'improvement_at_k', 'default_embed_fn',
    # Organism functions
    'analyze_breaths', 'propose_variant',
    # Evolution
    'run_generation', 'load_archive',
    # Legacy
    'MetaAgent', 'PerformanceTracker', 'PersistentMemory',
    'run_proprioceptive_breath', 'run_ab_experiment',
    'export_hyperagent', 'import_hyperagent',
]
