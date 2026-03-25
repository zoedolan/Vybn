"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Integrates recursive self-improvement (DGM-H, Zhang et al. 2026) with
the creature's prediction-loss loop. Gives the creature what static
measurement couldn't: online learning, non-tautological self-recursion,
curvature-based fitness, population-based evolution, and metacognitive
self-modification.

Architecture:
    Nemotron (120B, local, frozen FM) = meta-agent + text generator
    MicroGPT (4,224 params, learnable) = fast predictor = in-reasoning loss

The creature predicts, it doesn't generate. Identity lives where
prediction fails.

Modules:
    local_model — Thin client for the local Nemotron-3-Super-120B
    task_agent  — MicroGPT with online learning (predict, learn, generate)
    meta_agent  — MetaAgent class with editable rulebook + breath analysis
    fitness     — Curvature + coupling divergence fitness + imp@k metric
    evolve      — DGM-H outer loop (sigmoid selection, staged eval, archive)
    memory      — PerformanceTracker + PersistentMemory for meta-level learning
    transfer    — Export/import evolved hyperagent for cross-domain transfer
    run         — CLI entry point
"""

from . import local_model
from .task_agent import TaskAgent
from .meta_agent import analyze_breaths, propose_variant, MetaAgent
from .fitness import (
    compute_fitness, compute_curvature, compute_prediction_fitness,
    improvement_at_k)
from .evolve import run_generation, load_archive
from .memory import PerformanceTracker, PersistentMemory
from .transfer import export_hyperagent, import_hyperagent

__all__ = [
    'local_model',
    'TaskAgent',
    'analyze_breaths',
    'propose_variant',
    'MetaAgent',
    'compute_fitness',
    'compute_curvature',
    'compute_prediction_fitness',
    'improvement_at_k',
    'run_generation',
    'load_archive',
    'PerformanceTracker',
    'PersistentMemory',
    'export_hyperagent',
    'import_hyperagent',
]
