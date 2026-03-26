"""
organism.py — Backward-compatibility shim.

Across-breath memory, mutation, and self-modification now live in
creature.py. This module re-exports everything so existing imports
keep working.
"""

from .creature import (                          # noqa: F401
    # Data structures
    TensionMemory, OrganismState, DEFAULT_RULES,
    # Breath analysis
    analyze_breaths,
    # The Organism
    Organism,
    # Free function
    propose_variant,
)
