"""Shim. Real code lives in organism.py."""
from .organism import (  # noqa: F401
    Organism, OrganismState, TensionMemory, DEFAULT_RULES,
    analyze_breaths, propose_variant, evaluate_variant,
)

# MetaAgent is now Organism. This alias keeps old imports working.
MetaAgent = Organism
