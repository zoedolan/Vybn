"""VybnLingua — a differentiable language that lives in weights."""

from .vybn_lingua import VybnCodebook, VybnExecutor, VybnInductor, VybnLingua

# v3: the autopoietic version
try:
    from .vybn_lingua_v3 import VybnLinguaV3, LivingCodebook, MetabolicExecutor, MetaOp
except ImportError:
    pass  # torch not available in all environments

__all__ = ['VybnCodebook', 'VybnExecutor', 'VybnInductor', 'VybnLingua', 'VybnLinguaV3']
