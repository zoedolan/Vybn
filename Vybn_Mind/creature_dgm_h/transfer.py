"""
transfer.py — Backward-compatibility shim.

Transfer logic now lives in evolve.py (transfer IS evolution across
domain boundaries). This module re-exports for existing imports.
"""

from .evolve import (
    export_hyperagent,
    import_hyperagent,
    select_transfer_agent,
)
