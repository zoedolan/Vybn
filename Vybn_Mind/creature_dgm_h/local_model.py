"""
local_model.py — Backward-compatibility shim.

The FM client now lives in creature.py (the source of sensation is part
of the field). This module re-exports everything so existing imports
like `from . import local_model; local_model.is_available()` keep
working.
"""

from .creature import (
    LLAMA_URL, MODEL_NAME,
    fm_available as is_available,
    fm_complete as complete,
    fm_stream as stream_tokens,
)
