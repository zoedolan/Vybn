#!/usr/bin/env python3
"""Backward-compatibility shim for code that imports from memory.py.

This module was renamed to context_assembler.py during the
2026-03-08 consolidation. Its actual role is prompt/context assembly
for the model call — not memory governance (which is memory_fabric.py).

This shim re-exports the public symbols so nothing breaks.
It should be removed once all import sites are updated.
"""
from context_assembler import ContextAssembler, BootError, _check_soul  # noqa: F401

# Preserve the old name for callers that use it
MemoryAssembler = ContextAssembler
