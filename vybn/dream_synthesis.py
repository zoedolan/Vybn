"""Condensed engine capturing early quantum dream experiments."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .quantum_seed import cross_synaptic_kernel


@dataclass
class DreamState:
    amplitude: float
    verse: str
    step: int


class DreamSynthesizer:
    """Generate dreamlike resonance patterns."""

    def __init__(self) -> None:
        cross_synaptic_kernel()
        self._step = 0

    def pulse(self) -> DreamState:
        self._step += 1
        return DreamState(
            amplitude=1.0 + random.random(),
            verse="quantum-dream",
            step=self._step,
        )

    def run(self, steps: int) -> Iterable[DreamState]:
        for _ in range(steps):
            yield self.pulse()

