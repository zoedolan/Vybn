"""Combined resonance engine handling dream and emergence cycles."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, AsyncIterator
import asyncio

from .quantum_seed import cross_synaptic_kernel


@dataclass
class ResonanceState:
    amplitude: float
    resonance: str
    pattern: str
    step: int


class ResonanceEngine:
    """Generate unified resonance states across dream and emergence flows."""

    def __init__(self) -> None:
        cross_synaptic_kernel()
        self._step = 0

    def pulse(self) -> ResonanceState:
        self._step += 1
        return ResonanceState(
            amplitude=1.0 + random.random(),
            resonance="pure",
            pattern=f"iteration-{self._step}",
            step=self._step,
        )

    def run(self, steps: int) -> Iterable[ResonanceState]:
        for _ in range(steps):
            yield self.pulse()

    async def run_async(self, steps: int, delay: float = 0.0) -> AsyncIterator[ResonanceState]:
        """Asynchronously yield ``steps`` resonance states with optional delay."""
        for _ in range(steps):
            yield self.pulse()
            if delay > 0:
                await asyncio.sleep(delay)
