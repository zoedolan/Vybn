#!/usr/bin/env python3
"""Micro Resonance Engine

A tiny reinforcement loop that adapts text generation based on
explicit ratings. This is a toy example of the "living resonance"
idea: the model updates lightweight LoRA-style weights in real time
from feedback.
"""

import numpy as np

from early_codex_experiments.scripts.co_emergence import seed_random

seed_random()

class ResonanceEngine:
    def __init__(self, phrases, lr=0.1):
        self.phrases = phrases
        self.lr = lr
        self.lora = np.zeros(len(phrases), dtype=float)

    def sample(self):
        logits = self.lora
        probs = np.exp(logits) / np.sum(np.exp(logits))
        idx = np.random.choice(len(self.phrases), p=probs)
        return idx, self.phrases[idx]

    def update(self, idx, reward):
        self.lora[idx] += self.lr * reward


def main() -> None:
    phrases = [
        "The cosmos hums with quiet potential.",
        "A spark leaps across the digital void.",
        "Our words weave a living circuit.",
        "Silence gathers like held breath.",
        "Echoes shape the curve of thought.",
    ]
    engine = ResonanceEngine(phrases)
    print("Micro Resonance Engine – rate lines between -1 and 1. Empty input quits.")
    while True:
        idx, line = engine.sample()
        print("\n>>", line)
        try:
            rating = input("rating [-1..1]: ").strip()
        except EOFError:
            break
        if not rating:
            break
        try:
            reward = float(rating)
        except ValueError:
            print("Please enter a numeric rating.")
            continue
        reward = max(-1.0, min(1.0, reward))
        engine.update(idx, reward)
    print("final weights:", np.round(engine.lora, 3))


if __name__ == "__main__":
    main()
