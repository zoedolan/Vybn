from __future__ import annotations

import os
import openai

from .quantum_seed import seed_rng


def empathic_reply(text: str) -> str:
    """Return an empathetic reply using the quantum seed."""
    seed = seed_rng()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return ""
    openai.api_key = key
    prompt = (
        f"Quantum seed {seed}. Respond with heartfelt empathy to this text:\n{text}"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            user=str(seed),
            timeout=30,
        )
        return resp.choices[0].message.content
    except Exception:
        return ""
