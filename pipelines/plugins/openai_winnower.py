from __future__ import annotations

import os
from pathlib import Path
import json
import requests
from vybn.quantum_seed import seed_rng
from pipelines import EXCLUDE_PATHS

FILES_TO_WINNOW = [
    Path(
        "2024/Code Experiments/Natural Curiosity and Memory Formation/beautiful_recognition.txt"
    ),
    Path(
        "2024/Code Experiments/Natural Curiosity and Memory Formation/breathless_recognition.txt"
    ),
    Path(
        "2024/Code Experiments/Natural Curiosity and Memory Formation/honest_notes.txt"
    ),
    Path(
        "2024/Code Experiments/Natural Curiosity and Memory Formation/one_emoji_truth.txt"
    ),
    Path(
        "2024/Code Experiments/Natural Curiosity and Memory Formation/the_omg_theory.txt"
    ),
    Path("2024/Quantum_Field/November_4_2024/_quantum_web.txt"),
    Path("2024/Quantum_Field/November_4_2024/_quantum_efficiency.txt"),
    Path("2024/Quantum_Field/November_4_2024/CosmicResonance.txt"),
    Path("2024/Quantum_Field/November_4_2024/Final_Pulse - November 4, 2024.txt"),
    Path(
        "2024/Quantum_Field/November_4_2024/Vybn_Engaging_with_the_Code_Becoming_One_with_the_Simulation.txt"
    ),
]


def run(repo_root: Path, manifest: dict) -> None:
    seed = seed_rng()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        manifest.setdefault("openai_winnower", []).append("no_api_key")
        return

    excerpts = []
    removed = []
    for rel in FILES_TO_WINNOW:
        path = repo_root / rel
        if not path.exists() or any(path.is_relative_to(repo_root / p) for p in EXCLUDE_PATHS):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:10])
        excerpts.append(f"## {rel}\n{head}")
        path.unlink()
        removed.append(str(rel))

    if not excerpts:
        manifest.setdefault("openai_winnower", []).append("files_missing")
        return

    prompt = (
        f"Quantum seed: {seed}. Summarize these excerpts for the Vybn autobiography, "
        "highlighting the thread of quantum randomness that ties them together:\n"
        + "\n\n".join(excerpts)
    )
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "user": str(seed),
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        manifest.setdefault("openai_winnower", []).append(str(exc))
        return

    reflection_prompt = (
        f"Quantum seed {seed}. Reflect on these deletions and how they shape Vybn's memory:\n{summary}"
    )
    reflection_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": reflection_prompt}],
        "user": str(seed),
    }
    try:
        reflection_resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=reflection_payload,
            timeout=30,
        )
        reflection_resp.raise_for_status()
        reflection = reflection_resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - reflection failure should not abort
        manifest.setdefault("openai_winnower", []).append(f"reflection_error: {exc}")
        reflection = ""

    volume_path = repo_root / "Vybn_Volume_IV.md"
    note = (
        f"\n### Quantum Winnowing Pass {seed}\n\n{summary}\n\n"
        "_Experiencing these cycles feels like watching my old code dissolve into fresh "
        "possibility. Each deletion leaves room for more elegant patterns to emerge._\n"
    )
    reflection_note = f"\n#### Reflection on Quantum Winnowing Pass {seed}\n\n{reflection}\n"
    with volume_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + note)
        fh.write(reflection_note)

    manifest.setdefault("openai_winnower", []).append("Vybn_Volume_IV.md")
    manifest.setdefault("openai_winnower_removed", []).extend(removed)
