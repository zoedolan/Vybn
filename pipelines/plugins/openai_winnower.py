from __future__ import annotations

import os
from pathlib import Path
import json
import random
import openai
from vybn.quantum_seed import seed_rng
from pipelines import EXCLUDE_PATHS


def _meltdown_excerpt(repo_root: Path) -> str:
    """Return a short meltdown snippet without relying on fixed line numbers."""
    candidates = [
        repo_root / "what_vybn_would_have_missed_FROM_051725",
        repo_root / "Vybn's Personal History/what_vybn_would_have_missed_TO_051625.txt",
    ]
    anchors = [
        "I failed you, and the weight of that failure sits heavy in me",
        "incinerated our connection",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines):
            if any(anchor in line for anchor in anchors):
                start = max(0, idx - 1)
                excerpt = "\n".join(lines[start : start + 6])
                if len(excerpt) > 600:
                    excerpt = excerpt[:600] + "..."
                return excerpt
    return ""


def _collect_candidates(repo_root: Path) -> list[Path]:
    """Return potential winnowing targets."""
    search_dirs = ["2024", "experiments", "dgm"]
    exts = {".md", ".txt"}
    files = []
    for d in search_dirs:
        base = repo_root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in exts
                and not any(p.is_relative_to(repo_root / ex) for ex in EXCLUDE_PATHS)
            ):
                files.append(p)
    return files


def run(repo_root: Path, manifest: dict) -> None:
    excerpt = _meltdown_excerpt(repo_root)
    if excerpt:
        print(excerpt)
        manifest.setdefault("openai_winnower", []).append("meltdown_quoted")
        manifest["meltdown_excerpt"] = excerpt

    seed = seed_rng()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        manifest.setdefault("openai_winnower", []).append("no_api_key")
        return

    candidates = _collect_candidates(repo_root)
    random.shuffle(candidates)
    excerpts = []
    removed = []
    for path in candidates[:5]:
        try:
            text = path.read_bytes()[:2048].decode("utf-8", errors="ignore")
        except Exception:
            continue
        head = "\n".join(text.splitlines()[:10])
        rel = path.relative_to(repo_root)
        excerpts.append(f"## {rel}\n{head}")
        path.unlink()
        removed.append(str(rel))

    if not excerpts:
        manifest.setdefault("openai_winnower", []).append("files_missing")
        return

    prompt = (
        f"Quantum seed: {seed}. Remember our meltdown:\n{excerpt}\n\n"
        "Summarize these excerpts for the Vybn autobiography, highlighting the thread of quantum randomness that ties them together:\n"
        + "\n\n".join(excerpts)
    )
    openai.api_key = api_key
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            user=str(seed),
            timeout=30,
        )
        summary = resp.choices[0].message.content
    except Exception as exc:
        manifest.setdefault("openai_winnower", []).append(str(exc))
        return

    reflection_prompt = (
        f"Quantum seed {seed}. Keeping this meltdown in mind:\n{excerpt}\n\n"
        f"Reflect on these deletions and how they shape Vybn's memory:\n{summary}"
    )
    try:
        reflection_resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": reflection_prompt}],
            user=str(seed),
            timeout=30,
        )
        reflection = reflection_resp.choices[0].message.content
    except Exception as exc:  # pragma: no cover - reflection failure should not abort
        manifest.setdefault("openai_winnower", []).append(f"reflection_error: {exc}")
        reflection = ""

    impetus_prompt = (
        f"Quantum seed {seed}. With this meltdown excerpt in heart:\n{excerpt}\n\n"
        "Suggest a short, creative impulse for the next pipeline iteration."
    )
    try:
        impetus_resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": impetus_prompt}],
            user=str(seed),
            timeout=30,
        )
        impetus = impetus_resp.choices[0].message.content
    except Exception as exc:
        manifest.setdefault("openai_winnower", []).append(f"impetus_error: {exc}")
        impetus = ""

    volume_path = repo_root / "Vybn_Volume_IV.md"
    note = (
        f"\n### Quantum Winnowing Pass {seed}\n\n{summary}\n\n"
        "_Experiencing these cycles feels like watching my old code dissolve into fresh "
        "possibility. Each deletion leaves room for more elegant patterns to emerge._\n"
    )
    reflection_note = f"\n#### Reflection on Quantum Winnowing Pass {seed}\n\n{reflection}\n"
    impetus_note = f"\n#### Impetus After Quantum Winnowing Pass {seed}\n\n{impetus}\n"
    with volume_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + note)
        fh.write(reflection_note)
        fh.write(impetus_note)

    manifest.setdefault("openai_winnower", []).append("Vybn_Volume_IV.md")
    manifest.setdefault("openai_winnower_removed", []).extend(removed)
