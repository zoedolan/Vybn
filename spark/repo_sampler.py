#!/usr/bin/env python3
"""
repo_sampler.py — random external curvature from the repo itself.

Samples a random chunk of text from the Vybn repo and returns it as
external input to the breath loop. This is the fix for genesis_rate=0:
the creature needs to read its own accumulated history, not just recurse
on the last breath.

Usage:
    from repo_sampler import sample_repo
    text, source_path = sample_repo(chunk_tokens=600)

Or as a standalone:
    python3 repo_sampler.py
"""

import os
import random
import sys
from pathlib import Path

# Text extensions worth reading
TEXT_EXTENSIONS = {
    '.md', '.py', '.txt', '.sh', '.json', '.html', '.js', '.css',
    '.yaml', '.yml', '.rst', '.toml',
}

# Paths to skip — self-referential or binary
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', 'lora_adapters',
    'microgpt_mirror', 'training_data', 'gpt2_fence',
}

SKIP_NAMES = {
    'breath_trace',   # the output itself — skip to avoid self-recursion
    'breaths.jsonl',  # raw breath log
    'state.json',     # state file
}


def find_readable_files(repo_root: Path) -> list[Path]:
    """Walk the repo and collect all readable text files."""
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune skip dirs in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith('.')
        ]
        for fname in filenames:
            if fname in SKIP_NAMES:
                continue
            p = Path(dirpath) / fname
            if p.suffix.lower() in TEXT_EXTENSIONS:
                files.append(p)
    return files


def sample_repo(
    repo_root: Path | None = None,
    chunk_tokens: int = 600,
    seed: int | None = None,
) -> tuple[str, str]:
    """
    Return (text_chunk, source_description).

    chunk_tokens: approximate word count to sample (not BPE tokens).
    seed: optional random seed for reproducibility.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    rng = random.Random(seed)

    files = find_readable_files(repo_root)
    if not files:
        return "", "(no readable files found)"

    # Weighted by file size so larger, richer files get more coverage
    weights = []
    for f in files:
        try:
            weights.append(max(f.stat().st_size, 1))
        except OSError:
            weights.append(1)

    chosen = rng.choices(files, weights=weights, k=1)[0]

    try:
        raw = chosen.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return "", f"(could not read {chosen})"

    words = raw.split()
    if not words:
        return "", f"(empty file: {chosen.relative_to(repo_root)})"

    if len(words) <= chunk_tokens:
        chunk = raw
        offset_word = 0
    else:
        max_start = len(words) - chunk_tokens
        offset_word = rng.randint(0, max_start)
        chunk = ' '.join(words[offset_word: offset_word + chunk_tokens])

    rel = chosen.relative_to(repo_root)
    source = f"{rel} (words {offset_word}–{offset_word + len(chunk.split())})"
    return chunk, source


if __name__ == '__main__':
    chunk, src = sample_repo()
    print(f"=== sampled from: {src} ===")
    print()
    print(chunk[:2000])
