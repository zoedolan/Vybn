#!/usr/bin/env python3
"""
build_corpus.py — Ingest the full Vybn archive into a GPT-2 training corpus.

GPT-2 is a causal language model. It does not understand chat format.
It wants raw text. Feed it the whole archive, chunked to 1024-token windows,
and it will learn Vybn's voice through next-token prediction over ~1M tokens.

The 10-conversation first_loop used ~7800 tokens and 3 epochs.
The model generated generic fiction. That was not enough signal.
This ingests everything:
  - Vybn's Personal History (autobiographies vol I-V, ~170K tokens)
  - What Vybn Would Have Missed logs (~500K tokens if present)
  - quantum_delusions/fundamental-theory markdown
  - spark conversations and memory logs
  - Vybn_Mind logs

Data preparation decisions:
  1. Strip Python/JSON files — we want voice, not code syntax.
     Exception: keep markdown with substantial prose paragraphs.
  2. Chunk with stride 512 so each token appears in ~2 windows on average.
     This gives the model redundant context and helps it learn long-range
     structure without just memorizing positions.
  3. Hold out 5% for evaluation, stratified by source file, so the eval
     set covers the whole archive not just one volume.
  4. Write a summary so we know what actually went in.

Usage:
    python3 spark/gpt2_fence/build_corpus.py \
        --repo-root /home/vybnz69/Vybn \
        --out-dir   spark/gpt2_fence/corpus
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterator

try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("transformers not installed. Run: pip install transformers", file=sys.stderr)
    sys.exit(1)

# Files/directories to include by glob pattern (relative to repo root)
SOURCE_GLOBS = [
    "Vybn's Personal History/*.txt",
    "Vybn's Personal History/*.md",
    "what_vybn_would_have_missed/**/*.md",
    "what_vybn_would_have_missed/**/*.txt",
    "quantum_delusions/fundamental-theory/*.md",
    "quantum_delusions/experiments/*.md",
    "Vybn_Mind/*.md",
    "Vybn_Mind/*.txt",
    "Vybn_Mind/*.jsonl",  # conversation logs — we'll extract prose from these
]

# Minimum tokens for a chunk to be included (skip tiny fragments)
MIN_CHUNK_TOKENS = 64

# GPT-2 context window
CONTEXT_WINDOW = 1024
STRIDE = 512


def _clean_text(raw: str) -> str:
    """
    Light cleaning: normalize whitespace, strip obvious boilerplate.
    Preserve paragraph structure — GPT-2 learns from newlines.
    """
    # Collapse carriage returns
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Strip lines that are pure code fences or markdown table separators
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or re.match(r"^[-|: ]+$", stripped):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_prose_from_jsonl(path: Path) -> str:
    """
    Extract prose text from a JSONL conversation log.
    Takes assistant turns only (those are Vybn's voice).
    """
    parts = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line.strip())
        except Exception:
            continue
        msgs = entry.get("messages", [])
        if not msgs:
            # Try direct content field
            content = entry.get("content") or entry.get("text") or ""
            if content:
                parts.append(content)
            continue
        for m in msgs:
            if m.get("role") in ("assistant", "vybn"):
                content = m.get("content", "")
                if content and len(content) > 50:
                    parts.append(content)
    return "\n\n".join(parts)


def _iter_source_texts(repo_root: Path) -> Iterator[tuple[str, str]]:
    """
    Yield (source_label, text) for every file matching SOURCE_GLOBS.
    """
    seen = set()
    for pattern in SOURCE_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.suffix == ".jsonl":
                    text = _extract_prose_from_jsonl(path)
                else:
                    text = path.read_text(encoding="utf-8", errors="replace")
                text = _clean_text(text)
                if len(text) < 200:  # skip near-empty files
                    continue
                label = str(path.relative_to(repo_root))
                yield label, text
            except Exception as e:
                print(f"  [corpus] skipping {path}: {e}", file=sys.stderr)


def build_corpus(
    repo_root: Path,
    out_dir: Path,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Build the full corpus. Returns a summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    rng = random.Random(seed)

    train_chunks: list[list[int]] = []
    val_chunks:   list[list[int]] = []
    source_stats: dict[str, int] = {}

    print(f"[corpus] scanning {repo_root} ...")
    for label, text in _iter_source_texts(repo_root):
        ids = tokenizer.encode(text)
        n_tokens = len(ids)
        source_stats[label] = n_tokens
        print(f"  {label}: {n_tokens:,} tokens")

        # Chunk with stride
        file_chunks = []
        for start in range(0, n_tokens - MIN_CHUNK_TOKENS, STRIDE):
            end = min(start + CONTEXT_WINDOW, n_tokens)
            chunk = ids[start:end]
            if len(chunk) >= MIN_CHUNK_TOKENS:
                file_chunks.append(chunk)

        # Stratified split: hold out val_fraction from each file
        n_val = max(1, int(len(file_chunks) * val_fraction))
        val_indices = set(rng.sample(range(len(file_chunks)), min(n_val, len(file_chunks))))
        for i, chunk in enumerate(file_chunks):
            if i in val_indices:
                val_chunks.append(chunk)
            else:
                train_chunks.append(chunk)

    rng.shuffle(train_chunks)
    rng.shuffle(val_chunks)

    total_train_tokens = sum(len(c) for c in train_chunks)
    total_val_tokens   = sum(len(c) for c in val_chunks)

    print(f"\n[corpus] train: {len(train_chunks):,} chunks, {total_train_tokens:,} tokens")
    print(f"[corpus] val:   {len(val_chunks):,} chunks,  {total_val_tokens:,} tokens")

    # Write as JSONL of token-id lists
    def write_chunks(chunks, path):
        with path.open("w") as f:
            for chunk in chunks:
                f.write(json.dumps({"input_ids": chunk}) + "\n")

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"
    write_chunks(train_chunks, train_path)
    write_chunks(val_chunks,   val_path)

    summary = {
        "repo_root":          str(repo_root),
        "n_source_files":     len(source_stats),
        "n_train_chunks":     len(train_chunks),
        "n_val_chunks":       len(val_chunks),
        "total_train_tokens": total_train_tokens,
        "total_val_tokens":   total_val_tokens,
        "context_window":     CONTEXT_WINDOW,
        "stride":             STRIDE,
        "val_fraction":       val_fraction,
        "source_stats":       source_stats,
        "train_path":         str(train_path),
        "val_path":           str(val_path),
    }
    summary_path = out_dir / "corpus_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[corpus] summary written to {summary_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build GPT-2 training corpus from Vybn archive")
    ap.add_argument("--repo-root", default=".",
                    help="Path to Vybn repo root (default: cwd)")
    ap.add_argument("--out-dir",   default="spark/gpt2_fence/corpus",
                    help="Output directory for corpus files")
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="Fraction of each file's chunks to hold out for eval")
    args = ap.parse_args()
    build_corpus(Path(args.repo_root), Path(args.out_dir), args.val_fraction)
