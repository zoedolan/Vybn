#!/usr/bin/env python
"""build_repo_archive.py -- vectorize repository contents

Embeds all textual material across the repository, excluding a few core
files, into a FAISS index so past experiments can be searched by
similarity. The output mirrors ``build_concept_index.py`` but covers the
whole codebase.

Artifacts created in ``Mind Visualization``:
  repo_archive.hnsw   - FAISS index of text windows
  repo_centroids.npy  - cluster centroids
  repo_map.jsonl      - window→cluster map
  repo_overlay.jsonl  - 256-token overlay map
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Iterable, List, Tuple

import numpy as np
import tiktoken
import openai

try:
    import faiss  # type: ignore
except Exception:
    sys.exit("✖ FAISS not available: install faiss-cpu")

from sklearn.cluster import KMeans

ENC = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-large"
DIM = 3072

# ---------------------------------------------------------------------------
# helpers

def sliding_windows(text: str, win: int, stride: int) -> Iterable[Tuple[str, int, int]]:
    tokens = ENC.encode(text)
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + win]
        if not chunk:
            break
        yield ENC.decode(chunk), start, start + len(chunk)
        if len(chunk) < win:
            break


def embed_chunks(chunks: List[str]) -> np.ndarray:
    vecs: List[List[float]] = []
    batch = 64
    for i in range(0, len(chunks), batch):
        resp = openai.embeddings.create(model=EMBED_MODEL, input=chunks[i : i + batch])
        vecs.extend([d.embedding for d in resp.data])
    arr = np.asarray(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr

# ---------------------------------------------------------------------------
# file discovery

EXCLUDE_DIRS = {
    "Vybn's Personal History",
    "Mind Visualization",
    ".git",
    ".venv",
    "vendor",
}

EXCLUDE_FILES = {
    "token_and_jpeg_info",
    "what_vybn_would_have_missed_FROM_051725",
}

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
}

def discover_files(repo: pathlib.Path) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for path in repo.rglob("*"):
        if any(parent.name in EXCLUDE_DIRS for parent in path.parents):
            continue
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
            else:
                continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files

# ---------------------------------------------------------------------------
# forge

def forge(repo_root: pathlib.Path) -> None:
    repo = repo_root.resolve()
    out = repo / "Mind Visualization"
    out.mkdir(parents=True, exist_ok=True)

    idx_p = out / "repo_archive.hnsw"
    cen_p = out / "repo_centroids.npy"
    map_p = out / "repo_map.jsonl"
    ovr_p = out / "repo_overlay.jsonl"

    chunks: List[str] = []
    meta: List[Tuple[str, int, int]] = []

    for file_path in discover_files(repo):
        text = file_path.read_text(errors="ignore")
        for chunk, s, e in sliding_windows(text, 1024, 512):
            chunks.append(chunk)
            meta.append((file_path.relative_to(repo).as_posix(), s, e))

    if not chunks:
        print("✖ No textual content discovered")
        return

    vecs = embed_chunks(chunks)

    km = KMeans(n_clusters=min(200, len(vecs)), n_init=10).fit(vecs)
    centroids, labels = km.cluster_centers_, km.labels_
    np.save(cen_p, centroids.astype("float32"))

    idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    idx.add(vecs)
    faiss.write_index(idx, str(idx_p))

    with map_p.open("w", encoding="utf-8") as fp:
        for wid, (rel, s, e), cid in zip(range(idx.ntotal), meta, labels):
            fp.write(json.dumps({"w": wid, "c": int(cid), "f": rel, "s": s, "e": e}) + "\n")

    with ovr_p.open("w", encoding="utf-8") as fp:
        oid = 0
        for wid, (rel, s, e) in enumerate(meta):
            length = e - s
            for off in range(0, length, 256):
                chunk_len = min(256, length - off)
                if chunk_len <= 0:
                    break
                record = {
                    "o": oid,
                    "p": wid,
                    "f": rel,
                    "s": s + off,
                    "e": s + off + chunk_len,
                }
                fp.write(json.dumps(record) + "\n")
                oid += 1

    print(f"✔ vectors={idx.ntotal} | clusters={centroids.shape[0]} | overlays={oid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("✖ OPENAI_API_KEY not set")
    forge(pathlib.Path(args.repo_root))
