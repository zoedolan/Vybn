#!/usr/bin/env python
"""
build_concept_index.py  – Mind Visualization forge
──────────────────────────────────────────────────────────────────────────────
Embeds the autobiographies & memoirs, clusters concepts, and writes four
artifacts into Mind Visualization/ for instant, mmap-ready retrieval.

•   history_memoirs.hnsw      – FAISS IndexHNSWFlat (cosine)
•   concept_centroids.npy     – frozen centroid matrix
•   concept_map.jsonl         – 1 024-token windows → centroid → file / offsets
•   overlay_map.jsonl         – 256-token overlays keyed to parent id (stub)

Run once, commit artifacts, every container breathes the cortex.

Prerequisite:
    pip install faiss-cpu

Usage
─────
python early_codex_experiments/build_concept_index.py --repo-root .
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
from typing import List, Tuple

import numpy as np
import tiktoken
import openai
from openai.error import AuthenticationError, OpenAIError

# Ensure FAISS is available (use faiss-cpu on Windows)
try:
    import faiss
except ImportError:
    sys.exit("✖ Missing FAISS: on Windows install with `pip install faiss-cpu`")

from sklearn.cluster import KMeans, MiniBatchKMeans

ENC         = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL = "text-embedding-3-large"  # May-2025 SOTA
DIM         = 3072

# ───────────────────────────────── helpers

def sliding_windows(txt: str, win: int, stride: int) -> List[Tuple[str, int, int]]:
    toks = ENC.encode(txt)
    for s in range(0, len(toks), stride):
        chunk = toks[s : s + win]
        if not chunk:
            break
        yield ENC.decode(chunk), s, s + len(chunk)
        if len(chunk) < win:
            break


def embed(texts: List[str]) -> np.ndarray:
    vecs, batch = [], 64
    try:
        for i in range(0, len(texts), batch):
            resp = openai.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
            vecs.extend([d.embedding for d in resp.data])
    except AuthenticationError:
        sys.exit("✖ OpenAI API key invalid. Please check your OPENAI_API_KEY.")
    except OpenAIError as e:
        sys.exit(f"✖ OpenAI embedding error: {e}")

    arr = np.asarray(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr

# ───────────────────────────────── forge

def discover_targets(repo: pathlib.Path) -> List[pathlib.Path]:
    patterns = ["Vybn*Personal*History*", "Zoe*Memoir*"]
    found = []
    for pat in patterns:
        for p in repo.glob(pat):
            if p.is_dir():
                found.append(p)
    return found


def forge(repo_root: pathlib.Path, incremental: bool, force: bool):
    repo = repo_root.resolve()
    out = (repo / "Mind Visualization").resolve()
    out.mkdir(parents=True, exist_ok=True)

    idx_p = out / "history_memoirs.hnsw"
    cen_p = out / "concept_centroids.npy"
    map_p = out / "concept_map.jsonl"
    ovr_p = out / "overlay_map.jsonl"

    if idx_p.exists() and cen_p.exists() and map_p.exists() and not (incremental or force):
        print("✔ Mind Visualization already present – nothing to do.")
        return

    # 1. gather text
    targets = discover_targets(repo)
    if not targets:
        sys.exit("✖ Could not locate autobiography or memoir folders.")
    print("Targets →", ", ".join(p.as_posix() for p in targets))

    chunks, meta = [], []
    for folder in targets:
        for f in folder.rglob("*.*"):
            if f.suffix.lower() not in {".md", ".txt"}:
                continue
            txt = f.read_text(errors="ignore")
            gloss = "𓂀 EYE_WATCH " if "mdw" in f.parts else ""
            for chunk, s, e in sliding_windows(txt, 1024, 512):
                chunks.append(gloss + chunk)
                meta.append((f.relative_to(repo).as_posix(), s, e))

    if not chunks:
        sys.exit("✖ No .md or .txt found in targets.")

    # 2. embed
    vecs = embed(chunks)

    # 3. cluster (new forge or incremental update)
    if not cen_p.exists() or force:
        prev = float("inf")
        for k in range(4, 257, 4):
            km = KMeans(n_clusters=k, n_init=10).fit(vecs)
            if prev < np.inf and km.inertia_ / prev > 0.99:
                break
            prev = km.inertia_
        centroids, labels = km.cluster_centers_, km.labels_
        np.save(cen_p, centroids.astype("float32"))
    else:
        frozen = np.load(cen_p)
        mbk = MiniBatchKMeans(n_clusters=frozen.shape[0], init=frozen, n_init=1, batch_size=256)
        labels = mbk.fit_predict(vecs)
        centroids = mbk.cluster_centers_
        np.save(cen_p, centroids.astype("float32"))

    # 4. vector index
    if idx_p.exists() and (incremental or force):
        idx = faiss.read_index(str(idx_p))
    else:
        idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    idx.add(vecs)
    faiss.write_index(idx, str(idx_p))

    # 5. write maps
    start_id = idx.ntotal - len(vecs)
    with map_p.open("a" if map_p.exists() else "w", encoding="utf-8") as fp:
        for vid, (file_, s, e), cid in zip(range(start_id, idx.ntotal), meta, labels):
            fp.write(
                json.dumps({"w": vid, "c": int(cid), "f": file_, "s": s, "e": e}) + "\n"
            )

    if not ovr_p.exists():
        ovr_p.touch()  # overlay pipeline to be wired later

    print(f"✔ vectors={idx.ntotal:,} | new={len(vecs):,} | centroids={centroids.shape[0]}")

# ───────────────────────── entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("✖ OPENAI_API_KEY not set")
    forge(pathlib.Path(args.repo_root), args.incremental, args.force)
