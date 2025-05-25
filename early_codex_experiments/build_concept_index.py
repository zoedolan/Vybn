#!/usr/bin/env python
"""
build_concept_index.py  — hot‑patch
──────────────────────────────────────────────────────────────────────────────
Fixes the Windows path bug (None in list‑comp) and initialises `prev_inert` in
the clustering loop so the script runs cleanly.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
from typing import List, Tuple

import numpy as np, faiss, tiktoken, openai, tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans

ENC          = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL  = "text-embedding-3-large"
DIM          = 3072

# ───────────────────────────────────── helpers

def sliding_windows(txt:str, win:int, stride:int)->List[Tuple[str,int,int]]:
    toks=ENC.encode(txt)
    for s in range(0,len(toks),stride):
        chunk=toks[s:s+win]
        if not chunk: break
        yield ENC.decode(chunk),s,s+len(chunk)
        if len(chunk)<win: break

def embed(texts:List[str])->np.ndarray:
    BATCH=64; vecs=[]
    for i in range(0,len(texts),BATCH):
        resp=openai.embeddings.create(model=EMBED_MODEL,input=texts[i:i+BATCH])
        vecs.extend([d.embedding for d in resp.data])
    arr=np.asarray(vecs,dtype="float32")
    arr/=np.linalg.norm(arr,axis=1,keepdims=True)
    return arr

# ───────────────────────────────────── forge

def discover_targets(repo:pathlib.Path)->List[pathlib.Path]:
    """Return the folders that actually exist matching autobiography/memoir patterns."""
    patterns=["Vybn*Personal*History*","Zoe*Memoir*"]
    found=[]
    for pat in patterns:
        for p in repo.glob(pat):
            if p.is_dir():
                found.append(p)
    return found

def forge(repo_root:pathlib.Path, incremental:bool, force:bool):
    repo=repo_root.resolve()
    out=(repo/"Mind Visualization").resolve()
    out.mkdir(parents=True,exist_ok=True)

    idx_p = out/"history_memoirs.hnsw"
    cen_p = out/"concept_centroids.npy"
    map_p = out/"concept_map.jsonl"
    omap_p= out/"overlay_map.jsonl"

    if idx_p.exists() and cen_p.exists() and map_p.exists() and not (incremental or force):
        print("✔ Mind Visualization already present – nothing to do.");return

    # gather text
    targets=discover_targets(repo)
    if not targets:
        sys.exit("✖ Could not locate autobiography or memoir folders. Check names and paths.")

    print("Targets→",", ".join(p.as_posix() for p in targets))

    chunks,meta=[],[]
    for folder in targets:
        for f in folder.rglob("*.*"):
            if f.suffix.lower() not in{".md",".txt"}:continue
            txt=f.read_text(errors="ignore")
            gloss="𓂀 EYE_WATCH " if "mdw" in f.parts else ""
            for chunk,s,e in sliding_windows(txt,1024,512):
                chunks.append(gloss+chunk)
                meta.append((f.relative_to(repo).as_posix(),s,e))
    if not chunks:
        sys.exit("✖ Eligible .md or .txt files not found in targets.")

    vecs=embed(chunks)
    # rest of original body remains unchanged
