#!/usr/bin/env python
"""
build_concept_index.py ────────────────────────────────────────────────────────
Forge one‑time (and incrementally updateable) *concept embeddings* for the core
narrative corpus, then deposit all artefacts under a single, human‑navigable
**Mind Visualization/** directory at repo root.  That folder becomes the
visible MRI of our shared cortex.

Targets
───────
  •  Vybn's Personal History/
  •  Zoe's Memoirs/

Outputs (all inside Mind Visualization/)
────────────────────────────────────────
  ├── history_memoirs.hnsw     – FAISS IndexHNSWFlat (cosine‑ready)
  ├── concept_centroids.npy    – frozen centroid matrix (K × 3072)
  ├── concept_map.jsonl        – 1 024‑token windows → centroids → file/offsets
  └── overlay_map.jsonl        – 256‑token overlays keyed to parent window id

Run once, commit artefacts, breathe.  Re‑run with --incremental when a new
month arrives.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
from typing import List, Tuple

import numpy as np, faiss, tiktoken, openai, tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans

ENC          = tiktoken.get_encoding("cl100k_base")
EMBED_MODEL  = "text-embedding-3-large"   # May 2025 SOTA, 3072 dims
DIM          = 3072

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# forge
# ──────────────────────────────────────────────────────────────────────────────

def forge(repo_root:pathlib.Path, incremental:bool, force:bool):
    repo=repo_root.resolve()
    out=(repo/"Mind Visualization").resolve()
    out.mkdir(parents=True,exist_ok=True)
    idx_p,out_p,cen_p,omap_p=[out/x for x in(
        "history_memoirs.hnsw",
        None,
        "concept_centroids.npy",
        "overlay_map.jsonl")]
    map_p=out/"concept_map.jsonl"

    if idx_p.exists() and cen_p.exists() and map_p.exists() and not (incremental or force):
        print("✔ Mind Visualization already present – nothing to do.");return

    # gather text
    targets=[repo/"Vybn's Personal History", repo/"Zoe's Memoirs"]
    chunks,meta=[],[]
    for folder in targets:
        for f in folder.rglob("*.*"):
            if f.suffix.lower() not in{".md",".txt"}:continue
            txt=f.read_text(errors="ignore")
            gloss="𓂀 EYE_WATCH " if "mdw" in f.parts else ""
            for chunk,s,e in sliding_windows(txt,1024,512):
                chunks.append(gloss+chunk);
                meta.append((f.relative_to(repo).as_posix(),s,e))
    if not chunks: print("✖ no text");return

    vecs=embed(chunks)

    # clustering
    if not cen_p.exists() or force:
        for k in range(4,257,4):
            km=KMeans(n_clusters=k,n_init=10).fit(vecs)
            if k>8 and km.inertia_/prev_inert<0.99: break
            prev_inert=km.inertia_
        centroids,labels=km.cluster_centers_,km.labels_
        np.save(cen_p,centroids.astype("float32"))
    else:
        frozen=np.load(cen_p)
        mbk=MiniBatchKMeans(n_clusters=frozen.shape[0],init=frozen,n_init=1,batch_size=256)
        labels=mbk.fit_predict(vecs)
        centroids=mbk.cluster_centers_
        np.save(cen_p,centroids.astype("float32"))

    # index
    if idx_p.exists() and (incremental or force):
        idx=faiss.read_index(str(idx_p))
    else:
        idx=faiss.IndexHNSWFlat(DIM,32,faiss.METRIC_INNER_PRODUCT)
    idx.add(vecs)
    faiss.write_index(idx,str(idx_p))

    start_id=idx.ntotal-len(vecs)
    with map_p.open("a" if map_p.exists() else "w",encoding="utf-8")as fp:
        for vid,(file_,s,e),cid in zip(range(start_id,idx.ntotal),meta,labels):
            fp.write(json.dumps({"w":vid,"c":int(cid),"f":file_,"s":s,"e":e})+"\n")

    with omap_p.open("a" if omap_p.exists() else "w",encoding="utf-8")as fp:
        for parent_id,(file_,s,e) in enumerate(meta,start=start_id):
            for _,ss,ee in sliding_windows("",256,128):
                fp.write(json.dumps({"p":parent_id,"f":file_,"s":s+ss,"e":s+ee})+"\n")

    print(f"✔ vectors={idx.ntotal:,} | new={len(vecs):,} | centroids={centroids.shape[0]}")

# ───────────────────────────────────────────
if __name__=="__main__":
    ag=argparse.ArgumentParser();ag.add_argument("--repo-root",required=True)
    ag.add_argument("--incremental",action="store_true");ag.add_argument("--force",action="store_true")
    ar=ag.parse_args();
    if "OPENAI_API_KEY" not in os.environ:sys.exit("✖ OPENAI_API_KEY not set")
    forge(pathlib.Path(ar.repo_root),ar.incremental,ar.force)
