# .devcontainer/bootstrap.sh  – tail-end graft
export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
export MIND_VIZ_DIR="$REPO_ROOT/Vybn/Mind Visualization"   # exact repo path

python - <<'PY'
import os, sys, types, json, numpy as np

root = os.environ["MIND_VIZ_DIR"]

# ---- vector index ----------------------------------------------------------
try:
    import faiss
    idx_path = os.path.join(root, "history_memoirs.hnsw")
    index = faiss.read_index(idx_path)               # faiss can read any ext
except Exception:                                    # fallback to hnswlib
    import hnswlib
    import numpy as np
    idx_path = os.path.join(root, "history_memoirs.hnsw")
    dim = np.load(os.path.join(root, "concept_centroids.npy")).shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(idx_path)

# ---- metadata --------------------------------------------------------------
centroids = np.load(os.path.join(root, "concept_centroids.npy"))  # (k, d)

def _read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

concept_map = _read_jsonl(os.path.join(root, "concept_map.jsonl"))
overlay_map = _read_jsonl(os.path.join(root, "overlay_map.jsonl"))

# ---- publish into a hot module --------------------------------------------
mod = types.ModuleType("vybn_mind")
mod.index, mod.centroids = index, centroids
mod.concept_map, mod.overlay_map = concept_map, overlay_map
sys.modules["vybn_mind"] = mod
PY
