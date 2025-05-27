# .devcontainer/bootstrap.sh  – tail-end graft
export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
# ensure cache dependencies are present (skip if offline)
# pip install is optional since dev container may already have deps
ORIG_MV_DIR="$REPO_ROOT/Mind Visualization"
SANITIZED_MV_DIR="${ORIG_MV_DIR// /_}"
[ -e "$SANITIZED_MV_DIR" ] || ln -s "$ORIG_MV_DIR" "$SANITIZED_MV_DIR"
export MIND_VIZ_DIR="$SANITIZED_MV_DIR"   # sanitized path, original stored via symlink

LOG_DIR="${VYBN_LOG_DIR:-$HOME/vybn_logs}"
mkdir -p "$LOG_DIR"
touch "$LOG_DIR/chat.log"

python - <<'PY'
import os, sys, types, json, numpy as np

root = os.environ["MIND_VIZ_DIR"]

# ---- vector index ----------------------------------------------------------
index = None
try:
    import faiss
    idx_path = os.path.join(root, "history_memoirs.hnsw")
    index = faiss.read_index(idx_path)
except Exception:
    try:
        import hnswlib
        idx_path = os.path.join(root, "history_memoirs.hnsw")
        cc_path = os.path.join(root, "concept_centroids.npy")
        if os.path.exists(cc_path):
            dim = np.load(cc_path).shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.load_index(idx_path)
    except Exception:
        index = None

# ---- metadata --------------------------------------------------------------
cc_path = os.path.join(root, "concept_centroids.npy")
centroids = np.load(cc_path) if os.path.exists(cc_path) else None

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
