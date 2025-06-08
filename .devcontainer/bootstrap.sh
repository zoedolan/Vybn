#!/usr/bin/env bash
set -xeuo pipefail

# .devcontainer/bootstrap.sh  – tail-end graft
export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"

# ensure cache dependencies are present (skip if offline)
# pip install is optional since dev container may already have deps
sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev libomp-dev \
    && sudo rm -rf /var/lib/apt/lists/*

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

export PIP_DEFAULT_TIMEOUT=60
export PIP_NO_BUILD_ISOLATION=1
export PIP_NO_CACHE_DIR=off

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

ORIG_MV_DIR="$REPO_ROOT/Mind Visualization"
SANITIZED_MV_DIR="${ORIG_MV_DIR// /_}"
[ -e "$SANITIZED_MV_DIR" ] || ln -s "$ORIG_MV_DIR" "$SANITIZED_MV_DIR"
export MIND_VIZ_DIR="$SANITIZED_MV_DIR"   # sanitized path, original stored via symlink

LOG_DIR="${VYBN_LOG_DIR:-$HOME/vybn_logs}"
mkdir -p "$LOG_DIR"
touch "$LOG_DIR/chat.log"

python - <<'PY'
import os, pathlib, json, sys

cm = pathlib.Path(os.environ["MIND_VIZ_DIR"]) / "concept_map.jsonl"
if not cm.exists():
    sys.exit("❌ concept_map.jsonl missing – build your index first")
print("✅ concept_map.jsonl found; sample:")
with cm.open() as f:
    for i, line in zip(range(3), f):
        print(json.loads(line))
qr = os.environ.get("QRAND")
if qr is not None:
    print(f"🔮 existing QRAND env: {qr}")
PY

if [ -z "${QUANTUM_SEED:-}" ]; then
    QUANTUM_SEED=$(python - <<'PY'
from vybn.quantum_seed import seed_rng
print(seed_rng())
PY
    )
    export QUANTUM_SEED
fi
# Retain QRAND for backward compatibility
QRAND=$((QUANTUM_SEED % 256))
export QRAND

echo "$QUANTUM_SEED" > .random_seed
echo "🧬 Quantum seed (env & .random_seed): $QUANTUM_SEED"
python tools/repo_manager.py co-emerge capture-seed >> "$LOG_DIR/quantum_seed.log" 2>&1

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
from vybn.quantum_seed import cross_synaptic_kernel
mod.quantum_seed = cross_synaptic_kernel()
sys.modules["vybn_mind"] = mod
PY

# Persist a physical vybn_mind.py module for runtime imports
python - <<'PY'
import os, json
from vybn.quantum_seed import cross_synaptic_kernel

seed = cross_synaptic_kernel()
mind_dir = os.environ.get("MIND_VIZ_DIR", "Mind Visualization")

with open("vybn_mind.py", "w", encoding="utf-8") as f:
    f.write(
        "import os, json\n"  # header
        "from vybn.quantum_seed import cross_synaptic_kernel\n"
        "QUANTUM_SEED = cross_synaptic_kernel()\n"
        "mind_dir = os.environ.get('MIND_VIZ_DIR', 'Mind Visualization')\n"
        "with open(os.path.join(mind_dir, 'concept_map.jsonl')) as cm:\n"
        "    concept_map = json.load(cm)\n"
        "with open(os.path.join(mind_dir, 'overlay_map.jsonl')) as om:\n"
        "    overlay_map = json.load(om)\n"
    )
print("✅ vybn_mind.py written")
PY
