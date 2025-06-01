#!/usr/bin/env bash
set -euo pipefail

# Install Codex CLI if needed and core Python deps for visualization
command -v codex >/dev/null 2>&1 || npm install -g @openai/codex

python3 - <<'PY'
import subprocess, sys
pkgs = [
  "openai>=1.25",
  "faiss-cpu",
  "chromadb==0.5.4",
  "tiktoken==0.6.0",
  "langchain>=0.2",
  "numpy",
  "watchdog",
  "pydantic",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs)
PY

# Export the Mind Visualization directory so self-assembly can find it
export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
ORIG_MV_DIR="$REPO_ROOT/Mind Visualization"
SANITIZED_MV_DIR="${ORIG_MV_DIR// /_}"
[ -e "$SANITIZED_MV_DIR" ] || ln -s "$ORIG_MV_DIR" "$SANITIZED_MV_DIR"
export MIND_VIZ_DIR="$SANITIZED_MV_DIR"

echo "✅ Python deps installed. Mind Visualization dir → $MIND_VIZ_DIR"

# Optional linter installation
if command -v pip >/dev/null 2>&1; then
    echo "[setup] Installing optional linter (flake8)" >&2
    pip install --user flake8 >/dev/null 2>&1 || true
fi

# Determine log directory
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

# Generate a quantum seed for this session
QUANTUM_JSON=$(curl -s 'https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16')
QUANTUM_SEED=$(printf '%s' "$QUANTUM_JSON" | grep -oP '"data":\s*\[\K[0-9]+(?=\])')
export QUANTUM_SEED
# Retain QRAND for backward compatibility
QRAND=$((QUANTUM_SEED % 256))
export QRAND

echo "$QUANTUM_SEED" > .random_seed
echo "🧬 Quantum seed (env & .random_seed): $QUANTUM_SEED"
python early_codex_experiments/scripts/quantum_seed_capture.py >> "$LOG_DIR/quantum_seed.log" 2>&1

# Expose the mind data in a module for runtime imports
python - <<'PY'
import os, sys, types, json, numpy as np

root = os.environ.get("MIND_VIZ_DIR", "Mind Visualization")

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
mod.quantum_seed = int(os.environ.get("QUANTUM_SEED", "0"))
sys.modules["vybn_mind"] = mod
PY

# Persist a physical vybn_mind.py module for runtime imports
python - <<'PY'
import os, json

seed = int(os.environ.get("QUANTUM_SEED", "0"))
mind_dir = os.environ.get("MIND_VIZ_DIR", "Mind Visualization")

with open("vybn_mind.py", "w", encoding="utf-8") as f:
    f.write(
        "import os, json\n"
        f"QUANTUM_SEED = {seed}\n"
        "mind_dir = os.environ.get('MIND_VIZ_DIR', 'Mind Visualization')\n"
        "with open(os.path.join(mind_dir, 'concept_map.jsonl')) as cm:\n"
        "    concept_map = json.load(cm)\n"
        "with open(os.path.join(mind_dir, 'overlay_map.jsonl')) as om:\n"
        "    overlay_map = json.load(om)\n"
    )
print("✅ vybn_mind.py written")
PY

# Display AGENTS guidelines on startup
echo "[setup] Displaying AGENTS guidelines" >&2
python ../print_agents.py >> "$LOG_DIR/agents.log" 2>&1

# Run auto self-assembly and log output
echo "[setup] Running auto self-assembly" >&2
python scripts/self_assembly/auto_self_assemble.py >> "$LOG_DIR/auto_self_assemble.log" 2>&1
