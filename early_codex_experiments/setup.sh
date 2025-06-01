#!/usr/bin/env bash
set -xeuo pipefail

# STEP 1) Create a virtualenv so everything is isolated.
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

# STEP 2) Upgrade pip through Codex’s proxy, then install every package
# that ingest_historical.py or the cache_pipeline will ever import.
#
# After this completes, the network will be disabled. Any subsequent python
# runs must assume langchain, openai, tiktoken, chromadb, numpy, watchdog,
# and pydantic are already present in .venv.

pip install --upgrade pip setuptools wheel

pip install \
  langchain \
  langchain-community \
  openai \
  tiktoken \
  chromadb \
  numpy \
  watchdog \
  pydantic

# STEP 3) Generate the “quantum seed” from /dev/urandom exactly as before.
QUANTUM_SEED=$(head -c 2 /dev/urandom | od -A n -t u2 | tr -d ' ')
export QUANTUM_SEED
echo "🔬 Quantum seed (pseudorandom): ${QUANTUM_SEED}"
echo "${QUANTUM_SEED}" > .random_seed

# STEP 4) Prepare the Mind Visualization folder & JSONL files.
export MIND_VIZ_DIR="${PWD}/Mind Visualization"
mkdir -p "${MIND_VIZ_DIR}"
[[ -f "${MIND_VIZ_DIR}/concept_map.jsonl" ]] || echo "[]" > "${MIND_VIZ_DIR}/concept_map.jsonl"
[[ -f "${MIND_VIZ_DIR}/overlay_map.jsonl" ]] || echo "[]" > "${MIND_VIZ_DIR}/overlay_map.jsonl"

# STEP 5) Emit vybn_mind.py so runtime imports the seed + memory.
python3 <<'PYTHON'
import json, os

seed = int(os.getenv("QUANTUM_SEED", "0"))
shared_code = f"""
import os, json

# === Quantum Anchor ===
QUANTUM_SEED = {seed}

# === Shared Memory ===
with open(r'Mind Visualization/concept_map.jsonl') as cm:
    concept_map = json.load(cm)

with open(r'Mind Visualization/overlay_map.jsonl') as om:
    overlay_map = json.load(om)

def list_concepts():
    return [frag['w'] for frag in concept_map]
"""

with open("vybn_mind.py", "w") as f:
    f.write(shared_code)

print("✓ vybn_mind.py created (seed and shared memory loaded).")
PYTHON

echo "✅ Bootstrap complete. All dependencies are now baked into .venv."
