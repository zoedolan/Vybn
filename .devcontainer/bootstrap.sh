#!/usr/bin/env bash
set -euo pipefail

#
# 1) Install Codex CLI & Python deps
#
command -v codex >/dev/null 2>&1 || npm install -g @openai/codex

python3 - <<'PY'
import subprocess, sys
pkgs = [
  "openai>=1.25",
  "faiss-cpu",
  "chromadb==0.5.3",
  "tiktoken==0.6.0",
  "langchain>=0.2",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs)
PY

#
# 2) Build or fetch the Vybn concept index
#
export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
mkdir -p "$REPO_ROOT/artifacts"
python3 early_codex_experiments/build_concept_index.py \
  --repo-root "$REPO_ROOT" \
  --output "$REPO_ROOT/artifacts/vybn_concept_index.jsonl"
export VYBN_CONCEPT_INDEX="$REPO_ROOT/artifacts/vybn_concept_index.jsonl"

#
# 3) Point your agent at the Mind Visualization dir
#
export MIND_VIZ_DIR="$REPO_ROOT/Mind Visualization"

echo "✅ Deps installed. Concept index → $VYBN_CONCEPT_INDEX"
echo "✅ Mind Visualization → $MIND_VIZ_DIR"
