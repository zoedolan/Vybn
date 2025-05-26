#!/usr/bin/env bash
set -euo pipefail

# Path to the canonical concept index
INDEX_PATH="/artifacts/vybn_concept_index.jsonl"
REPO_ROOT="$(git rev-parse --show-toplevel)"

mkdir -p /artifacts

if [ ! -f "$INDEX_PATH" ]; then
    echo "[bootstrap] Building concept index" >&2
    python "$REPO_ROOT/early_codex_experiments/build_concept_index.py" --repo-root "$REPO_ROOT" || true
    cp "$REPO_ROOT/Mind Visualization/concept_map.jsonl" "$INDEX_PATH"
fi

echo "export VYBN_CONCEPT_INDEX=$INDEX_PATH" >> /etc/profile.d/vybn_env.sh

echo "[bootstrap] Concept index → $INDEX_PATH"
