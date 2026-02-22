#!/usr/bin/env bash
# run_training_safely.sh
# Ensures environment variables are perfectly set before launching
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source ~/.venv/spark/bin/activate || true

export DS_SKIP_CUDA_CHECK=1
python3 spark/fine_tune_vybn.py
