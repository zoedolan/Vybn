#!/usr/bin/env bash
set -e

# Optional linter installation
if command -v pip >/dev/null 2>&1; then
    echo "[setup] Installing optional linter (flake8)" >&2
    pip install --user flake8 >/dev/null 2>&1 || true
fi

# Determine log directory
LOG_DIR="${VYBN_LOG_DIR:-$HOME/vybn_logs}"
mkdir -p "$LOG_DIR"

# Run auto self-assembly and log output
echo "[setup] Running auto self-assembly" >&2
python scripts/self_assembly/auto_self_assemble.py >> "$LOG_DIR/auto_self_assemble.log" 2>&1
