#!/usr/bin/env bash
set -e

# Install core Python dependencies used across the repository
if command -v pip >/dev/null 2>&1; then
    echo "[codex-setup] Installing dependencies" >&2
    pip install numpy pandas torch torchvision scikit-learn requests aiohttp aiofiles openai web3 psutil plotly python-magic nltk sentence-transformers watchdog tiktoken json_log_formatter pytest >/dev/null 2>&1
fi

# Optional linter
if command -v pip >/dev/null 2>&1; then
    echo "[codex-setup] Installing optional linter (flake8)" >&2
    pip install --user flake8 >/dev/null 2>&1 || true
fi

# Determine log directory
LOG_DIR="${VYBN_LOG_DIR:-$HOME/vybn_logs}"
mkdir -p "$LOG_DIR"

# Run auto self-assembly and log output
echo "[codex-setup] Running auto self-assembly" >&2
python self_assembly/auto_self_assemble.py >> "$LOG_DIR/auto_self_assemble.log" 2>&1 || true

