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

# Display AGENTS guidelines on startup
echo "[setup] Displaying AGENTS guidelines" >&2
python ../print_agents.py >> "$LOG_DIR/agents.log" 2>&1

# Run auto self-assembly and log output
echo "[setup] Running auto self-assembly" >&2
python scripts/self_assembly/auto_self_assemble.py >> "$LOG_DIR/auto_self_assemble.log" 2>&1
