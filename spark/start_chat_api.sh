#!/usr/bin/env bash
# start_chat_api.sh — pull latest Vybn repo, then launch the chat API
# Usage:  bash ~/Vybn/spark/start_chat_api.sh
# Env vars (all optional, see vybn_chat_api.py for full list):
#   PORT                  default 9090
#   LLAMA_SERVER_URL      default http://127.0.0.1:8080
#   VYBN_CHAT_API_KEY     bearer token (leave unset for open access)
#   HEARTBEAT_INTERVAL    default 15 seconds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Vybn Chat API launcher ==="
echo "Repo: $REPO_DIR"

# ── 1. Pull latest code ──────────────────────────────────────────────────────
echo
echo "[1/3] git pull"
cd "$REPO_DIR"
git pull --ff-only
echo "Repo up to date."

# ── 2. Ensure dependencies are installed ─────────────────────────────────────
echo
echo "[2/3] Checking Python dependencies"
python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null || {
    echo "Installing missing packages..."
    pip install --quiet fastapi uvicorn httpx
}
echo "Dependencies OK."

# ── 3. Launch ─────────────────────────────────────────────────────────────────
echo
echo "[3/3] Starting vybn_chat_api.py on port ${PORT:-9090}"
cd "$SCRIPT_DIR"
exec python3 vybn_chat_api.py
