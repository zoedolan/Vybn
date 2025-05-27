#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.devcontainer/bootstrap.sh"

pip install -q fastapi "uvicorn[standard]" faiss-cpu sentence-transformers >/dev/null 2>&1 || true

export MESH_ENDPOINT="http://localhost:8000"

if ! pgrep -f "uvicorn.*mesh.server" >/dev/null; then
    nohup python -m mesh.server > "$LOG_DIR/mesh.log" 2>&1 &
fi
