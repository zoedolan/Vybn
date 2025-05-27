#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.devcontainer/bootstrap.sh"

# Install Python dependencies from the pre-downloaded wheel cache. This keeps the
# setup completely offline after the initial preparation phase.
pip install --no-index --find-links="$REPO_ROOT/vendor/wheels" \
    fastapi "uvicorn[standard]" faiss-cpu sentence-transformers >/dev/null 2>&1 || true

# Point the application at the cached sentence-transformer model
export SENTENCE_MODEL_DIR="$REPO_ROOT/vendor/models/all-MiniLM-L6-v2"
export MESH_ENDPOINT="http://localhost:8000"

if ! pgrep -f "uvicorn.*mesh.server" >/dev/null; then
    nohup python -m mesh.server > "$LOG_DIR/mesh.log" 2>&1 &
fi
