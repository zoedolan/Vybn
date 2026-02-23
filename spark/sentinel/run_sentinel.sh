#!/bin/bash
# run_sentinel.sh — Runs a complete sentinel cycle:
#   1. Start llama-server if not already running
#   2. Wait for it to be healthy
#   3. Run sentinel
#   4. Kill llama-server when done (free GPU)
#
# Designed to be called from cron. All output goes to sentinel_cron.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/data/sentinel_cron.log"
MODEL="/home/vybnz69/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-merged.gguf"
LLAMA="/home/vybnz69/llama.cpp/build/bin/llama-server"
PORT=8081
HEALTH_URL="http://localhost:$PORT/health"
MAX_WAIT=300  # 5 minutes to load model

mkdir -p "$SCRIPT_DIR/data"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"
}

# Check if another sentinel is already running
if pgrep -f "run_sentinel.sh" | grep -v $$ > /dev/null 2>&1; then
    log "SKIP: Another sentinel instance is running"
    exit 0
fi

log "=== SENTINEL CYCLE START ==="

# 1. Start llama-server if needed
SERVER_PID=""
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    log "llama-server already running"
else
    log "Starting llama-server..."
    $LLAMA \
        --model "$MODEL" \
        --host 127.0.0.1 \
        --port $PORT \
        --ctx-size 4096 \
        --n-gpu-layers 99 \
        --parallel 1 \
        --threads 8 \
        > "$SCRIPT_DIR/data/llama_server.log" 2>&1 &
    SERVER_PID=$!
    log "llama-server PID: $SERVER_PID"
    
    # Wait for health check
    ELAPSED=0
    while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            log "ERROR: llama-server failed to start within ${MAX_WAIT}s"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    done
    log "llama-server healthy after ${ELAPSED}s"
fi

# 2. Run sentinel
log "Running sentinel..."
cd "$SCRIPT_DIR"
python3 -m sentinel.main --config config.yaml --local-only >> "$LOG" 2>&1
EXIT_CODE=$?
log "Sentinel exited with code $EXIT_CODE"

# 3. Kill the server we started (if we started it)
if [ -n "$SERVER_PID" ]; then
    log "Stopping llama-server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    # Wait up to 30s for graceful shutdown
    for i in $(seq 1 30); do
        kill -0 $SERVER_PID 2>/dev/null || break
        sleep 1
    done
    # Force kill if still alive
    kill -9 $SERVER_PID 2>/dev/null || true
    log "llama-server stopped"
fi

log "=== SENTINEL CYCLE END ==="
