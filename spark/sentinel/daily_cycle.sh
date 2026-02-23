#!/bin/bash
# daily_cycle.sh — The Spark's daily rhythm
#
# Orchestrates everything that needs the local model:
#   1. Start llama-server
#   2. Run sentinel (news crawl + claim extraction)
#   3. [Future] Run heartbeat pulse
#   4. [Future] Run training data review
#   5. Shut down server
#
# Call this from cron at 7:00 AM and 7:00 PM PST.
# Total runtime: ~45-90 min depending on feed volume.
#
# Usage: ./daily_cycle.sh [morning|evening]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPARK_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$SPARK_DIR")"
LOG_DIR="$SCRIPT_DIR/data"
LOG="$LOG_DIR/daily_cycle.log"
MODEL="/home/vybnz69/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-merged.gguf"
LLAMA="/home/vybnz69/llama.cpp/build/bin/llama-server"
PORT=8081
HEALTH_URL="http://localhost:$PORT/health"
MAX_WAIT=600  # 10 minutes max for model load

CYCLE="${1:-auto}"
HOUR=$(date +%H)
if [ "$CYCLE" = "auto" ]; then
    if [ "$HOUR" -lt 12 ]; then CYCLE="morning"; else CYCLE="evening"; fi
fi

mkdir -p "$LOG_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$CYCLE] $*" | tee -a "$LOG"
}

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "Cleanup: stopping llama-server (PID $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Prevent concurrent runs
LOCKFILE="/tmp/vybn_daily_cycle.lock"
if [ -f "$LOCKFILE" ]; then
    LOCK_PID=$(cat "$LOCKFILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        log "SKIP: Another cycle is running (PID $LOCK_PID)"
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"

log "=== DAILY CYCLE START ($CYCLE) ==="

# --- PHASE 1: Start model server ---
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
        > "$LOG_DIR/llama_server.log" 2>&1 &
    SERVER_PID=$!
    log "llama-server PID: $SERVER_PID (waiting for health...)"
    
    ELAPSED=0
    while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        if [ $ELAPSED -ge $MAX_WAIT ]; then
            log "ERROR: llama-server failed to start within ${MAX_WAIT}s"
            exit 1
        fi
        # Progress every 60s
        if [ $((ELAPSED % 60)) -eq 0 ]; then
            log "Still loading model... (${ELAPSED}s)"
        fi
    done
    log "llama-server healthy after ${ELAPSED}s"
fi

# --- PHASE 2: Sentinel ---
log "Running sentinel..."
cd "$SCRIPT_DIR"
if python3 -m sentinel.main --config config.yaml --local-only >> "$LOG" 2>&1; then
    log "Sentinel completed successfully"
    
    # Count results
    LATEST_CLAIMS=$(ls -t data/structured/claims_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_CLAIMS" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$LATEST_CLAIMS'))))")
        log "Extracted $COUNT claims"
    fi
else
    log "WARNING: Sentinel exited with error"
fi

# --- PHASE 3: Git sync ---
log "Syncing repo..."
cd "$REPO_DIR"
git fetch origin main --quiet 2>/dev/null || log "WARNING: git fetch failed"

# --- PHASE 4: Cleanup ---
# Kill the server to free GPU
if [ -n "$SERVER_PID" ]; then
    log "Stopping llama-server..."
    kill "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    log "llama-server stopped"
fi

rm -f "$LOCKFILE"
log "=== DAILY CYCLE END ($CYCLE) ==="
