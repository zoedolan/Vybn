#!/usr/bin/env bash
# start-server.sh — Find, launch, and wait for llama-server with MiniMax M2.5
#
# Usage:
#   bash spark/start-server.sh            # auto-discover binary + model
#   LLAMA_SERVER=/path/to/bin MODEL_PATH=/path/to/gguf bash spark/start-server.sh
#
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$(find ~/llama.cpp -type f -name 'llama-server' 2>/dev/null | head -1)}"
MODEL_PATH="${MODEL_PATH:-$(find ~/llama.cpp/models -type f -name '*MiniMax*-00001-of-*.gguf' 2>/dev/null | head -1)}"
PORT="${LLAMA_PORT:-8081}"
GPU_LAYERS="${GPU_LAYERS:-999}"
LOG_FILE="${HOME}/Vybn/llama_server.log"
TIMEOUT="${STARTUP_TIMEOUT:-600}"

if [ -z "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server binary not found."
    echo "  Set LLAMA_SERVER=/path/to/llama-server and retry."
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$(find ~/llama.cpp/models -maxdepth 1 -type f -name '*MiniMax*.gguf' 2>/dev/null | head -1)"
    if [ -z "$MODEL_PATH" ]; then
        echo "ERROR: No MiniMax GGUF found."
        echo "  Set MODEL_PATH=/path/to/model.gguf and retry."
        exit 1
    fi
fi

echo "Binary:  $LLAMA_SERVER"
echo "Model:   $MODEL_PATH"
echo "Port:    $PORT"
echo "Log:     $LOG_FILE"
echo ""

# Kill any existing server on this port
pkill -f llama-server 2>/dev/null || true
sleep 2

# Launch
nohup "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --port "$PORT" \
    -ngl "$GPU_LAYERS" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Started llama-server (PID $SERVER_PID)"
echo "Waiting for model to load..."

# Poll /health until ready
SECONDS=0
while true; do
    STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo ""
        echo "READY — server healthy after ${SECONDS}s"
        curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: llama-server exited unexpectedly. Last log lines:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    if [ "$SECONDS" -gt "$TIMEOUT" ]; then
        echo ""
        echo "TIMEOUT: server not ready after ${TIMEOUT}s. Last log lines:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    printf "."
    sleep 5
done
