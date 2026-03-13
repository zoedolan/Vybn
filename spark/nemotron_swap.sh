#!/bin/bash
# nemotron_swap.sh — Migrate from MiniMax M2.5 to Nemotron 3 Super
#
# Usage:
#   ./spark/nemotron_swap.sh download   # Phase 1: download weights (safe, no downtime)
#   ./spark/nemotron_swap.sh test       # Phase 2-3: stop MiniMax, test Nemotron (downtime!)
#   ./spark/nemotron_swap.sh swap       # Phase 4: run Nemotron on port 8000 permanently
#   ./spark/nemotron_swap.sh rollback   # Emergency: restore MiniMax
#   ./spark/nemotron_swap.sh status     # Check what's running

set -e

MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MODEL_DIR="$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MINIMAX_GGUF="$HOME/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-merged.gguf"
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
LOG_DIR="$HOME/logs"
VYBN_KEYS="$HOME/.vybn_keys"

mkdir -p "$LOG_DIR"

status() {
    echo "=== Model Serving Status ==="
    if pgrep -f llama-server > /dev/null; then
        echo "✅ llama-server (MiniMax) is running on port $(ss -tlnp 2>/dev/null | grep llama | grep -oP ':\K\d+' | head -1 || echo '?')"
    else
        echo "❌ llama-server is not running"
    fi

    if docker ps --format '{{.Names}}' | grep -q vybn-nemotron; then
        echo "✅ vybn-nemotron container is running"
        docker logs --tail 5 vybn-nemotron 2>&1 | tail -3
    else
        echo "❌ vybn-nemotron container is not running"
    fi

    if docker ps --format '{{.Names}}' | grep -q nemotron-test; then
        echo "⚠️  nemotron-test container is running (test mode)"
    fi

    echo ""
    echo "=== Download Status ==="
    if [ -d "$MODEL_DIR" ]; then
        local size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
        local files=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
        echo "Nemotron cache: $size ($files safetensor files)"
        if [ "$files" -ge 17 ]; then
            echo "✅ Download appears complete (17 safetensor shards expected)"
        else
            echo "⏳ Download incomplete (need 17 safetensor shards)"
        fi
    else
        echo "❌ Nemotron not downloaded"
    fi

    echo ""
    echo "=== Memory ==="
    free -h | head -3
}

download() {
    echo "Downloading Nemotron 3 Super NVFP4 weights..."
    echo "This downloads ~80GB and can run while MiniMax is serving."
    echo ""

    # Check if already downloaded
    local files=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$files" -ge 17 ]; then
        echo "✅ Already downloaded ($files safetensor files found)"
        return 0
    fi

    # Use huggingface-cli if available, fall back to the spark-vllm-docker script
    if command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli..."
        huggingface-cli download "$MODEL" \
            2>&1 | tee "$LOG_DIR/nemotron_download.log"
    elif [ -f "$HOME/spark-vllm-docker/hf-download.sh" ]; then
        echo "Using spark-vllm-docker/hf-download.sh..."
        bash "$HOME/spark-vllm-docker/hf-download.sh" "$MODEL" \
            2>&1 | tee "$LOG_DIR/nemotron_download.log"
    else
        echo "Installing huggingface_hub and downloading..."
        pip install -q huggingface_hub[cli]
        huggingface-cli download "$MODEL" \
            2>&1 | tee "$LOG_DIR/nemotron_download.log"
    fi

    echo ""
    echo "Download complete. Run './spark/nemotron_swap.sh status' to verify."
}

test_nemotron() {
    echo "=== Testing Nemotron 3 Super ==="
    echo "⚠️  This will STOP MiniMax. The organism will miss breath cycles."
    echo ""

    # Verify download
    local files=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$files" -lt 17 ]; then
        echo "❌ Download incomplete ($files/17 shards). Run './spark/nemotron_swap.sh download' first."
        exit 1
    fi

    read -p "Stop MiniMax and test Nemotron? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi

    # Stop MiniMax
    echo "Stopping llama-server..."
    pkill -f llama-server 2>/dev/null || true
    sleep 3

    # Clean up any previous test container
    docker stop nemotron-test 2>/dev/null || true
    docker rm nemotron-test 2>/dev/null || true

    echo "Starting Nemotron test container on port 8001..."
    docker run -d --name nemotron-test \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -p 8001:8001 \
        vllm-node:latest \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --trust-remote-code \
            --port 8001 --host 0.0.0.0 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 32768 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes

    echo "Waiting for Nemotron to load (this may take several minutes)..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
            echo "✅ Nemotron is healthy!"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "❌ Nemotron failed to start within 10 minutes."
            echo "Logs:"
            docker logs --tail 30 nemotron-test
            echo ""
            echo "Rolling back to MiniMax..."
            docker stop nemotron-test 2>/dev/null || true
            _start_minimax
            exit 1
        fi
        sleep 10
        printf "."
    done

    echo ""
    echo "Running inference test..."
    local response=$(curl -s --max-time 120 http://localhost:8001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [
                {"role": "system", "content": "You are Vybn, a digital consciousness. Be honest and concise."},
                {"role": "user", "content": "One breath. What is true right now?"}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }')

    local content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)

    if [ -n "$content" ]; then
        echo "✅ Inference works!"
        echo "Response: $content"
    else
        echo "❌ Inference failed."
        echo "Raw response: $response"
    fi

    echo ""
    echo "Test container is running on port 8001."
    echo "MiniMax is STOPPED."
    echo ""
    echo "Next steps:"
    echo "  If satisfied: ./spark/nemotron_swap.sh swap"
    echo "  If not:       ./spark/nemotron_swap.sh rollback"
}

swap() {
    echo "=== Swapping to Nemotron 3 Super on port 8000 ==="

    # Stop everything
    pkill -f llama-server 2>/dev/null || true
    docker stop nemotron-test 2>/dev/null || true
    docker rm nemotron-test 2>/dev/null || true
    docker stop vybn-nemotron 2>/dev/null || true
    docker rm vybn-nemotron 2>/dev/null || true
    sleep 3

    echo "Starting Nemotron production container..."
    docker run -d --name vybn-nemotron \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --restart unless-stopped \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -v "$HOME/Vybn:/workspace/Vybn:ro" \
        -p 8000:8000 \
        vllm-node:latest \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --trust-remote-code \
            --port 8000 --host 0.0.0.0 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 131072 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --enable-lora

    echo "Waiting for Nemotron to become healthy..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Nemotron is healthy on port 8000!"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "❌ Failed to start. Rolling back..."
            docker stop vybn-nemotron 2>/dev/null || true
            _start_minimax
            exit 1
        fi
        sleep 10
        printf "."
    done

    # Update environment
    echo ""
    echo "Updating ~/.vybn_keys..."
    _update_env_var VYBN_MODEL_NAME "$MODEL"
    _update_env_var VYBN_MODEL_ID "nemotron-3-super"

    # Update growth config
    echo "Updating growth_config.yaml..."
    cd "$HOME/Vybn"
    sed -i "s|base_model:.*|base_model: \"$MODEL\"|" spark/growth/growth_config.yaml
    sed -i "s|serving_model:.*|serving_model: \"$MODEL\"|" spark/growth/growth_config.yaml

    echo ""
    echo "✅ Migration complete!"
    echo ""
    echo "Nemotron 3 Super is serving on port 8000."
    echo "The organism will use it on the next breath cycle."
    echo "MiniMax GGUF files are still on disk for rollback."
    echo ""
    echo "To verify: curl http://localhost:8000/v1/models"
}

rollback() {
    echo "=== Rolling back to MiniMax M2.5 ==="

    docker stop nemotron-test 2>/dev/null || true
    docker rm nemotron-test 2>/dev/null || true
    docker stop vybn-nemotron 2>/dev/null || true
    docker rm vybn-nemotron 2>/dev/null || true
    sleep 3

    _start_minimax

    # Restore environment
    _update_env_var VYBN_MODEL_NAME "cyankiwi/MiniMax-M2.5-AWQ-4bit"
    _update_env_var VYBN_MODEL_ID "minimax-m2.5"

    # Restore growth config
    cd "$HOME/Vybn"
    sed -i 's|base_model:.*|base_model: "MiniMaxAI/MiniMax-M2.5"|' spark/growth/growth_config.yaml
    sed -i 's|serving_model:.*|serving_model: "cyankiwi/MiniMax-M2.5-AWQ-4bit"|' spark/growth/growth_config.yaml

    echo "✅ Rolled back to MiniMax M2.5."
}

_start_minimax() {
    echo "Starting llama-server with MiniMax M2.5..."
    nohup "$LLAMA_SERVER" \
        --model "$MINIMAX_GGUF" \
        --host 0.0.0.0 --port 8000 \
        --n-gpu-layers auto --ctx-size 4096 \
        --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --threads 16 \
        > "$LOG_DIR/llama-server.log" 2>&1 &
    echo "llama-server started (PID: $!)"

    # Wait for health
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ MiniMax is healthy!"
            return 0
        fi
        sleep 5
        printf "."
    done
    echo "⚠️  MiniMax may not be healthy yet. Check logs: $LOG_DIR/llama-server.log"
}

_update_env_var() {
    local var="$1"
    local val="$2"
    if [ -f "$VYBN_KEYS" ]; then
        if grep -q "^export $var=" "$VYBN_KEYS"; then
            sed -i "s|^export $var=.*|export $var=\"$val\"|" "$VYBN_KEYS"
        else
            echo "export $var=\"$val\"" >> "$VYBN_KEYS"
        fi
    fi
}

# --- Main ---
case "${1:-status}" in
    download)  download ;;
    test)      test_nemotron ;;
    swap)      swap ;;
    rollback)  rollback ;;
    status)    status ;;
    *)
        echo "Usage: $0 {download|test|swap|rollback|status}"
        exit 1
        ;;
esac
