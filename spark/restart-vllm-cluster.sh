#!/usr/bin/env bash
# restart-vllm-cluster.sh
# Idempotent: safe to run at @reboot or manually.
# Ensures the vllm_node training container is running with the correct mounts.
#
# The container provides the GPU-enabled Python environment (PyTorch, PEFT,
# TRL, mamba-ssm, causal-conv1d) for LoRA fine-tuning. It needs two mounts:
#   1. /home/vybnz69/Vybn → /workspace/Vybn (the repo)
#   2. /home/vybnz69/.cache/huggingface → /root/.cache/huggingface (model weights)
#
# Also ensures llama-server is running for inference.
#
# Install in crontab on spark-2b7c (crontab -e):
#   @reboot sleep 30 && /home/vybnz69/Vybn/spark/restart-vllm-cluster.sh >> ~/vllm_restart.log 2>&1

set -euo pipefail

LOG_PREFIX="[restart-vllm] $(date '+%Y-%m-%d %H:%M:%S')"
VLLM_CONTAINER="vllm_node"
VLLM_IMAGE="vllm-node:latest"
HEALTH_URL="http://localhost:8000/health"
MAX_WAIT=300  # seconds to wait for health

echo "$LOG_PREFIX starting vllm_node setup"

# 1. If container exists, check it has the correct mounts.
#    If mounts are wrong, recreate it.
if docker ps -q -f name="$VLLM_CONTAINER" | grep -q .; then
    # Container running — verify mounts
    HAS_HF_MOUNT=$(docker inspect "$VLLM_CONTAINER" \
        --format='{{range .HostConfig.Binds}}{{.}}{{"\n"}}{{end}}' 2>/dev/null \
        | grep -c ".cache/huggingface" || true)
    if [ "$HAS_HF_MOUNT" -gt 0 ]; then
        echo "$LOG_PREFIX $VLLM_CONTAINER running with correct mounts — nothing to do"
    else
        echo "$LOG_PREFIX $VLLM_CONTAINER missing HF cache mount — recreating"
        docker stop "$VLLM_CONTAINER" 2>/dev/null || true
        docker rm "$VLLM_CONTAINER" 2>/dev/null || true
    fi
elif docker ps -aq -f name="$VLLM_CONTAINER" | grep -q .; then
    echo "$LOG_PREFIX $VLLM_CONTAINER exists but stopped — removing to recreate"
    docker rm "$VLLM_CONTAINER" 2>/dev/null || true
fi

# 2. Create container if it doesn't exist
if ! docker ps -aq -f name="$VLLM_CONTAINER" | grep -q .; then
    echo "$LOG_PREFIX creating $VLLM_CONTAINER with repo + HF cache mounts"
    docker run -d \
        --name "$VLLM_CONTAINER" \
        --runtime nvidia \
        --gpus all \
        -v /home/vybnz69/Vybn:/workspace/Vybn \
        -v /home/vybnz69/.cache/huggingface:/root/.cache/huggingface \
        "$VLLM_IMAGE" \
        sleep infinity
    echo "$LOG_PREFIX container created"

    # Install PEFT + TRL (not baked into the image yet)
    echo "$LOG_PREFIX installing PEFT + TRL in container"
    docker exec "$VLLM_CONTAINER" \
        pip install --quiet peft==0.18.1 trl==0.29.0 2>&1 | tail -3
    echo "$LOG_PREFIX PEFT + TRL installed"
fi

# 3. Ensure container is running
if ! docker ps -q -f name="$VLLM_CONTAINER" | grep -q .; then
    docker start "$VLLM_CONTAINER"
    echo "$LOG_PREFIX started $VLLM_CONTAINER"
fi

# 4. Check llama-server health (separate from the container)
if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
    echo "$LOG_PREFIX llama-server healthy"
else
    echo "$LOG_PREFIX WARNING: llama-server not responding at $HEALTH_URL"
    echo "$LOG_PREFIX   (llama-server is managed separately via vybn-llama.service)"
fi

echo "$LOG_PREFIX done"
