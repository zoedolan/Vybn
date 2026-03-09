#!/usr/bin/env bash
# restart-vllm-cluster.sh
# Idempotent: safe to run at @reboot or manually.
# Brings up the distributed MiniMax M2.5 vLLM cluster across both Sparks.
#
# Install in crontab on spark-2b7c (crontab -e):
#   @reboot sleep 30 && /home/zoe/Vybn/spark/restart-vllm-cluster.sh >> ~/vllm_restart.log 2>&1

set -euo pipefail

LOG_PREFIX="[restart-vllm] $(date '+%Y-%m-%d %H:%M:%S')"
MODEL="cyankiwi/MiniMax-M2.5-AWQ-4bit"
VLLM_CONTAINER="vllm_node"
RAY_WORKER_SSH_HOST="spark-1c8f"
HEALTH_URL="http://localhost:8000/v1/models"
MAX_WAIT=300  # seconds to wait for model load

echo "$LOG_PREFIX starting vLLM cluster for $MODEL"

# 1. Already up? Nothing to do.
if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
    echo "$LOG_PREFIX vLLM already responding at $HEALTH_URL — nothing to do"
    exit 0
fi

# 2. Ensure Ray head is running on this node.
if ! ray status > /dev/null 2>&1; then
    echo "$LOG_PREFIX starting Ray head on spark-2b7c"
    ray start --head --port=6379 --dashboard-host=0.0.0.0 2>&1 | tail -5
else
    echo "$LOG_PREFIX Ray head already running"
fi

# 3. Ensure Ray worker on spark-1c8f is connected.
RAY_HEAD_IP=$(hostname -I | awk '{print $1}')
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$RAY_WORKER_SSH_HOST" \
    "ray status 2>/dev/null | grep -q 'Active' || ray start --address='${RAY_HEAD_IP}:6379'" 2>&1; then
    echo "$LOG_PREFIX Ray worker on spark-1c8f confirmed"
else
    echo "$LOG_PREFIX WARNING: could not confirm Ray worker on spark-1c8f — proceeding single-node"
fi

# 4. Start or restart the vllm_node container.
if docker ps -q -f name="$VLLM_CONTAINER" | grep -q .; then
    echo "$LOG_PREFIX $VLLM_CONTAINER already running"
elif docker ps -aq -f name="$VLLM_CONTAINER" | grep -q .; then
    echo "$LOG_PREFIX restarting stopped container $VLLM_CONTAINER"
    docker start "$VLLM_CONTAINER"
else
    echo "$LOG_PREFIX ERROR: $VLLM_CONTAINER container not found."
    echo "$LOG_PREFIX Recover the original docker run command with:"
    echo "$LOG_PREFIX   docker inspect $VLLM_CONTAINER 2>/dev/null || docker history $VLLM_CONTAINER"
    exit 1
fi

# 5. Wait for health check.
echo "$LOG_PREFIX waiting up to ${MAX_WAIT}s for API to become healthy..."
WAITED=0
until curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "$LOG_PREFIX TIMEOUT after ${MAX_WAIT}s — last container logs:"
        docker logs "$VLLM_CONTAINER" --tail 20 2>&1
        exit 2
    fi
    echo "$LOG_PREFIX   still loading... (${WAITED}s elapsed)"
done

echo "$LOG_PREFIX vLLM cluster healthy"
curl -sf "$HEALTH_URL" | python3 -m json.tool 2>/dev/null || true
