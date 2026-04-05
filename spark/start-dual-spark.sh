#!/usr/bin/env bash
# start-dual-spark.sh — Launch Nemotron FP8 across BOTH DGX Sparks
# ================================================================
#
# This is the CORRECT way to start the model server.
# It uses both Sparks (256 GB total) with tensor parallelism.
#
# DO NOT use llama-server on a single node instead. That wastes
# half the hardware and requires heavy quantization.
#
# Usage:
#   bash ~/Vybn/spark/start-dual-spark.sh          # foreground
#   bash ~/Vybn/spark/start-dual-spark.sh --daemon  # background
#
# Prerequisites:
#   - Both Sparks reachable (ping 169.254.51.101)
#   - spark-vllm-docker repo at ~/spark-vllm-docker
#   - vllm-node Docker image built on both nodes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="$HOME/spark-vllm-docker"
RECIPE="nemotron-3-super-fp8.yaml"
REMOTE_IP="169.254.51.101"

echo "=== Dual Spark Startup ==="
echo "Primary:   spark-2b7c (this machine)"
echo "Secondary: spark-1c8f ($REMOTE_IP)"
echo ""

# Step 1: Verify the second Spark is reachable
echo "Checking connectivity to spark-1c8f..."
if ! ping -c 1 -W 3 "$REMOTE_IP" >/dev/null 2>&1; then
    echo "ERROR: Cannot reach spark-1c8f at $REMOTE_IP"
    echo "The second Spark must be online for dual-node serving."
    echo "Check the CX7 cable and network configuration."
    exit 1
fi
echo "✓ spark-1c8f reachable"

# Step 2: Verify the second Spark's GPU is available
echo "Checking GPU on spark-1c8f..."
REMOTE_GPU=$(ssh -o ConnectTimeout=5 "$REMOTE_IP" "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null || echo "UNREACHABLE")
if [[ "$REMOTE_GPU" == "UNREACHABLE" ]]; then
    echo "ERROR: Cannot SSH to spark-1c8f or nvidia-smi failed"
    exit 1
fi
echo "✓ spark-1c8f GPU: $REMOTE_GPU"

# Step 3: Kill any existing single-node llama-server (it's the wrong config)
if pgrep -f "llama-server" >/dev/null 2>&1; then
    echo "WARNING: Killing single-node llama-server (wrong config for dual Spark)"
    pkill -f "llama-server" || true
    sleep 2
fi

# Step 4: Launch the cluster using the recipe
echo ""
echo "Launching Nemotron FP8 across both Sparks..."
cd "$CLUSTER_DIR"

DAEMON_FLAG=""
if [[ "${1:-}" == "--daemon" || "${1:-}" == "-d" ]]; then
    DAEMON_FLAG="-d"
    echo "(Running in daemon mode)"
fi

# Use run-recipe.sh with the FP8 recipe
./run-recipe.sh $DAEMON_FLAG "$RECIPE"

echo ""
echo "=== Dual Spark serving started ==="
echo "API endpoint: http://localhost:8000/v1"
echo "Both Sparks active. 256 GB unified memory."
