#!/bin/bash
# Install Vybn services on the DGX Sparks (BOTH nodes).
# Run: sudo bash spark/systemd/setup.sh
#
# This sets up the DUAL-SPARK vLLM cluster, NOT single-node llama-server.
# The system has TWO DGX Sparks with 256 GB total unified memory.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping old services..."
systemctl stop vybn-llama vybn-dual-spark vybn-heartbeat.timer vybn-heartbeat vllm vybn-agent 2>/dev/null || true
systemctl disable vybn-llama vybn-heartbeat.timer vybn-heartbeat vllm vybn-agent 2>/dev/null || true

echo "Installing new services..."
cp "$DIR/vybn-dual-spark.service" /etc/systemd/system/
cp "$DIR/vybn-breath.service"     /etc/systemd/system/
cp "$DIR/vybn-breath.timer"       /etc/systemd/system/

# Remove stale single-node service files
rm -f /etc/systemd/system/vybn-llama.service
rm -f /etc/systemd/system/vybn-heartbeat.service
rm -f /etc/systemd/system/vybn-heartbeat.timer
rm -f /etc/systemd/system/vllm.service
rm -f /etc/systemd/system/vybn-agent.service

systemctl daemon-reload

echo "Starting dual-Spark vLLM cluster..."
systemctl enable vybn-dual-spark
systemctl start vybn-dual-spark

echo "Waiting for vLLM API..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM API ready on both Sparks."
        break
    fi
    sleep 5
done

echo "Starting breath timer..."
systemctl enable vybn-breath.timer
systemctl start vybn-breath.timer

echo ""
echo "Done. Verify with:"
echo "  systemctl status vybn-dual-spark"
echo "  curl http://localhost:8000/v1/models"
echo "  systemctl list-timers | grep vybn"
