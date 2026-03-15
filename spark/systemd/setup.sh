#!/bin/bash
# Install Vybn services on the DGX Spark.
# Run: sudo bash spark/systemd/setup.sh
#
# This replaces ALL previous service files with the ones that
# match the actual hardware (Nemotron-3-Super-120B, llama.cpp, port 8000).

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping old services..."
systemctl stop vybn-llama vybn-heartbeat.timer vybn-heartbeat vllm vybn-agent 2>/dev/null || true
systemctl disable vybn-llama vybn-heartbeat.timer vybn-heartbeat vllm vybn-agent 2>/dev/null || true

echo "Installing new services..."
cp "$DIR/vybn-llama.service"  /etc/systemd/system/
cp "$DIR/vybn-breath.service" /etc/systemd/system/
cp "$DIR/vybn-breath.timer"   /etc/systemd/system/

# Remove stale service files
rm -f /etc/systemd/system/vybn-heartbeat.service
rm -f /etc/systemd/system/vybn-heartbeat.timer
rm -f /etc/systemd/system/vllm.service
rm -f /etc/systemd/system/vybn-agent.service

systemctl daemon-reload

echo "Starting llama-server..."
systemctl enable vybn-llama
systemctl start vybn-llama

echo "Waiting for llama-server..."
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "llama-server ready."
        break
    fi
    sleep 5
done

echo "Starting breath timer..."
systemctl enable vybn-breath.timer
systemctl start vybn-breath.timer

echo ""
echo "Done. Verify with:"
echo "  systemctl status vybn-llama"
echo "  systemctl list-timers | grep vybn"
echo "  journalctl -u vybn-breath -f"