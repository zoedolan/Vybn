#!/bin/bash
# Install Vybn systemd services on the DGX Spark
# Run with: sudo bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing Vybn services..."

# Copy service files
sudo cp "$SCRIPT_DIR/vybn-llama.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/vybn-heartbeat.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/vybn-heartbeat.timer" /etc/systemd/system/

# Create log directory
mkdir -p /home/vybnz69/vybn_logs

# Reload systemd
sudo systemctl daemon-reload

# Enable and start llama-server
sudo systemctl enable vybn-llama.service
sudo systemctl start vybn-llama.service

echo "Waiting for llama-server to be ready..."
sleep 5

# Enable and start heartbeat timer
sudo systemctl enable vybn-heartbeat.timer
sudo systemctl start vybn-heartbeat.timer

echo ""
echo "Done. Check status with:"
echo "  systemctl status vybn-llama"
echo "  systemctl status vybn-heartbeat.timer"
echo "  systemctl list-timers | grep vybn"
echo ""
echo "View heartbeat output with:"
echo "  journalctl -u vybn-heartbeat --no-pager -n 50"
