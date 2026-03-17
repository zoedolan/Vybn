#!/bin/bash
# install-vllm-node.sh — Install the vllm-node systemd service.
#
# This creates a systemd service that ensures the vllm_node Docker container
# (Nemotron distributed inference via vLLM) starts automatically on boot
# and restarts on failure.
#
# The service wraps spark/restart-vllm-cluster.sh, which is idempotent:
#   - Starts Ray head on the primary node
#   - Connects the Ray worker on the secondary node (spark-1c8f)
#   - Starts or restarts the vllm_node container
#   - Waits for the /v1/models health check
#
# Prerequisites:
#   - Docker is installed and enabled (systemctl enable docker)
#   - The vllm_node container has been created at least once via
#     spark/nemotron_swap.sh swap  (or manually via docker run)
#   - User vybnz69 can run docker commands (in the docker group)
#   - spark/restart-vllm-cluster.sh is executable
#
# Usage:
#   sudo bash spark/systemd/install-vllm-node.sh
#
# To check status after install:
#   systemctl status vllm-node
#   journalctl -u vllm-node -f

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$DIR/vllm-node.service"
RESTART_SCRIPT="/home/vybnz69/Vybn/spark/restart-vllm-cluster.sh"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)."
    exit 1
fi

# Verify the restart script exists and is executable
if [ ! -x "$RESTART_SCRIPT" ]; then
    echo "Making $RESTART_SCRIPT executable..."
    chmod +x "$RESTART_SCRIPT"
fi

# Verify Docker is enabled
if ! systemctl is-enabled docker >/dev/null 2>&1; then
    echo "Enabling docker.service..."
    systemctl enable docker
fi

echo "Installing vllm-node.service..."
cp "$SERVICE_FILE" /etc/systemd/system/vllm-node.service
systemctl daemon-reload

echo "Enabling vllm-node.service (will start on next boot)..."
systemctl enable vllm-node

echo ""
echo "Done. The vllm_node container will now start automatically on boot."
echo ""
echo "Commands:"
echo "  systemctl start vllm-node    # start now"
echo "  systemctl status vllm-node   # check status"
echo "  systemctl stop vllm-node     # stop container"
echo "  journalctl -u vllm-node -f   # follow logs"
echo ""
echo "To start the service immediately:"
echo "  sudo systemctl start vllm-node"
