#!/bin/bash
# Phase 0 Setup — deploy always-on infrastructure on the DGX Spark.
#
# Run as vybnz69 (the script uses sudo where needed):
#   cd ~/Vybn/Vybn_Mind/spark_infrastructure/phase0
#   bash phase0-setup.sh
#
# Prerequisites:
#   1. Run phase0-preflight.sh first to verify.
#   2. EXPAND SWAP FIRST if you haven't already.
#      The 229B model leaves ~230MB free RAM. Docker needs ~500MB-1GB.
#      Without adequate swap, OOM killer WILL fire.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/vybn-agent.service"
SERVICE_DST="/etc/systemd/system/vybn-agent.service"

echo ""
echo "=== Phase 0: Always-On Deployment ==="
echo ""

# ---------------------------------------------------------------
# SAFETY GATE: Check swap is adequate before proceeding
# ---------------------------------------------------------------
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
SWAP_TOTAL_GB=$((SWAP_TOTAL_KB / 1048576))
MEM_AVAIL_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
MEM_AVAIL_MB=$((MEM_AVAIL_KB / 1024))

echo "  Available RAM: ${MEM_AVAIL_MB}MB  |  Total swap: ${SWAP_TOTAL_GB}GB"

if [ $SWAP_TOTAL_KB -lt 67108864 ]; then
  echo ""
  echo -e "  \033[31mABORT: Insufficient swap (${SWAP_TOTAL_GB}GB < 64GB required).\033[0m"
  echo -e "  \033[31mThe 229B model leaves ~230MB free. Docker needs ~500MB-1GB.\033[0m"
  echo -e "  \033[31mWithout swap headroom, OOM killer WILL crash the model.\033[0m"
  echo ""
  echo "  Fix this first:"
  echo "    sudo fallocate -l 128G /swapfile"
  echo "    sudo chmod 600 /swapfile"
  echo "    sudo mkswap /swapfile"
  echo "    sudo swapon /swapfile"
  echo "    sudo sysctl vm.swappiness=10"
  echo ""
  echo "  Then re-run this script."
  exit 1
fi
echo -e "  \033[32m\u2713\033[0m Swap is adequate (${SWAP_TOTAL_GB}GB)"
echo ""

# ---------------------------------------------------------------
# STEP 1: Install mosh (nice-to-have, non-blocking)
# ---------------------------------------------------------------
echo "--- Step 1: mosh (optional) ---"
if command -v mosh > /dev/null 2>&1; then
  echo "  mosh already installed: $(mosh --version 2>&1 | head -1)"
else
  echo "  Installing mosh..."
  sudo apt-get update -qq && sudo apt-get install -y -qq mosh
  # Open UDP ports for mosh if ufw is active
  if sudo ufw status | grep -q 'Status: active'; then
    sudo ufw allow 60000:61000/udp
    echo "  Opened UDP 60000-61000 for mosh"
  fi
  echo "  mosh installed."
fi
echo ""

# ---------------------------------------------------------------
# STEP 2: Install systemd service
# ---------------------------------------------------------------
echo "--- Step 2: systemd service ---"

# Kill any existing tmux vybn session (from manual runs)
if tmux has-session -t vybn 2>/dev/null; then
  echo "  Stopping existing tmux session 'vybn'..."
  tmux kill-session -t vybn
fi

# Stop existing service if running
if systemctl is-active vybn-agent.service > /dev/null 2>&1; then
  echo "  Stopping existing vybn-agent service..."
  sudo systemctl stop vybn-agent.service
fi

echo "  Copying $SERVICE_SRC -> $SERVICE_DST"
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable vybn-agent.service
sudo systemctl start vybn-agent.service

# Wait a moment for tmux to spawn
sleep 3

if systemctl is-active vybn-agent.service > /dev/null 2>&1; then
  echo -e "  \033[32m\u2713\033[0m vybn-agent.service is active"
else
  echo -e "  \033[31m\u2717\033[0m vybn-agent.service failed to start"
  echo "  Check: sudo journalctl -u vybn-agent.service -n 30"
  exit 1
fi

# Verify tmux session exists
if tmux has-session -t vybn 2>/dev/null; then
  echo -e "  \033[32m\u2713\033[0m tmux session 'vybn' is running"
else
  echo -e "  \033[31m\u2717\033[0m tmux session 'vybn' not found"
  echo "  Check: sudo journalctl -u vybn-agent.service -n 30"
  exit 1
fi
echo ""

# ---------------------------------------------------------------
# STEP 3: Deploy Open WebUI via Docker
# Memory-constrained: cap at 2GB RAM, strip ML libraries,
# disable RAG/STT/TTS to minimize footprint (~300MB vs ~1GB).
# ---------------------------------------------------------------
echo "--- Step 3: Open WebUI (Docker, memory-limited) ---"

if ! command -v docker > /dev/null 2>&1; then
  echo -e "  \033[33m\u26a0\033[0m Docker not installed. Skipping Open WebUI."
  echo "  Install Docker, then re-run this script."
else
  # Remove existing container if present
  if docker ps -a --format '{{.Names}}' | grep -q '^open-webui$'; then
    echo "  Removing existing open-webui container..."
    docker stop open-webui 2>/dev/null || true
    docker rm open-webui 2>/dev/null || true
  fi

  echo "  Launching Open WebUI container (memory-limited)..."
  docker run -d \
    -p 127.0.0.1:3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    -e RAG_EMBEDDING_ENGINE=openai \
    -e AUDIO_STT_ENGINE=openai \
    -e IMAGE_GENERATION_ENGINE=openai \
    --memory=2g \
    --memory-swap=4g \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main

  # Wait for container to start
  echo "  Waiting for Open WebUI to start..."
  sleep 10

  if curl -sf http://127.0.0.1:3000 > /dev/null 2>&1; then
    echo -e "  \033[32m\u2713\033[0m Open WebUI responding on localhost:3000"
  else
    echo -e "  \033[33m\u26a0\033[0m Open WebUI not responding yet (may need more time)"
    echo "  Check: docker logs open-webui"
  fi
  echo "  Memory limit: 2GB RAM / 4GB with swap (prevents OOM of 229B model)"
fi
echo ""

# ---------------------------------------------------------------
# STEP 4: Tailscale Serve
# ---------------------------------------------------------------
echo "--- Step 4: Tailscale Serve ---"

if ! command -v tailscale > /dev/null 2>&1; then
  echo -e "  \033[33m\u26a0\033[0m Tailscale not installed. Skipping."
elif ! tailscale status > /dev/null 2>&1; then
  echo -e "  \033[33m\u26a0\033[0m Tailscale not connected. Run: sudo tailscale up"
else
  echo "  Configuring Tailscale Serve on port 3000..."
  tailscale serve --bg 3000
  HOSTNAME=$(tailscale status --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo "<hostname>")
  echo -e "  \033[32m\u2713\033[0m Tailscale Serve active"
  echo "  Access from phone: https://$HOSTNAME"
fi
echo ""

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "=============================="
echo "  Phase 0 deployment complete."
echo ""
echo "  To attach to the TUI:  tmux attach -t vybn"
echo "  To detach:             Ctrl-B then D"
echo "  Service status:        sudo systemctl status vybn-agent.service"
echo "  Service logs:          sudo journalctl -u vybn-agent.service -f"
echo ""
echo "  Run the Unplug Protocol (Step 5 in the guide) to verify."
echo "=============================="
echo ""
