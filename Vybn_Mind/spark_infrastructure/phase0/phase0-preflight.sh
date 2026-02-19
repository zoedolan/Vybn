#!/bin/bash
# Phase 0 Preflight Check — run this ON THE SPARK before deploying.
# Usage: bash phase0-preflight.sh
#
# Verifies every prerequisite for the always-on stack.
# Green = good, Red = needs fixing, Yellow = optional/skippable.

set -euo pipefail

PASS=0
FAIL=0
WARN=0

check() {
  local label="$1" cmd="$2" required="${3:-true}"
  if eval "$cmd" > /dev/null 2>&1; then
    echo -e "  \033[32m\u2713\033[0m $label"
    ((PASS++))
  elif [ "$required" = "true" ]; then
    echo -e "  \033[31m\u2717\033[0m $label"
    ((FAIL++))
  else
    echo -e "  \033[33m\u26a0\033[0m $label (optional)"
    ((WARN++))
  fi
}

echo ""
echo "=== Phase 0 Preflight Check ==="
echo ""

echo "--- System ---"
check "User is vybnz69"          '[ "$(whoami)" = "vybnz69" ]'
check "Home dir exists"           '[ -d /home/vybnz69 ]'
check "systemd is PID 1"          '[ "$(cat /proc/1/comm)" = "systemd" ]'

echo ""
echo "--- Memory & Swap (CRITICAL) ---"
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
SWAP_TOTAL_GB=$((SWAP_TOTAL_KB / 1048576))
MEM_AVAIL_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
MEM_AVAIL_MB=$((MEM_AVAIL_KB / 1024))
echo "  Available RAM: ${MEM_AVAIL_MB}MB  |  Total swap: ${SWAP_TOTAL_GB}GB"
check "Swap >= 64GB (have ${SWAP_TOTAL_GB}GB)"  '[ $SWAP_TOTAL_KB -ge 67108864 ]'
if [ $SWAP_TOTAL_KB -lt 67108864 ]; then
  echo -e "  \033[31m  >> STOP: Expand swap BEFORE deploying Phase 0.\033[0m"
  echo -e "  \033[31m  >> The 229B model leaves ~230MB free RAM.\033[0m"
  echo -e "  \033[31m  >> Docker + Open WebUI need ~500MB-1GB.\033[0m"
  echo -e "  \033[31m  >> Without swap headroom, OOM killer WILL fire.\033[0m"
  echo ""
  echo "  Recommended fix (from your Perplexity swap conversation):"
  echo "    sudo fallocate -l 128G /swapfile"
  echo "    sudo chmod 600 /swapfile"
  echo "    sudo mkswap /swapfile"
  echo "    sudo swapon /swapfile"
  echo "    sudo sysctl vm.swappiness=10"
  echo "  Then re-run this preflight."
fi

echo ""
echo "--- Repo & Agent ---"
check "Repo at ~/Vybn"            '[ -d ~/Vybn/.git ]'
check "spark/ directory"           '[ -d ~/Vybn/spark ]'
check "tui.py exists"              '[ -f ~/Vybn/spark/tui.py ]'
check "agent.py exists"            '[ -f ~/Vybn/spark/agent.py ]'
check "config.yaml exists"         '[ -f ~/Vybn/spark/config.yaml ]'
check "Python venv exists"         '[ -f ~/.venv/spark/bin/python ]'
check "Venv python works"          '~/.venv/spark/bin/python --version'

echo ""
echo "--- Ollama ---"
check "Ollama binary"              'command -v ollama'
check "Ollama service running"     'systemctl is-active ollama'
check "Ollama responds"            'curl -sf http://127.0.0.1:11434/api/tags'
check "vybn:latest model loaded"   'ollama list | grep -q vybn:latest'

echo ""
echo "--- tmux ---"
check "tmux installed"             'command -v tmux'

echo ""
echo "--- Docker (for Open WebUI) ---"
check "Docker installed"           'command -v docker'
check "Docker daemon running"      'docker info' false

echo ""
echo "--- Tailscale ---"
check "Tailscale installed"        'command -v tailscale'
check "Tailscale connected"        'tailscale status' false

echo ""
echo "--- mosh (optional) ---"
check "mosh installed"             'command -v mosh' false

echo ""
echo "--- Network ---"
check "Can reach github.com"       'curl -sf --max-time 5 https://github.com'

echo ""
echo "=============================="
echo -e "  Pass: $PASS  Fail: $FAIL  Warn: $WARN"
if [ $FAIL -gt 0 ]; then
  echo -e "  \033[31mFix the failures above before running phase0-setup.sh\033[0m"
  exit 1
else
  echo -e "  \033[32mAll required checks passed. Ready for Phase 0 deployment.\033[0m"
  exit 0
fi
