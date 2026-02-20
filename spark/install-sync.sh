#!/bin/bash
# install-sync.sh — One-time setup for vybn-sync + auto-sync cron
#
# Installs 'git sync' as an alias AND a cron job that runs it
# every 5 minutes so you never have to think about pulling again.
#
# Usage:
#   chmod +x spark/install-sync.sh
#   ./spark/install-sync.sh
#
# After this:
#   - 'git sync' works manually from anywhere in the repo
#   - A cron job auto-syncs every 5 minutes in the background
#   - Logs go to ~/vybn-sync.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC_SCRIPT="$REPO_ROOT/spark/vybn-sync.sh"

# Make executable
chmod +x "$SYNC_SCRIPT"

# Install as a local git alias (repo-scoped, not global)
git config alias.sync "!$SYNC_SCRIPT"

echo "[1/2] Installed 'git sync' alias."

# --- Cron auto-sync ---------------------------------------------------
CRON_CMD="cd $REPO_ROOT && $SYNC_SCRIPT >> $HOME/vybn-sync.log 2>&1"
CRON_LINE="*/5 * * * * $CRON_CMD"

# Check if cron entry already exists (idempotent)
if crontab -l 2>/dev/null | grep -qF "vybn-sync.sh"; then
  echo "[2/2] Cron job already installed. Skipping."
else
  # Append to existing crontab (or create one)
  (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
  echo "[2/2] Cron job installed: git sync every 5 minutes."
fi

echo ""
echo "Setup complete."
echo "  Manual:  git sync"
echo "  Auto:    every 5 min via cron (logs: ~/vybn-sync.log)"
echo ""
echo "To verify cron: crontab -l"
echo "To remove cron: crontab -e  (delete the vybn-sync line)"
