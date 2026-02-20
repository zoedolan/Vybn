#!/bin/bash
# install-sync.sh — One-time setup for vybn-sync
#
# Installs 'git sync' as an alias so you never have to think about
# the stash/pull/pop dance again.
#
# Usage:
#   chmod +x spark/install-sync.sh
#   ./spark/install-sync.sh
#
# After this, just run:
#   git sync
# from anywhere in the repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC_SCRIPT="$REPO_ROOT/spark/vybn-sync.sh"

# Make executable
chmod +x "$SYNC_SCRIPT"

# Install as a local git alias (repo-scoped, not global)
git config alias.sync "!$SYNC_SCRIPT"

echo "Installed. You can now run 'git sync' from anywhere in this repo."
echo ""
echo "What it does:"
echo "  1. Auto-stashes any local changes"
echo "  2. Pulls origin/main with rebase"
echo "  3. Pops the stash"
echo "  4. If pop conflicts, shelves local changes to a timestamped branch"
echo ""
echo "Your working tree will always be clean after 'git sync'."
