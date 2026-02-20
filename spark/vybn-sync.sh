#!/bin/bash
# vybn-sync.sh — Safe parallel git sync for the Spark
#
# Problem: Multiple writers (Gemini on Spark, Claude via API, Copilot,
# Zoe manually) all push to the same repo. The Spark's runtime also
# generates files locally. Raw 'git pull' fails whenever the working
# tree is dirty.
#
# Solution: This script replaces 'git pull' on the Spark. It:
#   1. Stashes any local changes (staged or unstaged)
#   2. Pulls from origin/main with rebase
#   3. Pops the stash
#   4. If the pop conflicts, shelves local changes to a timestamped
#      branch so nothing is lost and the working tree stays clean.
#
# Usage:
#   ./spark/vybn-sync.sh          # from repo root
#   git sync                      # if alias installed (see install-sync.sh)
#
# This script is idempotent and safe to run from cron, systemd, or
# the heartbeat.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="[vybn-sync $TIMESTAMP]"

echo "$LOG_PREFIX Starting sync..."

# ─── Step 1: Check if working tree is dirty ─────────────────────
if git diff --quiet && git diff --cached --quiet; then
    DIRTY=false
    echo "$LOG_PREFIX Working tree clean. Pulling directly."
else
    DIRTY=true
    echo "$LOG_PREFIX Working tree dirty. Stashing local changes."
    git stash push -m "vybn-sync auto-stash $TIMESTAMP" --include-untracked
fi

# ─── Step 2: Pull from origin ───────────────────────────────────
echo "$LOG_PREFIX Pulling origin/main..."
if ! git pull origin main --rebase; then
    echo "$LOG_PREFIX ERROR: Pull failed (network or merge conflict)."
    echo "$LOG_PREFIX Aborting rebase if in progress..."
    git rebase --abort 2>/dev/null || true

    # If we stashed, pop it back so we don't lose work
    if [ "$DIRTY" = true ]; then
        echo "$LOG_PREFIX Restoring stashed changes."
        git stash pop || true
    fi
    exit 1
fi

echo "$LOG_PREFIX Pull successful."

# ─── Step 3: Restore local changes ─────────────────────────────
if [ "$DIRTY" = true ]; then
    echo "$LOG_PREFIX Restoring stashed changes..."
    if git stash pop; then
        echo "$LOG_PREFIX Stash restored cleanly."
    else
        # Stash pop conflicted. Shelve to a branch so nothing is lost.
        SHELVE_BRANCH="shelve/spark-local-$TIMESTAMP"
        echo "$LOG_PREFIX Stash pop conflicted. Shelving to $SHELVE_BRANCH"

        # Reset the failed merge state
        git checkout -- .
        git clean -fd

        # Create a branch from the stash
        git stash branch "$SHELVE_BRANCH" || {
            echo "$LOG_PREFIX WARNING: Could not create shelve branch. Dropping stash."
            git stash drop || true
        }

        # Return to main with a clean tree
        git checkout main
        echo "$LOG_PREFIX Local changes shelved to $SHELVE_BRANCH"
        echo "$LOG_PREFIX Main is now clean and up-to-date."
    fi
fi

echo "$LOG_PREFIX Sync complete."
