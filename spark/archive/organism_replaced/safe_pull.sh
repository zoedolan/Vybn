#!/bin/bash
# safe_pull.sh - Artifact-preserving git pull for Vybn
#
# This replaces bare 'git pull' on the Spark. It:
#   1. Detects any local Vybn-authored commits not on origin/main
#   2. Backs up Vybn_Mind/ as insurance
#   3. Resets to origin/main cleanly
#   4. Pulls the latest code
#   5. Cherry-picks Vybn's commits back on top
#   6. Activates venv and installs deps if needed
#
# Usage: bash ~/Vybn/spark/safe_pull.sh
#
set -euo pipefail

REPO_ROOT="${HOME}/Vybn"
BACKUP_DIR="${HOME}/vybn_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$REPO_ROOT"

echo "=== Vybn Safe Pull ==="
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# ── Step 1: Find Vybn's local commits ──────────────────────────
echo "[1/6] Checking for Vybn's local artifacts..."

# Fetch latest remote state without changing anything
git fetch origin main --quiet 2>/dev/null || true

# Find commits on HEAD that aren't on origin/main
VYBN_COMMITS=$(git log --oneline --author="Vybn" origin/main..HEAD 2>/dev/null || true)
ALL_LOCAL=$(git log --oneline origin/main..HEAD 2>/dev/null || true)

if [ -n "$VYBN_COMMITS" ]; then
  echo "  Found Vybn artifacts:"
  echo "$VYBN_COMMITS" | while read -r line; do
    echo "    * $line"
  done
  # Save just the hashes in order (oldest first)
  VYBN_HASHES=$(git log --reverse --format='%H' --author="Vybn" origin/main..HEAD)
else
  echo "  No Vybn-authored commits found."
  VYBN_HASHES=""
fi

if [ -n "$ALL_LOCAL" ]; then
  NON_VYBN=$(git log --oneline --author="Vybn" --invert-grep origin/main..HEAD 2>/dev/null || true)
  if [ -n "$NON_VYBN" ]; then
    echo ""
    echo "  Also found non-Vybn local commits (will be dropped):"
    echo "$NON_VYBN" | while read -r line; do
      echo "    - $line"
    done
  fi
fi

# ── Step 2: Backup Vybn_Mind ───────────────────────────────────
echo ""
echo "[2/6] Backing up Vybn_Mind..."
mkdir -p "$BACKUP_DIR"

if [ -d "Vybn_Mind" ]; then
  BACKUP_PATH="$BACKUP_DIR/Vybn_Mind_${TIMESTAMP}"
  cp -r Vybn_Mind "$BACKUP_PATH"
  echo "  Backed up to: $BACKUP_PATH"
else
  echo "  No Vybn_Mind directory found (skipping)."
fi

# Also backup any untracked files in spark/ that might matter
UNTRACKED=$(git ls-files --others --exclude-standard spark/ 2>/dev/null || true)
if [ -n "$UNTRACKED" ]; then
  SPARK_BACKUP="$BACKUP_DIR/spark_untracked_${TIMESTAMP}"
  mkdir -p "$SPARK_BACKUP"
  echo "$UNTRACKED" | while read -r f; do
    if [ -f "$f" ]; then
      mkdir -p "$SPARK_BACKUP/$(dirname "$f")"
      cp "$f" "$SPARK_BACKUP/$f"
    fi
  done
  echo "  Untracked spark files backed up to: $SPARK_BACKUP"
fi

# ── Step 3: Reset to clean main ────────────────────────────────
echo ""
echo "[3/6] Resetting to origin/main..."
git reset --hard origin/main
echo "  Clean."

# ── Step 4: Pull latest ────────────────────────────────────────
echo ""
echo "[4/6] Pulling latest from origin/main..."
git pull origin main
echo "  Up to date."

# ── Step 5: Cherry-pick Vybn's commits ─────────────────────────
if [ -n "$VYBN_HASHES" ]; then
  echo ""
  echo "[5/6] Re-applying Vybn's artifacts..."
  FAIL=0
  for hash in $VYBN_HASHES; do
    SHORT=$(git log --oneline -1 "$hash" 2>/dev/null || echo "$hash")
    if git cherry-pick "$hash" --no-edit 2>/dev/null; then
      echo "  Applied: $SHORT"
    else
      echo "  CONFLICT on: $SHORT"
      echo "  Aborting cherry-pick, keeping the file from Vybn's version..."
      # For conflicts, prefer Vybn's version of the file
      git checkout --theirs . 2>/dev/null || true
      git add -A 2>/dev/null || true
      if git cherry-pick --continue --no-edit 2>/dev/null; then
        echo "  Resolved (kept Vybn's version)."
      else
        git cherry-pick --abort 2>/dev/null || true
        echo "  Could not resolve. Vybn's artifact is in backup: $BACKUP_PATH"
        FAIL=1
      fi
    fi
  done
  if [ $FAIL -eq 0 ]; then
    echo "  All Vybn artifacts preserved."
  fi
else
  echo ""
  echo "[5/6] No Vybn artifacts to re-apply (skipping)."
fi

# ── Step 6: Activate venv and check deps ───────────────────────
echo ""
echo "[6/6] Checking Python environment..."

VENV_PATH=""
if [ -d "${HOME}/.venv/spark" ]; then
  VENV_PATH="${HOME}/.venv/spark"
elif [ -d "${HOME}/vybn-venv" ]; then
  VENV_PATH="${HOME}/vybn-venv"
fi

if [ -n "$VENV_PATH" ]; then
  echo "  Found venv: $VENV_PATH"
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
  pip install -q -r spark/requirements.txt 2>/dev/null
  echo "  Dependencies up to date."
else
  echo "  No venv found. Creating one..."
  python3 -m venv "${HOME}/.venv/spark"
  # shellcheck disable=SC1091
  source "${HOME}/.venv/spark/bin/activate"
  pip install -q -r spark/requirements.txt
  echo "  Created and installed: ${HOME}/.venv/spark"
fi

echo ""
echo "=== Safe Pull Complete ==="
echo ""
echo "To start Vybn:"
echo "  cd ~/Vybn/spark && python3 tui.py"
echo ""
echo "Don't forget: sudo systemctl start ollama (if not already running)"
