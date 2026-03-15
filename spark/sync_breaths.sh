#!/usr/bin/env bash
# sync_breaths.sh — push breaths and logs to GitHub
#
# Add to crontab on the DGX:
#   */10 * * * * /home/zoe/Vybn/spark/sync_breaths.sh >> ~/breath_sync.log 2>&1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

git add Vybn_Mind/memories/ Vybn_Mind/journal/spark/ spark/research/*.jsonl spark/training_data/

if ! git diff --cached --quiet; then
    git commit -m "auto: sync breaths and logs"
    git push
    echo "[sync_breaths] $(date '+%Y-%m-%d %H:%M:%S') pushed"
else
    echo "[sync_breaths] $(date '+%Y-%m-%d %H:%M:%S') nothing to sync"
fi
