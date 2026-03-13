#!/usr/bin/env bash
# autonomous_cycle.sh — Cron-driven growth cycle check + execution
# Runs locally on the Spark. No API calls. No external dependencies.
#
# Cron entry (daily at 4am, after tidy):
#   0 4 * * * cd /home/vybnz69/Vybn && . ~/.vybn_keys && bash spark/growth/autonomous_cycle.sh >> /home/vybnz69/logs/growth.log 2>&1

set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG_PREFIX="[$TS] [growth]"

echo "$LOG_PREFIX checking trigger..."

# Check if growth cycle should fire (exit 1 = no)
if python3 -m spark.growth.trigger --check --memory-dir Vybn_Mind/memory 2>&1; then
    echo "$LOG_PREFIX trigger says: GO"
    
    # Run the cycle (not dry-run)
    echo "$LOG_PREFIX starting growth cycle..."
    python3 -m spark.growth.trigger --memory-dir Vybn_Mind/memory 2>&1
    EXIT=$?
    
    if [ $EXIT -eq 0 ]; then
        echo "$LOG_PREFIX growth cycle completed successfully"
        # Commit the updated continuity artifacts
        git add -A spark/growth/ Vybn_Mind/journal/spark/ 2>/dev/null || true
        git commit -m "Autonomous growth cycle $TS" --allow-empty 2>/dev/null || true
    else
        echo "$LOG_PREFIX growth cycle FAILED (exit $EXIT)"
    fi
else
    echo "$LOG_PREFIX trigger says: not yet (normal)"
fi
