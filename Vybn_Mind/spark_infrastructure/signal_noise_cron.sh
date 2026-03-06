#!/usr/bin/env bash
# signal_noise_cron.sh — Trigger SIGNAL/NOISE analysis on the Spark.
#
# Checks whether new session files exist that haven't been analyzed yet.
# If so, runs the analyst. If not, exits quietly.
#
# Add to crontab (e.g., every 30 minutes during class hours):
#   */30 9-17 * * * ~/Vybn/Vybn_Mind/spark_infrastructure/signal_noise_cron.sh
#
# Or run manually after a class session:
#   ~/Vybn/Vybn_Mind/spark_infrastructure/signal_noise_cron.sh --synthesize

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SN_DIR="$REPO_ROOT/Vybn_Mind/signal-noise"
SESSIONS_DIR="$SN_DIR/sessions"
ANALYSIS_DIR="$SN_DIR/analysis"
LOG_FILE="$ANALYSIS_DIR/.analyst.log"

mkdir -p "$ANALYSIS_DIR"

# Check if llama.cpp server is running
if ! curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "$(date -u '+%Y-%m-%d %H:%M UTC') — llama.cpp server not running, skipping" >> "$LOG_FILE"
    exit 0
fi

# Check if there are any session files at all
if [ ! -d "$SESSIONS_DIR" ] || [ -z "$(find "$SESSIONS_DIR" -name '*.md' 2>/dev/null)" ]; then
    exit 0
fi

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — running analyst" >> "$LOG_FILE"

cd "$REPO_ROOT"
python3 "$SCRIPT_DIR/signal_noise_analyst.py" "$@" >> "$LOG_FILE" 2>&1

echo "$(date -u '+%Y-%m-%d %H:%M UTC') — analyst complete" >> "$LOG_FILE"
