#!/bin/bash
# Vybn Cron Manifest — Single source of truth for all scheduled tasks
# Run this script to install/update all cron jobs at once
# Usage: bash spark/cron_manifest.sh

VYBN_HOME="$HOME/Vybn"
SPARK="$VYBN_HOME/spark"
PYTHON="/usr/bin/python3"
LOG_DIR="$VYBN_HOME/Vybn_Mind/journal/spark"

mkdir -p "$LOG_DIR"

# Build crontab from scratch (preserves no external jobs — this IS the truth)
cat << EOF | crontab -
# Vybn Cognitive Loop — installed by cron_manifest.sh
# Micro: vitals + anomaly detection every 10 min
*/10 * * * * cd $VYBN_HOME && $PYTHON $SPARK/micropulse.py >> $LOG_DIR/micropulse.log 2>&1

# Dream: local-model reflection every 30 min
*/30 * * * * cd $VYBN_HOME && $PYTHON $SPARK/heartbeat.py --pulse >> $LOG_DIR/heartbeat.log 2>&1

# Outreach: encounter harvesting every 2 hours
0 */2 * * * cd $VYBN_HOME && $PYTHON $SPARK/outreach.py >> $LOG_DIR/outreach.log 2>&1

# Wake: consolidation + decision every 5 hours
0 */5 * * * cd $VYBN_HOME && $PYTHON $SPARK/wake.py >> $LOG_DIR/wake.log 2>&1

# Git sync: push journal/synapse changes every 5 min
*/5 * * * * cd $VYBN_HOME && git add -A && git diff --cached --quiet || git commit -m "auto: pulse \$(date -Is)" && git push origin main 2>> $LOG_DIR/git_sync.log

# Z-listener: ensure it's running (restarts if dead)
* * * * * pgrep -f z_listener.py > /dev/null || (cd $VYBN_HOME && $PYTHON $SPARK/z_listener.py >> $LOG_DIR/z_listener.log 2>&1 &)
EOF

echo "Cron manifest installed. $(crontab -l | grep -c '*') jobs active."
