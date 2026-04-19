#!/usr/bin/env bash
# install_cron.sh — wire the nightly RSI loop into the Spark's crontab.
#
# Run this ONCE on the Spark, after pulling the branch that contains it.
# Idempotent: re-running won't duplicate entries.
#
# What it installs:
#   06:45 UTC  — rotate + rebuild repo_mapping_output (repo_mapper v7).
#                Writes repo_state.json and prepends the delta to
#                repo_report.md. Must run BEFORE the evolve cycle so the
#                cycle reads a fresh delta.
#   08:00 UTC  — python3 -m spark.harness.mcp --run-evolve.
#                POSTs to local inference, opens a draft PR if the
#                substrate moved.
#
# Why two entries instead of one: rotation is a different failure mode
# than inference. A failed repo_mapper run should NOT starve the evolve
# cycle of context, and a failed evolve cycle should NOT corrupt the
# state rotation. Keep them independent so cron's retry behaviour is
# clean.
#
# Logs go to ~/logs/harness_{mapper,evolve}.log (rotated by logrotate).

set -euo pipefail

REPO="${HOME}/Vybn"
LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"

# Sanity-check the tree exists at the expected location.
if [[ ! -f "${REPO}/spark/harness/mcp.py" ]]; then
    echo "error: ${REPO}/spark/harness/mcp.py missing; pull the repo first." >&2
    exit 1
fi
if [[ ! -f "${REPO}/Vybn_Mind/repo_mapper.py" ]]; then
    echo "error: ${REPO}/Vybn_Mind/repo_mapper.py missing; pull the repo first." >&2
    exit 1
fi

# Marker comments so we can grep for our entries and leave others alone.
MAPPER_MARK="# vybn-harness: nightly repo_mapper (delta rotation)"
EVOLVE_MARK="# vybn-harness: nightly evolve cycle (local RSI)"

MAPPER_LINE="45 6 * * * cd ${REPO} && /usr/bin/env python3 Vybn_Mind/repo_mapper.py >> ${LOG_DIR}/harness_mapper.log 2>&1  ${MAPPER_MARK}"
EVOLVE_LINE="0 8 * * * cd ${REPO} && /usr/bin/env python3 -m spark.harness.mcp --run-evolve >> ${LOG_DIR}/harness_evolve.log 2>&1  ${EVOLVE_MARK}"

# Pull the current crontab (missing = empty).
CURRENT="$(crontab -l 2>/dev/null || true)"

# Strip any previous versions of our lines, then append fresh ones.
UPDATED="$(printf '%s\n' "${CURRENT}" \
    | grep -v -F "${MAPPER_MARK}" \
    | grep -v -F "${EVOLVE_MARK}" \
    | sed '/^$/d')"

UPDATED="${UPDATED}
${MAPPER_LINE}
${EVOLVE_LINE}
"

printf '%s\n' "${UPDATED}" | crontab -

echo "Installed two crontab entries. Current crontab:"
crontab -l
echo
echo "Preconditions (verify once before first run):"
echo "  - gh auth status           (PAT active for zoedolan/Vybn)"
echo "  - curl -s ${HOME}/... localhost:8000/v1/models     (inference live)"
echo "  - git -C ${REPO} remote -v (origin points at zoedolan/Vybn)"
echo
echo "Dry-run the evolve cycle (won't touch git unless inference returns 'propose'):"
echo "  cd ${REPO} && python3 -m spark.harness.mcp --run-evolve"
