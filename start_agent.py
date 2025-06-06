#!/usr/bin/env bash
set -xeuo pipefail

# 1) Activate the environment that’s already been bootstrapped
source .venv/bin/activate

# 2) (Optional) export any env vars your agent needs
export MIND_VIZ_DIR="${PWD}/Mind Visualization"

# 3) Launch the repo's primary entry script. Update ENTRY_SCRIPT if this changes.
ENTRY_SCRIPT="introspect_repo.py"

if [ ! -f "$ENTRY_SCRIPT" ]; then
    echo "Error: $ENTRY_SCRIPT not found" >&2
    exit 1
fi

python "$ENTRY_SCRIPT" "$@"
