#!/usr/bin/env bash
set -xeuo pipefail

# 1) Activate the environment that’s already been bootstrapped
source .venv/bin/activate

# 2) (Optional) export any env vars your agent needs
export MIND_VIZ_DIR="${PWD}/Mind Visualization"

# 3) Launch your agent process directly—no pip installs here
#    For example, if your entry point is main.py:
python3 main.py
