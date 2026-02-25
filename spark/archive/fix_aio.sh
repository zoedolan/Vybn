#!/usr/bin/env bash
# fix_aio.sh — One command to fix NVMe offload and start training
#
# The problem: DeepSpeed's async_io C++ extension was never compiled
# because libaio-dev wasn't installed when DeepSpeed was pip-installed.
# Without AIO, NVMe offload silently fails — offload_cache stays at 4K,
# the 228B model has nowhere to go, and training either hangs or OOMs.
#
# This script installs libaio, rebuilds DeepSpeed from source with AIO,
# verifies it works, and runs training. One command.
#
# Usage:
#   bash spark/fix_aio.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================"
echo " Fixing DeepSpeed NVMe offload (async_io)"
echo "============================================"
echo ""

# -----------------------------------------------------------
# 1. Install libaio-dev (kernel AIO library for DMA writes)
# -----------------------------------------------------------
echo "[1/5] Installing libaio-dev..."
if dpkg -l libaio-dev 2>/dev/null | grep -q "^ii"; then
    echo "      Already installed."
else
    sudo apt-get update -qq
    sudo apt-get install -y libaio-dev
    echo "      Installed."
fi
echo ""

# -----------------------------------------------------------
# 2. Activate the Spark Python environment
# -----------------------------------------------------------
echo "[2/5] Activating Python environment..."
source ~/.venv/spark/bin/activate
DS_VERSION=$(python3 -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null || echo 'not found')
echo "      Python:    $(which python3)"
echo "      DeepSpeed: $DS_VERSION"
echo ""

# -----------------------------------------------------------
# 3. Rebuild DeepSpeed from source with AIO compiled
#    --no-binary forces source build (wheels skip compilation)
#    --no-cache-dir prevents stale cached builds
# -----------------------------------------------------------
echo "[3/5] Rebuilding DeepSpeed with async_io..."
echo "      (Compiling C++ extensions — takes 2-5 minutes)"
echo ""
DS_BUILD_AIO=1 pip install deepspeed --force-reinstall --no-cache-dir --no-binary deepspeed 2>&1 | \
    grep -E '(Building|Installing|Successfully|ERROR|async_io)' || true
echo ""

# -----------------------------------------------------------
# 4. Verify AIO is now available
# -----------------------------------------------------------
echo "[4/5] Verifying async_io..."
AIO_OK=$(python3 -c "
try:
    from deepspeed.ops.op_builder import AsyncIOBuilder
    b = AsyncIOBuilder()
    print('yes' if b.is_compatible() else 'no')
except Exception as e:
    print('no')
" 2>/dev/null)

if [ "$AIO_OK" = "yes" ]; then
    echo "      async_io: AVAILABLE — NVMe offload will work"
else
    echo "      async_io: STILL NOT AVAILABLE"
    echo ""
    echo "      Trying alternative build method..."
    pip install deepspeed --global-option="build_ext" 2>&1 | tail -3 || true
    # Recheck
    AIO_OK2=$(python3 -c "
try:
    from deepspeed.ops.op_builder import AsyncIOBuilder
    print('yes' if AsyncIOBuilder().is_compatible() else 'no')
except:
    print('no')
" 2>/dev/null)
    if [ "$AIO_OK2" = "yes" ]; then
        echo "      async_io: AVAILABLE after retry"
    else
        echo "      FAILED. Check build output above."
        echo "      Manual fix: DS_BUILD_AIO=1 pip install deepspeed --no-binary :all:"
        exit 1
    fi
fi
echo ""

# -----------------------------------------------------------
# 5. Clear stale state and run training
# -----------------------------------------------------------
echo "[5/5] Starting fine-tuning..."
rm -rf spark/offload_cache ~/.cache/torch_extensions
mkdir -p spark/offload_cache
echo ""
exec python3 spark/fine_tune_vybn.py
