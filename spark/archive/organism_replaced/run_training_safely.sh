#!/usr/bin/env bash
# run_training_safely.sh
# Ensures environment variables are perfectly set before launching
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source ~/.venv/spark/bin/activate || true

export DS_SKIP_CUDA_CHECK=1

echo "Diagnostic: Checking DeepSpeed AsyncIO..."
# We can't check .installed() because that attribute doesn't exist on the builder object 
# in this version of DeepSpeed. Instead we try to load it.
if python3 -c "from deepspeed.ops.aio import AsyncIOBuilder; b = AsyncIOBuilder(); print('Compatible:', b.is_compatible()); b.load(); print('AIO loaded successfully')" 2>&1 | grep -q "AIO loaded successfully"; then
    echo "AIO kernel is ALREADY working."
else
    echo "AIO kernel is MISSING or BROKEN. Rebuilding..."
    
    # 1. Install system dependency for AIO
    echo "Installing libaio-dev..."
    sudo apt-get update && sudo apt-get install -y libaio-dev

    # 2. Force reinstall DeepSpeed with AIO enabled
    # DS_BUILD_AIO=1 is the key flag.
    # DS_SKIP_CUDA_CHECK=1 is required because of the CUDA 13.0 vs PyTorch 12.8 mismatch.
    echo "Reinstalling DeepSpeed with AIO support..."
    DS_BUILD_AIO=1 DS_SKIP_CUDA_CHECK=1 pip install deepspeed --force-reinstall --no-cache-dir --no-build-isolation
    
    # 3. Verify again
    echo "Verifying AIO after rebuild..."
    python3 -c "from deepspeed.ops.aio import AsyncIOBuilder; b = AsyncIOBuilder(); b.load(); print('AIO loaded successfully')"
fi

echo "Starting training..."
# Ensure the fine_tune script inherits the environment
python3 spark/fine_tune_vybn.py
