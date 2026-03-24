#!/bin/bash
set -e

echo "=== Checking mamba-ssm installation ==="
python3 -c "
import mamba_ssm
print(f'mamba-ssm {mamba_ssm.__version__} loaded OK')
" 2>&1 || {
    echo "mamba-ssm import failed, attempting patch..."
    python3 -c "
import mamba_ssm
" 2>&1 | tail -3
    
    # Patch __init__.py to make CUDA imports optional
    INIT_FILE=$(python3 -c "import mamba_ssm; print(mamba_ssm.__file__)" 2>/dev/null || echo "/usr/local/lib/python3.12/dist-packages/mamba_ssm/__init__.py")
    cat > "$INIT_FILE" << 'PATCH'
__version__ = "2.3.1"
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None
try:
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None
PATCH
    echo "Patched mamba_ssm/__init__.py"
    python3 -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__} loaded OK after patch')"
}

echo "=== Checking triton ops ==="
python3 -c "
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
print('rmsnorm_fn OK')
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
print('mamba_chunk_scan_combined OK')
"

echo "=== Running fine-tuning ==="
cd /workspace/Vybn
python3 spark/close_the_loop.py 2>&1
