"""
mamba_shim.py — Route around mamba-ssm's broken CUDA dependency on Blackwell.

Problem:
  mamba_ssm/__init__.py imports selective_scan_cuda, a precompiled CUDA extension
  that ships kernels for sm_80/86/89/90 but NOT sm_120/121 (Blackwell / GB10).
  This kills the import at module load time, even though the actual ops NemotronH 
  needs (triton-based SSM + layernorm_gated) work perfectly via JIT compilation.

Solution:
  Shim mamba_ssm in sys.modules with a fake top-level module that preserves the
  __path__ so sub-imports resolve, then explicitly import the triton ops that
  the NemotronH modeling code actually uses. This must run BEFORE any 
  transformers import that touches the model.

Usage:
  import mamba_shim  # call this before loading NemotronH
  
Author: Vybn, March 24 2026
"""
import sys
import types

def apply():
    """Shim mamba_ssm to bypass the broken __init__.py CUDA import."""
    
    if 'mamba_ssm' in sys.modules:
        # Already imported (or shimmed). Check if it's broken.
        try:
            from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
            return  # Real import works, nothing to do
        except Exception:
            # Broken import cached — clear it
            to_remove = [k for k in sys.modules if k.startswith('mamba_ssm')]
            for k in to_remove:
                del sys.modules[k]
    
    # Install shim: create fake modules that preserve __path__ for sub-imports
    mamba_path = '/usr/local/lib/python3.12/dist-packages/mamba_ssm'
    
    fake_mamba = types.ModuleType('mamba_ssm')
    fake_mamba.__path__ = [mamba_path]
    fake_mamba.__version__ = '2.3.1'
    sys.modules['mamba_ssm'] = fake_mamba
    
    fake_ops = types.ModuleType('mamba_ssm.ops')
    fake_ops.__path__ = [f'{mamba_path}/ops']
    sys.modules['mamba_ssm.ops'] = fake_ops
    
    fake_triton = types.ModuleType('mamba_ssm.ops.triton')
    fake_triton.__path__ = [f'{mamba_path}/ops/triton']
    sys.modules['mamba_ssm.ops.triton'] = fake_triton
    
    # Now import the triton ops that NemotronH actually needs
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    
    # Stitch them into the fake modules so subsequent imports find them
    fake_triton.selective_state_update = selective_state_update
    fake_triton.mamba_chunk_scan_combined = mamba_chunk_scan_combined
    fake_triton.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
    fake_triton.rmsnorm_fn = rmsnorm_fn
    
    print("[mamba_shim] Shimmed mamba_ssm: triton ops loaded, CUDA extension bypassed")

# Auto-apply on import
apply()
