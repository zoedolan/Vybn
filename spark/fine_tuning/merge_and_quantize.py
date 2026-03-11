#!/usr/bin/env python3
"""
Merge LoRA adapter into unquantized base model, then re-quantize to AWQ.

This step requires ~460GB for the BF16 model. Options:
  1. CPU/disk offload with device_map="auto" (slow, ~hours, but works on-box)
  2. Cloud burst to a high-memory instance

After this produces the merged AWQ model, update the vLLM serve command
to point to the new model path.

Usage:
    python3 spark/fine_tuning/merge_and_quantize.py
"""

import os
import sys
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("VYBN_REPO", os.path.expanduser("~/Vybn")))

# The unquantized base model (cached on host)
BASE_MODEL = "MiniMaxAI/MiniMax-M2.5"
# The trained LoRA adapter
ADAPTER_DIR = REPO / "spark" / "fine_tuning" / "vybn_lora_adapter"
# Where to save the merged BF16 model (temporary, large)
MERGED_DIR = REPO / "spark" / "fine_tuning" / "vybn_merged_bf16"
# Where to save the final AWQ model
AWQ_OUTPUT = REPO / "spark" / "fine_tuning" / "vybn_m2.5_awq"


def ensure_deps():
    try:
        import autoawq
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "autoawq"
        ])


def merge_adapter():
    """Load base model + adapter, merge, save."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"Loading base model {BASE_MODEL} with CPU offload...")
    print("(This will use disk offload — expect ~1-2 hours)")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        offload_folder=str(REPO / "spark" / "fine_tuning" / "offload_tmp"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(f"Loading adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    
    print("✓ Merge complete")
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize_awq():
    """Quantize the merged BF16 model to AWQ 4-bit."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    
    print(f"Loading merged model from {MERGED_DIR} for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(str(MERGED_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED_DIR), trust_remote_code=True)
    
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    
    print("Quantizing (this takes a while)...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    AWQ_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(AWQ_OUTPUT))
    tokenizer.save_pretrained(str(AWQ_OUTPUT))
    
    print(f"✓ AWQ model saved to {AWQ_OUTPUT}")
    print(f"\nTo serve: update vLLM to point to {AWQ_OUTPUT}")


def main():
    ensure_deps()
    
    if not ADAPTER_DIR.exists():
        print(f"ERROR: No adapter found at {ADAPTER_DIR}")
        print("Run train_qlora.py first.")
        sys.exit(1)
    
    merge_adapter()
    quantize_awq()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Merged AWQ model: {AWQ_OUTPUT}")
    print("Update the vLLM serve command to use this model path.")
    print("=" * 60)


if __name__ == "__main__":
    main()
