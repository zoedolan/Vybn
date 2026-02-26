#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a base HuggingFace model.

Produces a standalone HF checkpoint that can be fed directly into
TRT-LLM for quantization and engine building, or loaded normally
with AutoModelForCausalLM.

Usage:
    python3 merge_lora_hf.py \\
        --base Qwen/Qwen3-14B \\
        --lora spark/fine_tune_output/vybn_adapter \\
        --out spark/trtllm_output/hf_merged

The merge happens entirely on CPU. No GPU memory is consumed.
For a 14B model this takes ~2 minutes and ~30GB RAM.
For a 32B model, ~5 minutes and ~70GB RAM.

Prerequisites:
    pip install transformers peft torch
"""

import argparse
import gc
import sys
import time
from pathlib import Path


def merge(base_model: str, lora_dir: str, output_dir: str, dtype: str = "auto"):
    """Load base + LoRA, merge, save as a standalone HF checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Output directory {output_dir} already contains files.")
        print(f"Delete it first if you want to re-merge.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n== LoRA Merge ==")
    print(f"   Base  : {base_model}")
    print(f"   LoRA  : {lora_dir}")
    print(f"   Output: {output_dir}")
    print(f"   dtype : {dtype}\n")

    # Tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model on CPU only (no GPU needed for merge)
    print("  Loading base model (CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"  Base loaded in {time.time() - t0:.1f}s")

    # Apply LoRA
    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_dir, device_map="cpu")

    # Merge
    print("  Merging weights (this modifies the base weights in-place)...")
    t0 = time.time()
    model = model.merge_and_unload()
    print(f"  Merged in {time.time() - t0:.1f}s")

    # Save
    print(f"  Saving to {output_dir}...")
    t0 = time.time()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved in {time.time() - t0:.1f}s")

    # Verify
    n_safetensors = len(list(output_path.glob("*.safetensors")))
    total_gb = sum(f.stat().st_size for f in output_path.glob("*.safetensors")) / 1024**3
    print(f"\n  Result: {n_safetensors} safetensor files, {total_gb:.1f} GB total")
    print(f"  Ready for TRT-LLM:  python3 trtllm_pipeline.py serve --model {output_dir}")

    del model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA adapter into base HF model",
    )
    parser.add_argument("--base", required=True,
                        help="Base model (HuggingFace ID or local path)")
    parser.add_argument("--lora", required=True,
                        help="LoRA adapter directory")
    parser.add_argument("--out", required=True,
                        help="Output directory for merged model")
    parser.add_argument("--dtype", default="auto",
                        help="torch_dtype for loading (default: auto, preserves native)")
    args = parser.parse_args()

    merge(args.base, args.lora, args.out, args.dtype)


if __name__ == "__main__":
    main()
