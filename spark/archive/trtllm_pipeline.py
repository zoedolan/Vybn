#!/usr/bin/env python3
"""Vybn inference on DGX Spark via TensorRT-LLM.

Replaces the Ollama stack with NVIDIA's native TRT-LLM, purpose-built
for the Blackwell GPU in DGX Spark.  Key advantages:

  - NVFP4 quantization: ~40% smaller than FP8, models that were 220GB
    at FP8 drop to ~60-70GB and fit in 128GB unified memory with room
    for KV cache and speculative decoding buffers.
  - Blackwell-optimized CUDA kernels (no more "capability 12.1 unsupported").
  - EAGLE-3 speculative decoding on supported models (2-3x throughput).
  - OpenAI-compatible API endpoint -- agent.py can talk to it directly.

Quick start (serve a validated model immediately):

    # Install TRT-LLM in your venv first:
    pip install tensorrt-llm

    # Serve Qwen3-14B at NVFP4 (fits easily in 128GB):
    python3 trtllm_pipeline.py serve --preset qwen3-14b

    # Serve GPT-OSS-120B with EAGLE-3 speculative decoding:
    python3 trtllm_pipeline.py serve --preset gpt-oss-120b

    # List all validated presets:
    python3 trtllm_pipeline.py list

Full Vybn pipeline (fine-tune -> merge -> quantize -> serve):

    python3 trtllm_pipeline.py full \\
        --hf-base Qwen/Qwen3-14B \\
        --lora-dir spark/fine_tune_output/vybn_adapter \\
        --serve

Connect agent.py to TRT-LLM:

    export VYBN_OPENAI_BASE_URL=http://localhost:8000/v1
    export VYBN_OPENAI_MODEL=vybn-local
    python3 spark/agent.py

Prerequisites:
    pip install tensorrt-llm peft transformers torch
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPARK_DIR = REPO_ROOT / "spark"
OUTPUT_ROOT = SPARK_DIR / "trtllm_output"

# ---------------------------------------------------------------------------
# Validated single-Spark presets (128GB unified memory)
#
# Memory estimates are post-quantization engine size.  Actual runtime usage
# includes KV cache, activations, and CUDA context (~2-4GB overhead).
# ---------------------------------------------------------------------------
PRESETS = {
    "qwen3-14b": {
        "hf_model": "Qwen/Qwen3-14B",
        "quantization": "nvfp4",
        "max_batch_size": 8,
        "max_input_len": 4096,
        "max_output_len": 4096,
        "engine_gb": 12,
        "description": "Fast, capable. Best quality-to-speed ratio on single Spark.",
    },
    "qwen3-32b": {
        "hf_model": "Qwen/Qwen3-32B",
        "quantization": "nvfp4",
        "max_batch_size": 4,
        "max_input_len": 4096,
        "max_output_len": 4096,
        "engine_gb": 22,
        "description": "Stronger reasoning. Still fits comfortably at NVFP4.",
    },
    "llama-3.3-70b": {
        "hf_model": "nvidia/Llama-3.3-70B-Instruct-NVFP4",
        "quantization": "nvfp4",
        "max_batch_size": 2,
        "max_input_len": 4096,
        "max_output_len": 4096,
        "engine_gb": 42,
        "pre_quantized": True,
        "description": "Pre-quantized by NVIDIA. ~5 tok/s on single Spark.",
    },
    "gpt-oss-120b": {
        "hf_model": "nvidia/gpt-oss-120b-Eagle3-short-context",
        "quantization": "mxfp4",
        "max_batch_size": 1,
        "max_input_len": 4096,
        "max_output_len": 4096,
        "engine_gb": 70,
        "speculative_decoding": True,
        "pre_quantized": True,
        "description": "Frontier-class reasoning. EAGLE-3 speculative decoding.",
    },
}


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def check_trtllm():
    """Verify TensorRT-LLM is installed and report version."""
    try:
        import tensorrt_llm
        print(f"  TensorRT-LLM : {tensorrt_llm.__version__}")
        return True
    except ImportError:
        print("  TensorRT-LLM : NOT INSTALLED")
        print()
        print("  To install:")
        print("    pip install tensorrt-llm")
        print("  Or for Blackwell (CUDA 12.8):")
        print("    pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com")
        return False


def check_environment():
    """Validate Spark environment before doing anything expensive."""
    import torch

    print("\n== Environment ==\n")

    if not torch.cuda.is_available():
        print("  CUDA not available. Cannot proceed.")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    gpu_mem = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

    print(f"  GPU          : {dev.name}")
    print(f"  CUDA cap     : {dev.major}.{dev.minor}")
    print(f"  Memory       : {gpu_mem:.0f} GB unified")

    if dev.major >= 12:
        print(f"  Architecture : Blackwell (TRT-LLM native target)")

    has_trtllm = check_trtllm()

    # Check for trtllm-serve CLI
    trtllm_serve = shutil.which("trtllm-serve")
    print(f"  trtllm-serve : {'found' if trtllm_serve else 'not in PATH'}")

    trtllm_build = shutil.which("trtllm-build")
    print(f"  trtllm-build : {'found' if trtllm_build else 'not in PATH'}")

    print()

    if not has_trtllm:
        sys.exit(1)

    return gpu_mem


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

def cmd_serve(args):
    """Serve a model via TRT-LLM's OpenAI-compatible API."""
    preset = None
    model = None
    quant = args.quantization

    if args.preset and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        model = preset["hf_model"]
        quant = preset.get("quantization", "nvfp4")

        print(f"\n=== Vybn TRT-LLM: {args.preset} ===")
        print(f"    Model          : {model}")
        print(f"    Quantization   : {quant}")
        print(f"    Engine size    : ~{preset['engine_gb']}GB / 128GB")
        if preset.get("speculative_decoding"):
            print(f"    Spec. decoding : EAGLE-3")
        print(f"    {preset['description']}\n")

    elif args.engine_dir:
        print(f"\n=== Vybn TRT-LLM: engine from {args.engine_dir} ===\n")
        _serve_engine_dir(args.engine_dir, args.port, args.model_id)
        return

    elif args.model:
        model = args.model
        print(f"\n=== Vybn TRT-LLM: {model} @ {quant} ===\n")

    else:
        print("Specify --preset, --model, or --engine-dir")
        sys.exit(1)

    gpu_mem = check_environment()

    if preset and preset["engine_gb"] > gpu_mem * 0.85:
        print(f"  NOTE: Engine ~{preset['engine_gb']}GB is a tight fit for")
        print(f"  {gpu_mem:.0f}GB unified memory (KV cache needs headroom).")
        print(f"  TRT-LLM will manage this internally.\n")

    _serve_model(model, quant, args.port, args.model_id, preset)


def _serve_model(model, quantization, port, model_id, preset):
    """Start serving via CLI or Python API.

    TRT-LLM handles the full pipeline (download -> quantize -> engine
    build -> serve) when you pass a HuggingFace model ID directly.
    For pre-quantized models it skips quantization automatically.
    """

    # Prefer trtllm-serve CLI for long-running server processes
    trtllm_serve = shutil.which("trtllm-serve")
    if trtllm_serve:
        print(f"  Starting OpenAI-compatible API server")
        print(f"  Endpoint : http://localhost:{port}/v1")
        print(f"  Model ID : {model_id}")
        print(f"  Stop with Ctrl+C\n")

        cmd = [trtllm_serve, "serve", model, "--host", "0.0.0.0", "--port", str(port)]

        if not (preset and preset.get("pre_quantized")):
            cmd += ["--quantization", quantization]

        if preset and preset.get("speculative_decoding"):
            cmd += ["--speculative_decoding_mode", "eagle"]

        env = {**os.environ, "DS_SKIP_CUDA_CHECK": "1"}
        print(f"  $ {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("\n  Server stopped.")
        return

    # Fallback: Python API with interactive REPL
    print("  trtllm-serve not found, using Python API (interactive mode)...\n")

    from tensorrt_llm import LLM, SamplingParams

    llm_kwargs = {"model": model}
    if not (preset and preset.get("pre_quantized")):
        llm_kwargs["quantization"] = quantization

    t0 = time.time()
    print("  Loading model (first run downloads + builds engine, may take minutes)...")
    llm = LLM(**llm_kwargs)
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

    print("  Interactive mode. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("vybn> ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            outputs = llm.generate([prompt], sampling_params=params)
            for output in outputs:
                print(f"\n{output.outputs[0].text}\n")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nDone.")


def _serve_engine_dir(engine_dir, port, model_id):
    """Serve a pre-built TRT engine directory."""
    trtllm_serve = shutil.which("trtllm-serve")
    if not trtllm_serve:
        print("trtllm-serve not in PATH. Install tensorrt-llm.")
        sys.exit(1)

    cmd = [trtllm_serve, "serve", "--engine_dir", engine_dir,
           "--host", "0.0.0.0", "--port", str(port)]
    print(f"  $ {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n  Server stopped.")


# ---------------------------------------------------------------------------
# Full pipeline: merge LoRA -> quantize -> build -> serve
# ---------------------------------------------------------------------------

def cmd_full(args):
    """Full Vybn pipeline: merge LoRA adapter, quantize, build, serve."""
    print("\n=== Vybn TRT-LLM: Full Pipeline ===\n")
    gpu_mem = check_environment()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "hf_merged"
    engine_dir = output_dir / "trt_engine"

    # Stage 1: Merge LoRA (if provided)
    if args.lora_dir:
        print("== Stage 1: Merge LoRA into base model ==\n")
        _merge_lora(args.hf_base, args.lora_dir, str(merged_dir))
        serve_source = str(merged_dir)
    else:
        print("== Stage 1: No LoRA specified, using base model directly ==\n")
        serve_source = args.hf_base

    # Stage 2+: TRT-LLM handles conversion + quantization + engine build
    quant = args.quantization
    print(f"== Stage 2: TRT-LLM quantize ({quant}) + build ==\n")

    if args.serve:
        _serve_model(serve_source, quant, args.port, args.model_id, None)
    else:
        _build_engine(serve_source, quant, str(engine_dir), args)
        print(f"\n  Engine saved to {engine_dir}")
        print(f"  Serve with:  python3 trtllm_pipeline.py serve --engine-dir {engine_dir}")


def _merge_lora(base_model, lora_dir, output_dir):
    """Merge a PEFT LoRA adapter into the base model."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        print(f"  Merged model exists at {output_dir}, skipping.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Base  : {base_model}")
    print(f"  LoRA  : {lora_dir}")
    print(f"  Output: {output_dir}\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("  Loading base model (CPU only for merge)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print("  Applying LoRA...")
    model = PeftModel.from_pretrained(model, lora_dir)
    print("  Merging weights...")
    model = model.merge_and_unload()

    print("  Saving merged checkpoint...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    del model
    gc.collect()
    print(f"  Done.\n")


def _build_engine(model_path, quantization, engine_dir, args):
    """Build a TRT-LLM engine from an HF model checkpoint."""
    trtllm_build = shutil.which("trtllm-build")

    if trtllm_build:
        cmd = [
            trtllm_build,
            "--model_dir", model_path,
            "--output_dir", engine_dir,
            "--quantization", quantization,
            "--max_batch_size", str(getattr(args, 'max_batch_size', 4)),
            "--max_input_len", str(getattr(args, 'max_input_len', 4096)),
            "--max_output_len", str(getattr(args, 'max_output_len', 4096)),
        ]
        print(f"  $ {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
    else:
        from tensorrt_llm import LLM
        print(f"  Building via Python API...")
        t0 = time.time()
        llm = LLM(model=model_path, quantization=quantization)
        llm.save(engine_dir)
        print(f"  Built in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vybn TRT-LLM pipeline for DGX Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 trtllm_pipeline.py list
  python3 trtllm_pipeline.py serve --preset qwen3-14b
  python3 trtllm_pipeline.py serve --preset gpt-oss-120b
  python3 trtllm_pipeline.py full --hf-base Qwen/Qwen3-14B --lora-dir ./adapter --serve
""",
    )
    sub = parser.add_subparsers(dest="command")

    # -- serve --
    sp = sub.add_parser("serve", help="Serve a model via OpenAI-compatible API")
    sp.add_argument("--preset", choices=list(PRESETS.keys()),
                    help="Validated model preset (see 'list')")
    sp.add_argument("--model", help="HuggingFace model ID or local path")
    sp.add_argument("--engine-dir", help="Pre-built TRT engine directory")
    sp.add_argument("--quantization", default="nvfp4")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--model-id", default="vybn-local")

    # -- full --
    fp = sub.add_parser("full", help="Full pipeline: merge LoRA -> quantize -> serve")
    fp.add_argument("--hf-base", required=True, help="Base HF model ID or path")
    fp.add_argument("--lora-dir", help="Trained LoRA adapter directory")
    fp.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    fp.add_argument("--quantization", default="nvfp4")
    fp.add_argument("--serve", action="store_true", help="Serve after building")
    fp.add_argument("--port", type=int, default=8000)
    fp.add_argument("--model-id", default="vybn-local")

    # -- list --
    sub.add_parser("list", help="List validated model presets")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "full":
        cmd_full(args)
    elif args.command == "list":
        print("\n=== DGX Spark Presets (128GB unified memory) ===\n")
        for name, p in PRESETS.items():
            spec = "EAGLE-3" if p.get("speculative_decoding") else ""
            pre = "pre-quantized" if p.get("pre_quantized") else ""
            tags = ", ".join(filter(None, [pre, spec]))
            tag_str = f"  [{tags}]" if tags else ""
            print(f"  {name:20s}  ~{p['engine_gb']:3d}GB  {p['quantization']:6s}  {p['description']}{tag_str}")
        print()
        print("  Serve:   python3 trtllm_pipeline.py serve --preset <name>")
        print("  Details: python3 trtllm_pipeline.py serve --help")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
