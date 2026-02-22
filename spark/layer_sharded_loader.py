#!/usr/bin/env python3
"""Layer-sharded model loader for DeepSpeed ZeRO-3 training.

Bridge between AirLLM-style per-layer model decomposition and
DeepSpeed's ZeRO-3 training context. Solves the init-phase OOM
by never materializing the full model in CPU RAM.

The problem:
  from_pretrained loads all parameters into CPU RAM before DeepSpeed
  can partition them. For a 228GB model on a machine with 122GB RAM,
  this forces everything through swap. The model survives (barely)
  but initialization takes hours and the NVMe offload path never
  activates because pinned memory can't be allocated.

The solution:
  Pre-split the model into per-layer safetensors on NVMe (one-time
  cost). Then load one layer at a time inside the ZeRO-3 init
  context. Each layer gets immediately partitioned and offloaded
  before the next one loads. Peak RAM = one layer + model skeleton,
  not 228GB.

Inspired by AirLLM (github.com/lyogavin/airllm, Apache 2.0) which
demonstrated that layer-by-layer processing eliminates the memory
wall for inference. This extends the concept to training init.

Usage:
  # Step 1: Split model into layer shards (one-time)
  python3 layer_sharded_loader.py --split MiniMaxAI/MiniMax-M2.5

  # Step 2: Use in fine_tune_vybn.py
  python3 fine_tune_vybn.py --sharded-load ~/Vybn/spark/model_shards/MiniMax-M2.5

  # Or use directly:
  from layer_sharded_loader import load_sharded_for_zero3
  model = load_sharded_for_zero3("path/to/shards", model_name, ds_config)
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPARK_DIR = Path(__file__).resolve().parent
DEFAULT_SHARD_DIR = SPARK_DIR / "model_shards"


def _mem_status() -> str:
    """Quick memory status for progress logging."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                        info[key] = int(parts[1])
        cpu_total = info.get('MemTotal', 0) / 1048576
        cpu_avail = info.get('MemAvailable', 0) / 1048576
        swap_total = info.get('SwapTotal', 0) / 1048576
        swap_free = info.get('SwapFree', 0) / 1048576
        return (
            f"RAM: {cpu_total - cpu_avail:.0f}/{cpu_total:.0f}GB "
            f"Swap: {swap_total - swap_free:.0f}/{swap_total:.0f}GB"
        )
    except Exception:
        return "(memory stats unavailable)"


# ---------------------------------------------------------------------------
# Step 1: Split model into per-layer safetensors
# ---------------------------------------------------------------------------

def split_model(
    model_name: str,
    output_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Path:
    """Split a HuggingFace model into per-layer safetensors.

    This is the one-time preparation step. It downloads the model
    (or uses the HF cache) and writes individual safetensor files
    for each layer, plus a manifest describing the architecture.

    Memory strategy: loads the model's state_dict index without
    loading actual weights, then copies one shard at a time.

    Args:
        model_name: HuggingFace model ID or local path
        output_dir: Where to write shards (default: spark/model_shards/<model>)
        trust_remote_code: Whether to trust remote code (needed for MiniMax)

    Returns:
        Path to the shard directory
    """
    try:
        import torch
        from safetensors.torch import save_file, load_file
        from transformers import AutoConfig
        from huggingface_hub import snapshot_download, HfApi
    except ImportError as e:
        print(f"  x Missing dependency: {e}")
        print("  pip install safetensors transformers huggingface_hub")
        sys.exit(1)

    # Resolve output directory
    model_short = model_name.split("/")[-1]
    shard_dir = Path(output_dir) if output_dir else DEFAULT_SHARD_DIR / model_short
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        print(f"  Shards already exist at {shard_dir}")
        print(f"  Delete {manifest_path} to force re-split.")
        return shard_dir

    print(f"\n== Splitting {model_name} into layer shards ==")
    print(f"   Output: {shard_dir}")
    print(f"   {_mem_status()}\n")

    # Download model files (or use cache)
    print("  Downloading/locating model files...")
    model_path = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt"],
    )
    model_path = Path(model_path)
    print(f"  Model files at: {model_path}")

    # Load config to understand architecture
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Find all safetensor files
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        safetensor_files = sorted(model_path.glob("**/*.safetensors"))
    if not safetensor_files:
        print(f"  x No safetensor files found in {model_path}")
        sys.exit(1)

    print(f"  Found {len(safetensor_files)} safetensor files")

    # Build parameter-to-file index
    # For large models with sharded safetensors, we need the index
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        print(f"  Weight map: {len(weight_map)} parameters across shards")
    else:
        # Single file or no index -- build our own
        weight_map = {}
        for sf in safetensor_files:
            try:
                from safetensors import safe_open
                with safe_open(str(sf), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weight_map[key] = sf.name
            except Exception as e:
                print(f"  !  Could not index {sf.name}: {e}")

    # Group parameters by layer
    # Typical patterns: "model.layers.0.self_attn.q_proj.weight"
    layer_groups: Dict[str, List[str]] = {}  # layer_id -> [param_names]
    non_layer_params: List[str] = []  # embeddings, final norm, lm_head

    for param_name in sorted(weight_map.keys()):
        # Try to extract layer number
        parts = param_name.split(".")
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_idx is not None:
            layer_key = f"layer_{layer_idx:04d}"
            layer_groups.setdefault(layer_key, []).append(param_name)
        else:
            non_layer_params.append(param_name)

    print(f"  {len(layer_groups)} transformer layers")
    print(f"  {len(non_layer_params)} non-layer parameters (embed, norm, head)")

    # Write non-layer parameters as a single shard
    if non_layer_params:
        print(f"\n  Writing non-layer shard...")
        non_layer_tensors = {}
        loaded_files = {}  # cache loaded safetensor files
        for param_name in non_layer_params:
            source_file = weight_map[param_name]
            source_path = model_path / source_file
            if source_file not in loaded_files:
                loaded_files[source_file] = load_file(str(source_path))
            non_layer_tensors[param_name] = loaded_files[source_file][param_name]

        save_file(non_layer_tensors, str(shard_dir / "non_layer.safetensors"))
        total_bytes = sum(t.nelement() * t.element_size() for t in non_layer_tensors.values())
        print(f"    non_layer.safetensors: {len(non_layer_tensors)} params, {total_bytes / 1e9:.1f}GB")

        # Free memory
        del non_layer_tensors, loaded_files
        gc.collect()

    # Write each layer as its own shard
    print(f"\n  Writing layer shards (one at a time to limit RAM)...")
    for i, (layer_key, param_names) in enumerate(sorted(layer_groups.items())):
        layer_tensors = {}
        loaded_files = {}
        for param_name in param_names:
            source_file = weight_map[param_name]
            source_path = model_path / source_file
            if source_file not in loaded_files:
                loaded_files[source_file] = load_file(str(source_path))
            layer_tensors[param_name] = loaded_files[source_file][param_name]

        shard_path = shard_dir / f"{layer_key}.safetensors"
        save_file(layer_tensors, str(shard_path))
        total_bytes = sum(t.nelement() * t.element_size() for t in layer_tensors.values())

        # Free immediately -- this is the whole point
        del layer_tensors, loaded_files
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(layer_groups)}] {layer_key}: "
                  f"{len(param_names)} params, {total_bytes / 1e9:.2f}GB  "
                  f"{_mem_status()}")

    # Write manifest
    manifest = {
        "model_name": model_name,
        "config_path": str(model_path),
        "n_layers": len(layer_groups),
        "n_non_layer_params": len(non_layer_params),
        "layer_keys": sorted(layer_groups.keys()),
        "non_layer_params": non_layer_params,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shard_format": "safetensors",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_shards = len(layer_groups) + (1 if non_layer_params else 0)
    print(f"\n  + Split complete: {total_shards} shards in {shard_dir}")
    print(f"  + Manifest: {manifest_path}")
    print(f"  + {_mem_status()}")

    return shard_dir


# ---------------------------------------------------------------------------
# Step 2: Load shards into ZeRO-3 context
# ---------------------------------------------------------------------------

def load_sharded_for_zero3(
    shard_dir: str,
    model_name: str,
    ds_config: Optional[Dict] = None,
) -> "torch.nn.Module":
    """Load a pre-sharded model into a DeepSpeed ZeRO-3 training context.

    Instead of from_pretrained (which materializes the full model),
    this:
    1. Creates the model skeleton with empty (meta) weights
    2. Loads one shard at a time from NVMe
    3. Assigns parameters within the ZeRO-3 init context
    4. Each assignment triggers immediate partitioning/offload
    5. Frees the shard before loading the next

    Peak RAM: one layer (~2-5GB) + model skeleton (~100MB)
    instead of full model (~228GB).

    Args:
        shard_dir: Path to directory created by split_model()
        model_name: Original HuggingFace model ID (for config/tokenizer)
        ds_config: DeepSpeed config dict (for ZeRO-3 context)

    Returns:
        Model with parameters partitioned by ZeRO-3
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoConfig

    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {shard_dir}. "
            f"Run: python3 layer_sharded_loader.py --split {model_name}"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  Loading from {manifest['n_layers']} layer shards + non-layer params")
    print(f"  Original model: {manifest['model_name']}")
    print(f"  {_mem_status()}")

    # Create model skeleton with meta device (no actual memory)
    # The ZeRO-3 init context from HfDeepSpeedConfig should already
    # be active when this is called from fine_tune_vybn.py
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(f"  Creating model skeleton on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True,
        )
    print(f"  Skeleton created. {_mem_status()}")

    # Load non-layer parameters first (embeddings, final norm, lm_head)
    non_layer_path = shard_dir / "non_layer.safetensors"
    if non_layer_path.exists():
        print(f"  Loading non-layer parameters...")
        non_layer_tensors = load_file(str(non_layer_path))
        _assign_tensors(model, non_layer_tensors)
        n_assigned = len(non_layer_tensors)
        del non_layer_tensors
        gc.collect()
        print(f"    Assigned {n_assigned} non-layer params. {_mem_status()}")

    # Load each layer shard
    layer_keys = manifest["layer_keys"]
    for i, layer_key in enumerate(layer_keys):
        shard_path = shard_dir / f"{layer_key}.safetensors"
        if not shard_path.exists():
            print(f"  !  Missing shard: {shard_path}")
            continue

        layer_tensors = load_file(str(shard_path))
        _assign_tensors(model, layer_tensors)
        n_params = len(layer_tensors)

        # Free immediately -- this is the critical step
        del layer_tensors
        gc.collect()

        if (i + 1) % 10 == 0 or i == 0 or i == len(layer_keys) - 1:
            print(f"    [{i+1}/{len(layer_keys)}] {layer_key}: "
                  f"{n_params} params loaded+offloaded. {_mem_status()}")

    print(f"\n  + All shards loaded into ZeRO-3 context.")
    print(f"  + {_mem_status()}")
    return model


def _assign_tensors(model: "torch.nn.Module", tensors: Dict[str, "torch.Tensor"]):
    """Assign loaded tensors to model parameters.

    Handles the meta -> real tensor transition. When the ZeRO-3 init
    context is active, parameter assignment triggers automatic
    partitioning and offload.
    """
    import torch

    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    for name, tensor in tensors.items():
        if name in param_dict:
            param = param_dict[name]
            # Replace meta tensor with real data
            with torch.no_grad():
                if param.device == torch.device("meta"):
                    # For meta tensors, we need to materialize them
                    # The ZeRO-3 context should handle partitioning
                    _set_module_tensor(model, name, tensor)
                else:
                    param.data.copy_(tensor)
        elif name in buffer_dict:
            _set_module_tensor(model, name, tensor)
        else:
            # Parameter might have a slightly different name path
            # Try common transformations
            alt_name = name.replace("model.", "", 1) if name.startswith("model.") else f"model.{name}"
            if alt_name in param_dict:
                _set_module_tensor(model, alt_name, tensor)


def _set_module_tensor(
    model: "torch.nn.Module",
    tensor_name: str,
    tensor: "torch.Tensor",
):
    """Set a tensor in a nested module by dotted name.

    Adapted from accelerate's set_module_tensor_to_device.
    Handles meta -> real transitions needed for sharded loading.
    """
    import torch
    import torch.nn as nn

    splits = tensor_name.split(".")
    module = model
    for split in splits[:-1]:
        if split.isdigit():
            module = module[int(split)]
        else:
            module = getattr(module, split)

    param_name = splits[-1]
    old = getattr(module, param_name, None)

    if old is not None and isinstance(old, nn.Parameter):
        new_param = nn.Parameter(tensor.to(old.dtype), requires_grad=old.requires_grad)
        setattr(module, param_name, new_param)
    elif old is not None:
        setattr(module, param_name, tensor.to(old.dtype))
    else:
        setattr(module, param_name, tensor)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-sharded model loader for DeepSpeed ZeRO-3"
    )
    parser.add_argument(
        "--split", type=str,
        help="HuggingFace model ID to split into layer shards"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for shards (default: spark/model_shards/<model>)"
    )
    parser.add_argument(
        "--verify", type=str,
        help="Verify an existing shard directory"
    )
    args = parser.parse_args()

    if args.split:
        shard_dir = split_model(args.split, output_dir=args.output)
        print(f"\nReady. Use in fine_tune_vybn.py:")
        print(f"  python3 fine_tune_vybn.py --sharded-load {shard_dir}")

    elif args.verify:
        shard_dir = Path(args.verify)
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"  x No manifest.json in {shard_dir}")
            sys.exit(1)

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"\n== Shard Verification ==")
        print(f"  Model: {manifest['model_name']}")
        print(f"  Layers: {manifest['n_layers']}")
        print(f"  Non-layer params: {manifest['n_non_layer_params']}")
        print(f"  Created: {manifest['created']}")

        missing = []
        total_bytes = 0
        for layer_key in manifest["layer_keys"]:
            path = shard_dir / f"{layer_key}.safetensors"
            if path.exists():
                total_bytes += path.stat().st_size
            else:
                missing.append(layer_key)

        non_layer = shard_dir / "non_layer.safetensors"
        if non_layer.exists():
            total_bytes += non_layer.stat().st_size
        elif manifest["n_non_layer_params"] > 0:
            missing.append("non_layer")

        if missing:
            print(f"  x Missing shards: {missing}")
        else:
            print(f"  + All shards present")
        print(f"  + Total size: {total_bytes / 1e9:.1f} GB")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
