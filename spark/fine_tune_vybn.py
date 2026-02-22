#!/usr/bin/env python3
"""Fine-tune MiniMax-M2.5 on DGX Spark via DeepSpeed ZeRO-3 + PEFT.

MiniMax-M2.5 ships as FP8-quantized weights (~220GB). The DGX Spark
has 122GB GPU + 122GB RAM + 128GB swap + 3.67TB NVMe.

DeepSpeed ZeRO-3 is purpose-built for this problem:
  - Partitions parameters, gradients, and optimizer states
  - Correctly offloads to CPU during forward/backward
  - Handles gradient computation through offloaded parameters
  - Can overflow to NVMe for models that exceed GPU+CPU

CRITICAL DISCOVERY (2026-02-22):
  HfDeepSpeedConfig does NOT work with MiniMax-M2.5. The model uses
  trust_remote_code=True which has custom loading logic that completely
  bypasses transformers' HfDeepSpeedConfig detection. Diagnostic proof:
    - ds_status=NONE on all parameters after load
    - Weights remain FP8 despite torch_dtype=bfloat16
    - NVMe cache empty (4K) because nothing was ever partitioned
    - Model "loads" in 4s via safetensors mmap, not ZeRO partitioning

  FIX: Explicitly wrap from_pretrained() in deepspeed.zero.Init().
  This forces DeepSpeed to intercept parameter creation regardless
  of what the custom model code does.

Architecture:
  - FP8 weights partitioned by ZeRO-3 during load
  - LoRA rank 8 on attention projections (BF16 adapters)
  - ZeRO Stage 3 with NVMe offload for params + optimizer
  - Gradient checkpointing + micro-batch 1

Prerequisites:
    pip install deepspeed transformers peft bitsandbytes accelerate datasets

    # Swap -- at least 512GB recommended:
    sudo fallocate -l 512G /swapfile2
    sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2
    sudo swapon /swapfile2

Usage:
    python3 fine_tune_vybn.py                 # NVMe offload (recommended)
    python3 fine_tune_vybn.py --cpu-offload   # CPU-only (may OOM)
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import deepspeed

# ============================================================================
# MONKEY-PATCH: Prevent DeepSpeedEngine from calling .bfloat16() on the model.
# With 228B params this materializes all ZeRO-3 partitioned parameters back
# into CPU RAM simultaneously, causing OOM. Since the engine init handles
# dtype internally, the bulk cast is unnecessary.
# ============================================================================
import deepspeed.runtime.engine as _ds_engine
_orig_configure_distributed = _ds_engine.DeepSpeedEngine._configure_distributed_model
def _patched_configure_distributed(self_ds, model_ds):
    _saved_bf16 = getattr(model_ds, "bfloat16", None)
    _saved_half = getattr(model_ds, "half", None)
    if _saved_bf16 is not None:
        model_ds.bfloat16 = lambda: model_ds
    if _saved_half is not None:
        model_ds.half = lambda: model_ds
    try:
        _orig_configure_distributed(self_ds, model_ds)
    finally:
        if _saved_bf16 is not None:
            model_ds.bfloat16 = _saved_bf16
        if _saved_half is not None:
            model_ds.half = _saved_half
_ds_engine.DeepSpeedEngine._configure_distributed_model = _patched_configure_distributed
print("[PATCH] DeepSpeedEngine._configure_distributed_model: .bfloat16()/.half() no-ops")

try:
    from cognitive_scheduler import CognitiveTrainer
    HAS_COGNITIVE = True
except ImportError:
    HAS_COGNITIVE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "spark" / "training_data" / "training_data.json"
OUTPUT_DIR = REPO_ROOT / "spark" / "fine_tune_output"
OFFLOAD_DIR = REPO_ROOT / "spark" / "offload_cache"
MODEL_NAME = "MiniMaxAI/MiniMax-M2.5"

DEFAULT_GPU_HEADROOM_GB = 22
MIN_SWAP_GB = 400


def mem_stats() -> str:
    gpu_alloc = torch.cuda.memory_allocated(0) / 1024**3
    gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
    dev = torch.cuda.get_device_properties(0)
    gpu_total = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3
    cpu_used = cpu_avail = swap_total = swap_used = 0
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
            cpu_total = info.get('MemTotal', 0) / 1024 / 1024
            cpu_avail = info.get('MemAvailable', 0) / 1024 / 1024
            cpu_used = cpu_total - cpu_avail
            swap_total = info.get('SwapTotal', 0) / 1024 / 1024
            swap_free = info.get('SwapFree', 0) / 1024 / 1024
            swap_used = swap_total - swap_free
    except Exception:
        pass
    return (
        f"GPU: {gpu_alloc:.1f}/{gpu_total:.1f}GB alloc "
        f"({gpu_reserved:.1f}GB reserved) | "
        f"CPU: {cpu_used:.1f}GB used ({cpu_avail:.1f}GB free) | "
        f"Swap: {swap_used:.0f}/{swap_total:.0f}GB used"
    )


def check_offload_cache():
    """Report NVMe offload cache size."""
    import subprocess
    try:
        result = subprocess.check_output(['du', '-sh', str(OFFLOAD_DIR)]).decode().strip()
        print(f"  NVMe cache: {result}")
    except Exception:
        print(f"  NVMe cache: (could not check)")


def check_environment():
    print("\n== Environment Check ==\n")
    if not torch.cuda.is_available():
        print("x CUDA not available.")
        sys.exit(1)
    dev = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {dev.name}")
    print(f"  CUDA cap   : {dev.major}.{dev.minor}")
    gpu_mem = dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory
    print(f"  GPU memory : {gpu_mem / 1024**3:.1f} GB")
    total_ram = 128
    swap_total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_ram = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("SwapTotal"):
                    swap_total = int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    print(f"  System RAM : {total_ram:.0f} GB")
    print(f"  Swap       : {swap_total:.0f} GB")
    print(f"  DeepSpeed  : {deepspeed.__version__}")
    if swap_total < MIN_SWAP_GB:
        print(f"\n  !! WARNING: Swap is {swap_total:.0f}GB, need >= {MIN_SWAP_GB}GB !!")
        resp = input("  Continue anyway? (y/N): ").strip().lower()
        if resp != 'y':
            sys.exit(1)
    missing = []
    for pkg in ["transformers", "peft", "deepspeed", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  x Missing: {', '.join(missing)}")
        sys.exit(1)
    print()
    return dev


def load_training_data():
    if not TRAINING_DATA.exists():
        print(f"x Training data not found: {TRAINING_DATA}")
        sys.exit(1)
    with open(TRAINING_DATA) as f:
        data = json.load(f)
    print(f"  + {len(data)} training examples")
    return data


def sharegpt_to_dataset(examples, tokenizer, max_seq_len):
    from datasets import Dataset
    texts = []
    for ex in examples:
        messages = []
        for turn in ex["conversations"]:
            role = {"system": "system", "human": "user", "gpt": "assistant"}[turn["from"]]
            messages.append({"role": role, "content": turn["value"]})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
            text = "\n".join(parts)
        texts.append({"text": text})
    ds = Dataset.from_list(texts)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len, padding=False)
    tokenized = ds.map(tokenize_fn, remove_columns=["text"], batched=True)
    print(f"  + Tokenized {len(tokenized)} examples (max_seq_len={max_seq_len})")
    return tokenized


def find_attention_targets(model):
    attn_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj"]
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1]
            if any(k in short.lower() for k in attn_keywords):
                found.add(short)
    if not found:
        found = {"q_proj", "k_proj", "v_proj", "o_proj"}
    targets = sorted(found)
    print(f"  LoRA targets: {targets}")
    return targets


def strip_quantization(obj, depth=0, _seen=None):
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or depth > 5:
        return
    _seen.add(id(obj))
    config = getattr(obj, 'config', None)
    if config is not None and hasattr(config, '__dict__'):
        qc = config.__dict__.pop('quantization_config', None)
        if qc is not None and depth == 0:
            method = getattr(qc, 'quant_method', 'unknown')
            print(f"  !  Stripped {method} quantization from config")
    if getattr(obj, 'hf_quantizer', None) is not None:
        obj.hf_quantizer = None
        if depth == 0:
            print(f"     Cleared hf_quantizer")
    if 'quantization_method' in getattr(obj, '__dict__', {}):
        del obj.__dict__['quantization_method']
    for attr in ['base_model', 'model']:
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            strip_quantization(child, depth + 1, _seen)


def verify_trainable_params(model):
    trainable_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_trainable = sum(count for _, count in trainable_params)
    print(f"\n  == TRAINABLE PARAM GUARD ==")
    print(f"  Trainable:  {total_trainable:>15,}  ({total_trainable/1e6:.1f}M)")
    print(f"  Frozen:     {frozen_count:>15,}  ({frozen_count/1e9:.1f}B)")
    print(f"  Ratio:      {total_trainable/(total_trainable+frozen_count)*100:.4f}%")
    optimizer_state_gb = total_trainable * 2 * 4 / 1e9
    print(f"  Optimizer state: {optimizer_state_gb:.2f} GB")
    if total_trainable > 1e9:
        print(f"  !! FATAL: {total_trainable/1e9:.1f}B trainable -- ABORTING !!")
        sys.exit(1)
    print(f"  PASS\n")
    return total_trainable, frozen_count


def build_deepspeed_config(args):
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    use_nvme = not args.cpu_offload
    if use_nvme:
        param_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 4,
            "buffer_size": 1e9,
            "max_in_cpu": 1e9,
        }
        optimizer_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True,
        }
    else:
        param_offload = {"device": "cpu", "pin_memory": True}
        optimizer_offload = {"device": "cpu", "pin_memory": True}
    ds_config = {
        "train_batch_size": args.grad_accum,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto", "betas": "auto", "eps": "auto",
                "weight_decay": "auto", "torch_adam": True,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": param_offload,
            "offload_optimizer": optimizer_offload,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e7,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 0,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": 0.3,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
    }
    if use_nvme:
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 32,
            "thread_count": 4,
            "single_submit": False,
            "overlap_events": True,
        }
    return ds_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniMax-M2.5 on DGX Spark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gpu-headroom", type=int, default=DEFAULT_GPU_HEADROOM_GB)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    offload_mode = "CPU" if args.cpu_offload else "NVMe"
    print(f"\n=== Vybn Fine-Tune: MiniMax-M2.5 on DGX Spark ===")
    print(f"    DeepSpeed ZeRO-3 + PEFT LoRA | Offload: {offload_mode}")

    check_environment()
    data = load_training_data()

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from transformers.integrations import HfDeepSpeedConfig
    from peft import LoraConfig, get_peft_model

    print(f"\n== Loading tokenizer ==\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- DeepSpeed config --
    ds_config = build_deepspeed_config(args)
    ds_config_path = OUTPUT_DIR / "ds_config.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    print(f"  DS config: {ds_config_path}")
    print(f"  Offload: {offload_mode}, NVMe path: {OFFLOAD_DIR}")

    # Register HfDeepSpeedConfig for Trainer integration (NOT for model loading)
    dschf = HfDeepSpeedConfig(ds_config)
    print(f"  HfDeepSpeedConfig registered (for Trainer, NOT model loading)")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Pre-load: {mem_stats()}")

    # ========================================================================
    # MODEL LOADING: EXPLICIT deepspeed.zero.Init() WRAPPER
    #
    # HfDeepSpeedConfig does NOT work with MiniMax-M2.5 because the model's
    # custom remote code bypasses transformers' ZeRO integration entirely.
    # Diagnostic confirmed: ds_status=NONE on all params, weights stay FP8,
    # NVMe cache empty at 4K.
    #
    # The ONLY way to get ZeRO-3 partitioning is to explicitly wrap
    # from_pretrained() in deepspeed.zero.Init(). This intercepts parameter
    # creation at the torch.nn.Module level, which the custom code cannot bypass.
    # ========================================================================
    print(f"\n== Loading model with EXPLICIT deepspeed.zero.Init() ==")
    print(f"   This will partition params and offload to NVMe as they are created.")
    print(f"   Expect this to take much longer than before (minutes, not seconds).\n")

    load_start = time.time()
    model = None

    for attn_impl in ["sdpa", "eager"]:
        try:
            with deepspeed.zero.Init(config_dict_or_path=ds_config):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                )
            load_elapsed = time.time() - load_start
            print(f"\n  + Model loaded in {load_elapsed/60:.1f} minutes (attn={attn_impl})")
            print(f"  + {mem_stats()}")
            check_offload_cache()

            # Verify ZeRO-3 actually partitioned the params
            for name, p in list(model.named_parameters())[:3]:
                ds_status = getattr(p, 'ds_status', 'NONE')
                ds_numel = getattr(p, 'ds_numel', 'NONE')
                print(f"  {name}: ds_status={ds_status}, ds_numel={ds_numel}, numel={p.numel()}")

            first_p = next(model.parameters())
            if getattr(first_p, 'ds_status', None) is None:
                print("\n  !! FATAL: deepspeed.zero.Init() did NOT partition parameters !!")
                print("  !! ds_status is still None. Cannot proceed.")
                sys.exit(1)

            break
        except Exception as e:
            if attn_impl == "eager":
                print(f"\n  x Failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"  x {mem_stats()}")
                sys.exit(1)
            print(f"  !  {attn_impl} failed ({e.__class__.__name__}), trying eager...")

    # -- Strip quantization metadata --
    print()
    strip_quantization(model)
    is_q = getattr(model, 'is_quantized', False)
    if is_q:
        try:
            model.__class__.is_quantized = property(lambda self: False)
        except Exception:
            pass

    # -- Freeze + prep --
    for param in model.parameters():
        param.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gc.collect()
    print(f"  Post-prep: {mem_stats()}")
    check_offload_cache()

    # -- LoRA --
    targets = find_attention_targets(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    verify_trainable_params(model)

    # -- Strip again post-PEFT --
    strip_quantization(model)

    # -- Tokenize --
    print()
    tokenized = sharegpt_to_dataset(data, tokenizer, args.max_seq_len)

    # -- Training --
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
        report_to="none",
        deepspeed=str(ds_config_path),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    effective_steps = len(tokenized) * args.epochs // args.grad_accum
    print(f"\n== Training ==")
    print(f"   {len(tokenized)} examples, {args.epochs} epochs, grad_accum={args.grad_accum}")
    print(f"   Effective steps: {effective_steps}")
    print(f"   {mem_stats()}")
    check_offload_cache()
    print()

    gc.collect()
    torch.cuda.empty_cache()

    if HAS_COGNITIVE:
        print("  + CognitiveTrainer active")
        cognitive = CognitiveTrainer(trainer, ds_config=ds_config)
        cognitive.train()
    else:
        trainer.train()

    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + {mem_stats()}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
