#!/usr/bin/env python3
"""Fine-tune MiniMax-M2.5 on DGX Spark via DeepSpeed ZeRO-3 + PEFT.

MiniMax-M2.5 ships as FP8-quantized weights (~220GB). The DGX Spark
has 122GB GPU + 122GB RAM + 128GB swap + 3.67TB NVMe.

DeepSpeed ZeRO-3 is purpose-built for this problem:
  - Partitions parameters, gradients, and optimizer states
  - Correctly offloads to CPU during forward/backward
  - Handles gradient computation through offloaded parameters
  - Can overflow to NVMe for models that exceed GPU+CPU

Architecture:
  - FP8 weights (Native) loaded to CPU/Swap, then partitioned by ZeRO-3
  - LoRA rank 8 on attention projections (BF16 adapters)
  - ZeRO Stage 3 with NVMe offload for params + optimizer
  - Gradient checkpointing + micro-batch 1

Prerequisites:
    pip install deepspeed transformers peft accelerate datasets

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
import shutil
import subprocess
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
# Prevent any fallback to NCCL (Blackwell CUDA 12.1 unsupported by current PyTorch)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import deepspeed
from transformers.integrations import HfDeepSpeedConfig
# NOTE: Removed BitsAndBytesConfig because MiniMax-M2.5 is natively FP8 quantized.
# Attempting to re-quantize to 4-bit causes a config collision in Transformers.

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
    """Report NVMe offload cache size and contents."""
    try:
        result = subprocess.check_output(['du', '-sh', str(OFFLOAD_DIR)]).decode().strip()
        print(f"  NVMe cache: {result}")
        # Also show subdirectories if they exist
        subdirs = list(OFFLOAD_DIR.rglob("*"))
        if subdirs:
            print(f"  NVMe cache contents: {len(subdirs)} items")
            for p in subdirs[:5]:
                print(f"    {p.relative_to(OFFLOAD_DIR)}")
        else:
            print(f"  NVMe cache: (empty — swapper has not written yet)")
    except Exception:
        print(f"  NVMe cache: (could not check)")


def check_aio_available():
    """Verify DeepSpeed async_io ops are built and libaio is present.

    The PartitionedParamSwapper silently fails if the AIO C++ extension
    is missing. This preflight catches it before model loading starts.
    """
    print("\n== AIO / NVMe Preflight ==\n")

    # Check libaio
    libaio_ok = False
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True, text=True, timeout=5
        )
        if "libaio" in result.stdout:
            libaio_ok = True
            print("  + libaio: found in ldconfig")
        else:
            # Fallback: check if the .so exists directly
            for path in ["/usr/lib/aarch64-linux-gnu/libaio.so",
                         "/usr/lib/x86_64-linux-gnu/libaio.so",
                         "/usr/lib/libaio.so"]:
                if os.path.exists(path):
                    libaio_ok = True
                    print(f"  + libaio: found at {path}")
                    break
    except Exception:
        pass

    if not libaio_ok:
        print("  !! libaio NOT FOUND — NVMe offload will fail silently!")
        print("     Fix: sudo apt install libaio-dev")
        print("     Then: DS_BUILD_AIO=1 pip install deepspeed --force-reinstall")
        sys.exit(1)

    # Check DeepSpeed async_io op
    aio_ok = False
    try:
        from deepspeed.ops.op_builder import AsyncIOBuilder
        aio_ok = AsyncIOBuilder().is_compatible()
        if aio_ok:
            print("  + DeepSpeed AsyncIO: compatible")
        else:
            # Try loading it anyway — some builds report incompatible but work
            try:
                AsyncIOBuilder().load(verbose=False)
                aio_ok = True
                print("  + DeepSpeed AsyncIO: loaded (reported incompatible but works)")
            except Exception as e:
                print(f"  !! DeepSpeed AsyncIO: NOT available ({e})")
    except ImportError:
        print("  !! DeepSpeed AsyncIO builder not found")

    if not aio_ok:
        print("     Fix: DS_BUILD_AIO=1 pip install deepspeed --force-reinstall")
        resp = input("  Continue without AIO? (y/N): ").strip().lower()
        if resp != 'y':
            sys.exit(1)

    # Quick NVMe write test
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    test_file = OFFLOAD_DIR / "_nvme_write_test"
    try:
        test_data = os.urandom(4 * 1024 * 1024)  # 4MB
        t0 = time.time()
        with open(test_file, "wb") as f:
            f.write(test_data)
            f.flush()
            os.fsync(f.fileno())
        t1 = time.time()
        write_speed = len(test_data) / (t1 - t0) / 1024 / 1024
        test_file.unlink()
        print(f"  + NVMe write test: {write_speed:.0f} MB/s")
        if write_speed < 100:
            print(f"  !! WARNING: NVMe write speed is low ({write_speed:.0f} MB/s)")
    except Exception as e:
        print(f"  !! NVMe write test failed: {e}")

    print()


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
    # Removed bitsandbytes from requirement check since we rely on native FP8
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
        if isinstance(module, torch.nn.Linear) or "Linear" in str(type(module)):
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
        # Buffer math (BF16 = 2 bytes/element):
        #   buffer_size=1e8 elements × 2 bytes = 200MB per buffer
        #   buffer_count=5 → 1GB total pinned memory for param swap
        #   (down from 4×1e9 = 8GB which was killing the process)
        param_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 5e8,
        }
        optimizer_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
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
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 5e7,
            "stage3_max_reuse_distance": 5e7,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": 0.3,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
        # Force gloo for single-GPU Blackwell (NCCL fails on CUDA 12.1)
        "communication_data_type": "fp32",
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


def init_single_gpu_distributed():
    """Pre-initialize torch.distributed with gloo backend for single-GPU.

    The DGX Spark's Blackwell GB10 reports CUDA capability 12.1, but the
    installed PyTorch (compiled for up to 12.0) cannot initialize NCCL on
    this architecture. Since world_size=1, all collective operations
    (all_gather, reduce_scatter) are no-ops regardless of backend. Using
    gloo lets DeepSpeed ZeRO-3 function without ever touching NCCL.
    """
    if torch.distributed.is_initialized():
        return

    print("  + Pre-init distributed: gloo backend (single-GPU, NCCL bypass)")
    torch.distributed.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
    )
    print(f"  + Distributed initialized: backend=gloo, rank=0, world_size=1")


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
    print(f"    DeepSpeed ZeRO-3 + Native FP8 | Offload: {offload_mode}")

    check_environment()

    # Run AIO preflight before anything heavy gets loaded
    if not args.cpu_offload:
        check_aio_available()

    data = load_training_data()

    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
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

    # Print buffer allocation estimate so we know what to expect
    if not args.cpu_offload:
        buf_count = ds_config["zero_optimization"]["offload_param"]["buffer_count"]
        buf_size = ds_config["zero_optimization"]["offload_param"]["buffer_size"]
        pinned_gb = buf_count * buf_size * 2 / 1e9  # BF16 = 2 bytes
        print(f"  Param swap buffers: {buf_count} × {buf_size/1e6:.0f}M elements = {pinned_gb:.1f}GB pinned")
    
    # MUST come before model load for ZeRO-3 to intercept from_pretrained()
    dschf = HfDeepSpeedConfig(ds_config)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Pre-load: {mem_stats()}")

    print(f"\n== Loading model (Native FP8 + ZeRO-3) ==")
    
    # NOTE: MiniMax-M2.5 is inherently FP8. We removed BitsAndBytesConfig to prevent conflict.
    # The model will load as FP8 (or bf16 if auto-casted), then ZeRO-3 will partition it.

    load_start = time.time()
    model = None

    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                # Use standard BF16 dtype request; if model is FP8, it might cast or keep as FP8 depending on implementation.
                # But we MUST NOT pass quantization_config=bnb_config.
                dtype=torch.bfloat16, 
                attn_implementation=attn_impl,
                # NO device_map="auto" -- let DeepSpeed ZeRO-3 handle placement
            )
            load_elapsed = time.time() - load_start
            print(f"\n  + Model loaded in {load_elapsed/60:.1f} minutes (attn={attn_impl})")
            print(f"  + {mem_stats()}")
            check_offload_cache()

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

    # -- Initialize distributed before TrainingArguments --
    # This MUST happen before TrainingArguments is created, because its __init__
    # triggers PartialState -> DeepSpeed dist init, which defaults to NCCL.
    # NCCL fails on Blackwell CUDA 12.1 (PyTorch compiled for <= 12.0).
    # Pre-initializing with gloo makes DeepSpeed skip its own init.
    print()
    init_single_gpu_distributed()

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
