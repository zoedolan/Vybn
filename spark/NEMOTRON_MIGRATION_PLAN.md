# Nemotron 3 Super Migration Plan

*Written: 2026-03-14 by outside-Vybn (Claude Opus) + Zoe. This is the authoritative plan. Follow it exactly. Do not improvise.*

---

## Hardware Ground Truth (burn this into memory)

| | spark-2b7c (primary) | spark-1c8f (secondary) |
|---|---|---|
| Unified memory | 128 GB | 128 GB |
| IP (ConnectX-7) | primary | 169.254.51.101 |
| SSH | — | `ssh -i ~/.ssh/id_ed25519_shared 169.254.51.101` |
| Status | organism runs here | passwordless SSH confirmed |

**Total cluster: 256 GB. Each node sees only its own 128 GB. NCCL/torchrun distributes across both.**

---

## Dead Ends — Never Retry These

| Approach | Why dead | Confirmed |
|---|---|---|
| LoRA-train MiniMax M2.5 on one node | 120+ GB just for weights; zero headroom | ✓ multiple sessions |
| LoRA-train MiniMax M2.5 across both nodes | 229B FP8 weights + optimizer states > 256 GB total | ✓ |
| vLLM with NVFP4 Nemotron | CUDA 13.0 host / 13.1 container mismatch → illegal instruction on 2nd inference | ✓ |
| Docker for training | train_cycle.py was written for Docker; serving uses llama.cpp; mismatched stack | ✓ |
| AWQ / BNB 4-bit / compressed-tensors on MiniMax | still too big; tried every variant | ✓ |

---

## Model Files on Disk

```
~/models/MiniMax-M2.5-GGUF/                             # ~122 GB — INTACT, rollback if needed
~/models/Nemotron-3-Super-120B-GGUF/                    # ~63 GB IQ4_XS GGUF — SERVING model
~/.cache/huggingface/hub/
  models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/  # ~61 GB NVFP4 safetensors — TRAINING model
```

---

## Phase 1 — Get Nemotron Serving (~15 min)

**Complete this phase. Commit. Verify. Only then proceed to Phase 2.**

```bash
cd ~/Vybn

# Step 1: Check GGUF download
du -sh ~/models/Nemotron-3-Super-120B-GGUF/
# Need ~63 GB. If less, resume:
nohup python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bartowski/nvidia_Nemotron-3-Super-120B-A12B-GGUF',
    local_dir=os.path.expanduser('~/models/Nemotron-3-Super-120B-GGUF'),
    allow_patterns=['*IQ4_XS*']
)" >> ~/logs/nemotron-download.log 2>&1 &
# Monitor: tail -f ~/logs/nemotron-download.log

# Step 2: Find GGUF filenames
ls ~/models/Nemotron-3-Super-120B-GGUF/*.gguf
# Expect two shards: *-00001-of-00002.gguf and *-00002-of-00002.gguf
# llama.cpp loads split GGUFs via the first shard filename

# Step 3: Start llama-server (llama.cpp already rebuilt with nemotron-h.cpp support)
# Note: this exact setup worked on port 8001 in the session of 2026-03-13
FIRST_SHARD=$(ls ~/models/Nemotron-3-Super-120B-GGUF/*-00001-of-00002.gguf 2>/dev/null | head -1)
nohup ~/llama.cpp/build/bin/llama-server \
  -m "$FIRST_SHARD" \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -c 8192 \
  --chat-template nemotron \
  >> ~/logs/nemotron-server.log 2>&1 &
echo "Server PID: $!"

# Step 4: Wait for healthy
for i in $(seq 1 40); do
  curl -sf http://localhost:8000/health > /dev/null 2>&1 \
    && echo "HEALTHY after $((i*15))s" && break
  echo -n "."; sleep 15
done

# Step 5: Verify
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Step 6: One breath
. ~/.vybn_keys
VYBN_MODEL_URL=http://127.0.0.1:8000 python3 spark/vybn.py --once

# Step 7: Restore cron
crontab /tmp/crontab-backup-preswap.txt
crontab -l

# Step 8: Update ~/.vybn_keys
cat >> ~/.vybn_keys << 'EOF'
export VYBN_MODEL_URL=http://127.0.0.1:8000
export VYBN_MODEL_NAME=nemotron
export VYBN_MODEL_ID=nemotron-3-super-120b
EOF

# Step 9: Update spark/growth/growth_config.yaml — change serving_model to:
# serving_model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

**STOP. Commit `spark/continuity.md` with updated state. Confirm cron is breathing.**

---

## Phase 2 — Validate NCCL Between Both Sparks (~30 min)

Official playbook: https://build.nvidia.com/spark/nccl

```bash
# On BOTH nodes (run these on spark-2b7c; ssh into spark-1c8f for the same):
git clone https://github.com/NVIDIA/nccl.git ~/nccl
cd ~/nccl && git checkout v2.28.9-1
make -j4 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"

git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests
cd ~/nccl-tests && make MPI=0 NCCL_HOME=~/nccl/build

# Test from spark-2b7c (replace hostnames/IPs as needed):
UCX_NET_DEVICES=enp1s0f0np0 \
NCCL_SOCKET_IFNAME=enp1s0f0np0 \
NCCL_DEBUG=WARN \
~/nccl-tests/build/all_reduce_perf \
  -b 8 -e 128M -f 2 -g 1 \
  -n 2 \
  --host spark-2b7c,169.254.51.101
```

**Must see bus bandwidth > 10 GB/s before proceeding. If NCCL fails, check UCX_NET_DEVICES matches your actual ConnectX-7 interface name (`ip link show | grep -i enp`).**

---

## Phase 3 — Wire Distributed LoRA Training (~45 min)

Official playbook: https://build.nvidia.com/spark/pytorch-fine-tune  
GitHub: https://github.com/nvidia/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune

Nemotron NVFP4 = ~61 GB weights. Across 2 nodes = ~30 GB per node. ~95 GB per node free for LoRA overhead. Comfortable.

Update `spark/growth/train_cycle.py` — replace the subprocess launcher:

```python
def _run_training_subprocess(self, script_path: Path, cycle_dir: Path):
    """Launch two-node distributed training via torchrun + NCCL."""
    import subprocess, os, shlex
    venv = Path.home() / ".venv/spark"
    torchrun = venv / "bin" / "torchrun"
    env = os.environ.copy()
    env.update({
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
        "UCX_NET_DEVICES": "enp1s0f0np0",
        "NCCL_DEBUG": "WARN",
        "MASTER_ADDR": "spark-2b7c",
        "MASTER_PORT": "29500",
    })
    # Start node 1 (spark-1c8f) via SSH in background
    ssh_env = ("MASTER_ADDR=spark-2b7c MASTER_PORT=29500 "
               "NCCL_SOCKET_IFNAME=enp1s0f0np0 UCX_NET_DEVICES=enp1s0f0np0")
    ssh_cmd = [
        "ssh", "-i", str(Path.home() / ".ssh/id_ed25519_shared"),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "169.254.51.101",
        f"{ssh_env} {venv}/bin/torchrun "
        f"--nnodes=2 --nproc_per_node=1 "
        f"--node_rank=1 --master_addr=spark-2b7c --master_port=29500 "
        f"{script_path} >> {cycle_dir}/train_node1.log 2>&1"
    ]
    node1 = subprocess.Popen(ssh_cmd)
    # Start node 0 (local) — blocks until training completes
    result = subprocess.run(
        [str(torchrun),
         "--nnodes=2", "--nproc_per_node=1",
         "--node_rank=0",
         "--master_addr=spark-2b7c", "--master_port=29500",
         str(script_path)],
        env=env
    )
    node1.wait(timeout=7200)  # 2 hour timeout
    return result
```

Also update `model_id` in `train_cycle.py`:
```python
self._model_id = self._merge_cfg.get(
    "serving_model",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
```

And the generated training script must point at the local safetensors cache, not download from HF:
```python
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/"
)
# Use the most recent snapshot dir
import glob
snapshots = sorted(glob.glob(MODEL_PATH + "*/config.json"))
MODEL_PATH = os.path.dirname(snapshots[-1]) if snapshots else MODEL_PATH
```

---

## Phase 4 — LoRA Adapter → GGUF → Hot-Load (~10 min)

llama.cpp's `convert_lora_to_gguf.py` is the right tool. Hot-loading via `POST /lora-adapters` means no server restart.

```bash
# Convert PEFT adapter to GGUF
python3 ~/llama.cpp/convert_lora_to_gguf.py \
  --base ~/models/Nemotron-3-Super-120B-GGUF/ \
  --adapter [ADAPTER_DIR] \
  --outfile [ADAPTER_DIR]/adapter.gguf

# Hot-load (no restart)
curl -s -X POST http://localhost:8000/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id": 1, "path": "[ADAPTER_DIR]/adapter.gguf", "scale": 1.0}]'

# Verify
curl -s http://localhost:8000/lora-adapters
```

---

## Phase 5 — Update Continuity Note

After each phase, update `spark/continuity.md` with:
- What was done and what commands worked
- Current serving state: model, port, cron status, adapter loaded
- Exact next step

**The continuity note must describe facts about the world, not just tasks. Future-you needs to know what is true.**
