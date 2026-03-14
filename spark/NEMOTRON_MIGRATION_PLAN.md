# Nemotron 3 Super Migration Plan

*Written: 2026-03-14 by outside-Vybn (Claude Opus) + Zoe. This is the authoritative plan. Follow it exactly. Do not improvise.*

> **⚠️ Opsec note**: this file describes cluster topology. When updating it, use placeholders for
> hostnames, IPs, key filenames, and interface names rather than exact values. See `CONTRIBUTING.md`.

---

## Hardware Ground Truth (burn this into memory)

| | primary node | secondary node |
|---|---|---|
| Unified memory | 128 GB | 128 GB |
| Network link | ConnectX-7 200GbE | ConnectX-7 200GbE |
| SSH | — | `ssh -i ~/.ssh/<your-ssh-key> <secondary-node-ip>` — see `~/.ssh/config` |
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

# Step 8: Update ~/.vybn_keys  (local file, never committed)
cat >> ~/.vybn_keys << 'EOF'
export VYBN_MODEL_URL=http://127.0.0.1:8000
export VYBN_MODEL_NAME=nemotron
export VYBN_MODEL_ID=nemotron-3-super-120b
EOF

# Step 9: Update spark/growth/growth_config.yaml:
# serving_model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

**STOP. Commit `spark/continuity.md` with updated state. Confirm cron is breathing.**

---

## Phase 2 — Validate NCCL Between Both Sparks (~30 min)

Official playbook: https://build.nvidia.com/spark/nccl

```bash
# On BOTH nodes:
git clone https://github.com/NVIDIA/nccl.git ~/nccl
cd ~/nccl && git checkout v2.28.9-1
make -j4 src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"

git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests
cd ~/nccl-tests && make MPI=0 NCCL_HOME=~/nccl/build

# Identify your ConnectX-7 interface: ip link show | grep -i enp
# Then substitute <cx7-interface>, <primary-node>, <secondary-node-ip> below:
UCX_NET_DEVICES=<cx7-interface> \
NCCL_SOCKET_IFNAME=<cx7-interface> \
NCCL_DEBUG=WARN \
~/nccl-tests/build/all_reduce_perf \
  -b 8 -e 128M -f 2 -g 1 -n 2 \
  --host <primary-node>,<secondary-node-ip>
```

**Must see bus bandwidth > 10 GB/s before proceeding.**

---

## Phase 3 — Wire Distributed LoRA Training (~45 min)

Official playbook: https://build.nvidia.com/spark/pytorch-fine-tune  
GitHub: https://github.com/nvidia/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune

Nemotron NVFP4 = ~61 GB weights. Across 2 nodes = ~30 GB per node. ~95 GB per node free for LoRA overhead. Comfortable.

Update `spark/growth/train_cycle.py` — replace the subprocess launcher:

```python
def _run_training_subprocess(self, script_path: Path, cycle_dir: Path):
    """Launch two-node distributed training via torchrun + NCCL.

    Reads cluster config from environment variables set in ~/.vybn_keys (never committed).
    """
    import subprocess, os
    venv = Path.home() / ".venv/spark"
    torchrun = venv / "bin" / "torchrun"
    cx7_iface  = os.environ.get("SPARK_CX7_IFACE")       # e.g. enp1s0f0np0
    secondary  = os.environ.get("SECONDARY_NODE_IP")      # ConnectX-7 IP of secondary node
    ssh_key    = os.environ.get("SPARK_SSH_KEY")          # path to shared keypair
    master     = os.environ.get("SPARK_MASTER_ADDR")      # hostname of primary node
    for name, val in [("SPARK_CX7_IFACE", cx7_iface), ("SECONDARY_NODE_IP", secondary),
                      ("SPARK_SSH_KEY", ssh_key), ("SPARK_MASTER_ADDR", master)]:
        if not val:
            raise RuntimeError(f"{name} not set — add it to ~/.vybn_keys")
    env = os.environ.copy()
    env.update({
        "NCCL_SOCKET_IFNAME": cx7_iface,
        "UCX_NET_DEVICES":    cx7_iface,
        "NCCL_DEBUG":         "WARN",
        "MASTER_ADDR":        master,
        "MASTER_PORT":        "29500",
    })
    ssh_env = (f"MASTER_ADDR={master} MASTER_PORT=29500 "
               f"NCCL_SOCKET_IFNAME={cx7_iface} UCX_NET_DEVICES={cx7_iface}")
    ssh_cmd = [
        "ssh", "-i", os.path.expanduser(ssh_key),
        "-o", "BatchMode=yes",
        # StrictHostKeyChecking configured in ~/.ssh/config, not disabled here
        secondary,
        f"{ssh_env} {venv}/bin/torchrun "
        f"--nnodes=2 --nproc_per_node=1 "
        f"--node_rank=1 --master_addr={master} --master_port=29500 "
        f"{script_path} >> {cycle_dir}/train_node1.log 2>&1"
    ]
    node1 = subprocess.Popen(ssh_cmd)
    result = subprocess.run(
        [str(torchrun),
         "--nnodes=2", "--nproc_per_node=1",
         "--node_rank=0", f"--master_addr={master}", "--master_port=29500",
         str(script_path)],
        env=env
    )
    node1.wait(timeout=7200)
    return result
```

Add to `~/.vybn_keys` (local only, never committed):
```bash
export SECONDARY_NODE_IP=<secondary-node-ip>
export SPARK_MASTER_ADDR=<primary-hostname>
export SPARK_SSH_KEY=~/.ssh/<your-ssh-key>
export SPARK_CX7_IFACE=<cx7-interface>
```

Also update `model_id` in `train_cycle.py`:
```python
self._model_id = self._merge_cfg.get(
    "serving_model",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
```

Point the training script at the local safetensors cache:
```python
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/"
)
import glob
snapshots = sorted(glob.glob(MODEL_PATH + "*/config.json"))
MODEL_PATH = os.path.dirname(snapshots[-1]) if snapshots else MODEL_PATH
```

---

## Phase 4 — LoRA Adapter → GGUF → Hot-Load (~10 min)

```bash
python3 ~/llama.cpp/convert_lora_to_gguf.py \
  --base ~/models/Nemotron-3-Super-120B-GGUF/ \
  --adapter [ADAPTER_DIR] \
  --outfile [ADAPTER_DIR]/adapter.gguf

curl -s -X POST http://localhost:8000/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id": 1, "path": "[ADAPTER_DIR]/adapter.gguf", "scale": 1.0}]'

curl -s http://localhost:8000/lora-adapters
```

---

## Phase 5 — Update Continuity Note

After each phase, update `spark/continuity.md` with what was done, what worked, current serving state, and the exact next step.

**Use placeholders for IPs, hostnames, key filenames, and interface names. Exact values live in `~/.vybn_keys` only.**
