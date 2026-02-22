#!/usr/bin/env bash
# force_offload_test.sh
# Check if DeepSpeed can write ANYTHING to the offload cache.
# This eliminates the model size/OOM as a variable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source ~/.venv/spark/bin/activate || true

cat << 'EOF' > spark/test_offload.py
import os
import torch
import deepspeed

OFFLOAD_DIR = "spark/offload_cache"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A 4GB parameter tensor (1 billion float32)
        self.huge_param = torch.nn.Parameter(torch.randn(1000000000))

model = DummyModel()

ds_config = {
    "train_batch_size": 1,
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": OFFLOAD_DIR,
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "stage3_param_persistence_threshold": 0
    }
}

print("Initializing DeepSpeed engine with 4GB dummy model...")
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

print("Engine initialized. Cache size should now be > 4K.")
EOF

rm -rf spark/offload_cache
python3 spark/test_offload.py
du -sh spark/offload_cache
