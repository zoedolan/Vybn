"""Experiment D v3: Geometric Training — with raw activation checkpoints for Experiment E.

Identical to v2 (same seed, same hyperparams, same architecture) but the
geometric_snapshot function now also saves:
  - A representative 384-dim activation vector per layer (mean over batch and
    sequence positions), suitable for QGT computation in E.2
  - The full unit-normalized centroid, suitable for building transition
    unitaries in E.3

Storage cost: ~30 snapshots × 6 layers × 384 floats × 2 runs ≈ 550 KB total.
Negligible. But it makes E.2 and E.3 honest — no invented embeddings, just
the actual geometry of the network's representations.

Run:
    /home/vybnz69/.venv/spark/bin/python3 experiment_D/run_D_v3.py
"""
import math
import os
import json
import time
import datetime
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Blackwell / GB10 backend tuning (identical to v2)
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# ---------------------------------------------------------------------------
# Config (identical to v2)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 256
BATCH_SIZE = 64
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.0
LEARNING_RATE = 1e-3
MAX_ITERS = 3000
WARMUP_ITERS = 100
LR_DECAY_ITERS = 3000
MIN_LR = 1e-4
EVAL_INTERVAL = 250
EVAL_ITERS = 200
SNAPSHOT_INTERVAL = 100  # geometric snapshot every N steps
LAMBDA_VALUES = [0.0, 0.5]  # baseline vs geometric
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------------------------------------------------------------------
# Data (identical to v2)
# ---------------------------------------------------------------------------
def get_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    fpath = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(fpath):
        print("Downloading tiny Shakespeare...")
        txt = requests.get(SHAKESPEARE_URL).text
        with open(fpath, "w") as f:
            f.write(txt)
    with open(fpath, "r") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    n = len(data)
    train_data = torch.tensor(encode(data[: int(n * 0.9)]), dtype=torch.long)
    val_data = torch.tensor(encode(data[int(n * 0.9) :]), dtype=torch.long)
    if DEVICE == "cuda":
        train_data = train_data.pin_memory()
        val_data = val_data.pin_memory()
    return train_data, val_data, len(chars), decode

def get_batch(split_data):
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([split_data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([split_data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    if DEVICE == "cuda":
        return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------------------
# Model (identical to v2)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CharGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.ModuleList([Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
        n_params = sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()
        print(f"Model params: {n_params/1e6:.2f}M")
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        layer_outputs = []
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, layer_outputs

# ---------------------------------------------------------------------------
# Arc-length regularizer (identical to v2)
# ---------------------------------------------------------------------------
def arc_length_loss(layer_outputs):
    total = 0.0
    for h in layer_outputs:
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1)
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(cos_sim)
        angle_var = angles.var(dim=-1).mean()
        total = total + angle_var
    return total / len(layer_outputs)

# ---------------------------------------------------------------------------
# Geometric snapshot — v3: NOW WITH RAW ACTIVATION VECTORS
# ---------------------------------------------------------------------------
@torch.no_grad()
def geometric_snapshot(model, xb, step_label=""):
    """Capture per-layer geometry AND raw activation centroids."""
    model.eval()
    with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
        _, _, layer_outputs = model(xb)
    snap = {}
    for i, h in enumerate(layer_outputs):
        h = h.float()  # (B, T, C) in float32

        # --- Summary statistics (same as v2) ---
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(cos_sim)
        norms = h.norm(dim=-1)

        # --- NEW: Raw activation centroid (384-dim) ---
        # Mean over batch and sequence → single representative vector
        centroid = h.mean(dim=(0, 1))  # (C,)
        # Also save the unit-normalized version for projective geometry
        centroid_norm = centroid / (centroid.norm() + 1e-12)

        snap[f"layer_{i}"] = {
            "mean_angle": angles.mean().item(),
            "std_angle": angles.std().item(),
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "angle_variance": angles.var(dim=-1).mean().item(),
            # v3 additions:
            "centroid": centroid.cpu().numpy().tolist(),
            "centroid_unit": centroid_norm.cpu().numpy().tolist(),
        }

    # Print geometry (same as v2)
    header = f"  [geometry @ {step_label}]"
    parts = []
    for i in range(N_LAYER):
        key = f"layer_{i}"
        a = snap[key]["mean_angle"]
        n = snap[key]["mean_norm"]
        v = snap[key]["angle_variance"]
        parts.append(f"L{i}(∠{a:.4f} ‖{n:.1f} σ²{v:.6f})")
    print(f"{header} {' '.join(parts)}", flush=True)

    model.train()
    return snap

# ---------------------------------------------------------------------------
# Training (identical to v2 except uses v3 snapshot)
# ---------------------------------------------------------------------------
def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(data)
            with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss, _ = model(xb, yb)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out

def train_run(lambda_geo, train_data, val_data, vocab_size):
    tag = f"lambda={lambda_geo}"
    print(f"\n{'='*60}")
    print(f"TRAINING: {tag}")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = CharGPT(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))

    snapshots = {}
    loss_log = []
    best_val = float("inf")

    t0 = time.time()
    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = get_batch(train_data)

        with torch.amp.autocast("cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
            logits, ce_loss, layer_outputs = model(xb, yb)
            if lambda_geo > 0:
                geo_loss = arc_length_loss(layer_outputs)
                loss = ce_loss + lambda_geo * geo_loss
            else:
                geo_loss = torch.tensor(0.0, device=DEVICE)
                loss = ce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % 10 == 0:
            dt = time.time() - t0
            geo_val = geo_loss.item() if lambda_geo > 0 else 0.0
            print(
                f"  step {it:5d} | ce {ce_loss.item():.4f} | "
                f"geo {geo_val:.6f} | lr {lr:.2e} | {dt:.1f}s",
                flush=True,
            )
            t0 = time.time()

        if it % SNAPSHOT_INTERVAL == 0:
            snap = geometric_snapshot(model, xb, step_label=f"step {it} ({tag})")
            snapshots[str(it)] = snap

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"  [eval] step {it} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}",
                flush=True,
            )
            loss_log.append({"step": it, **losses})
            if losses["val"] < best_val:
                best_val = losses["val"]

    # Final snapshot
    xb, _ = get_batch(val_data)
    final_snap = geometric_snapshot(model, xb, step_label=f"FINAL ({tag})")
    snapshots["final"] = final_snap

    return {
        "lambda_geo": lambda_geo,
        "best_val_loss": best_val,
        "loss_log": loss_log,
        "snapshots": snapshots,
        "final_snapshot": final_snap,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT D v3: Geometric Training — with raw activations")
    print("Identical to v2 but saves 384-dim centroids for Experiment E")
    print("=" * 60)
    print(f"Time: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.1f} GB")
    print(f"Torch: {torch.__version__}")
    print(f"Dtype: {DTYPE}")
    print(f"Model: {N_LAYER}L, {N_HEAD}H, {N_EMBD}E, {MAX_ITERS} iters")
    print(f"Lambda values: {LAMBDA_VALUES}")
    print(f"Snapshot interval: every {SNAPSHOT_INTERVAL} steps")

    train_data, val_data, vocab_size, decode = get_data()
    print(f"Vocab size: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")

    results = []
    for lam in LAMBDA_VALUES:
        result = train_run(lam, train_data, val_data, vocab_size)
        results.append(result)

    # Save results (v3 format with raw centroids)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "D_v3",
        "description": "Geometric training with raw 384-dim activation centroids for Experiment E QGT analysis",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
            "lambda_values": LAMBDA_VALUES, "dtype": str(DTYPE),
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "activation_dim": N_EMBD,
            "note": "centroid = mean over (batch, seq) per layer; centroid_unit = L2-normalized",
        },
        "runs": results,
    }
    outpath = os.path.join(RESULTS_DIR, "experiment_D_v3_result.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    # Quick sanity check: verify centroids are present and 384-dim
    for run in results:
        lam = run["lambda_geo"]
        snap0 = run["snapshots"].get("0", {})
        l0 = snap0.get("layer_0", {})
        centroid = l0.get("centroid", [])
        print(f"  lambda={lam}: snapshot 0, layer 0 centroid dim = {len(centroid)}")

if __name__ == "__main__":
    main()
