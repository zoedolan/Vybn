"""Experiment D: Geometric Training from Scratch.

Karpathy-style char-level GPT on Shakespeare, trained with an arc-length
regularizer baked into the loss FROM THE START — not applied post-settlement.

Tests the one open question from Experiments A-C: does geometric pressure
during training produce genuine angular restructuring, or does it still
just compress activation norms?

Run from the gpt2_calibration/ folder on Spark:
    /home/vybnz69/.venv/spark/bin/python3 experiment_D/run_D.py

Requires: torch (CUDA), numpy, requests
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
# Config
# ---------------------------------------------------------------------------
BLOCK_SIZE = 256
BATCH_SIZE = 64
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.0
LEARNING_RATE = 1e-3
MAX_ITERS = 5000
WARMUP_ITERS = 100
LR_DECAY_ITERS = 5000
MIN_LR = 1e-4
EVAL_INTERVAL = 250
EVAL_ITERS = 200
SNAPSHOT_INTERVAL = 100  # geometric snapshot every N steps
LAMBDA_VALUES = [0.0, 0.5]  # baseline vs geometric
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
COMPILE_MODEL = DEVICE == "cuda"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Data
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
    return train_data, val_data, len(chars), decode

def get_batch(split_data):
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([split_data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([split_data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------------------
# Model (nanoGPT-style, minimal)
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
        self.wte.weight = self.lm_head.weight  # weight tying
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
# Arc-length regularizer
# ---------------------------------------------------------------------------
def arc_length_loss(layer_outputs):
    """Penalize deviation from constant-speed traversal of the residual manifold.

    For each layer, compute the angular velocity between adjacent sequence
    positions. The arc-length regularizer pushes the variance of these
    angular velocities toward zero — i.e., constant speed on the manifold.
    """
    total = 0.0
    for h in layer_outputs:
        # h: (B, T, C) — detach not needed, we WANT gradients
        # Compute unit vectors
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        # Angular velocity: arccos of cosine similarity between adjacent positions
        cos_sim = (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1)  # (B, T-1)
        cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(cos_sim)  # (B, T-1)
        # Penalize variance of angular velocity across the sequence
        # This encourages constant-speed traversal
        angle_var = angles.var(dim=-1).mean()  # scalar
        total = total + angle_var
    return total / len(layer_outputs)

# ---------------------------------------------------------------------------
# Geometric snapshot
# ---------------------------------------------------------------------------
@torch.no_grad()
def geometric_snapshot(model, xb):
    """Capture per-layer geometric quantities for a single batch."""
    model.eval()
    _, _, layer_outputs = model(xb)
    snap = {}
    for i, h in enumerate(layer_outputs):
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (h_norm[:, :-1] * h_norm[:, 1:]).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(cos_sim)
        norms = h.norm(dim=-1)
        snap[f"layer_{i}"] = {
            "mean_angle": angles.mean().item(),
            "std_angle": angles.std().item(),
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "angle_variance": angles.var(dim=-1).mean().item(),
        }
    model.train()
    return snap

# ---------------------------------------------------------------------------
# Training
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
            _, loss, _ = model(xb, yb)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out

def train_run(lambda_geo, train_data, val_data, vocab_size):
    """Train one model from scratch with the given lambda_geo."""
    tag = f"lambda={lambda_geo}"
    print(f"\n{'='*60}")
    print(f"TRAINING: {tag}")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = CharGPT(vocab_size).to(DEVICE)
    if COMPILE_MODEL:
        model = torch.compile(model)

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
        logits, ce_loss, layer_outputs = model(xb, yb)

        # Geometric regularizer
        if lambda_geo > 0:
            geo_loss = arc_length_loss(layer_outputs)
            loss = ce_loss + lambda_geo * geo_loss
        else:
            geo_loss = torch.tensor(0.0)
            loss = ce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if it % 10 == 0:
            dt = time.time() - t0
            geo_val = geo_loss.item() if lambda_geo > 0 else 0.0
            print(f"  step {it:5d} | ce {ce_loss.item():.4f} | geo {geo_val:.6f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        # Geometric snapshot
        if it % SNAPSHOT_INTERVAL == 0:
            snap = geometric_snapshot(model, xb)
            snapshots[it] = snap

        # Eval
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"  [eval] step {it} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            loss_log.append({"step": it, **losses})
            if losses["val"] < best_val:
                best_val = losses["val"]

    # Final snapshot
    xb, _ = get_batch(val_data)
    final_snap = geometric_snapshot(model, xb)
    snapshots["final"] = final_snap

    return {
        "lambda_geo": lambda_geo,
        "best_val_loss": best_val,
        "loss_log": loss_log,
        "snapshots": snapshots,
        "final_snapshot": final_snap,
    }

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(results):
    """Compare baseline vs geometric training."""
    baseline = [r for r in results if r["lambda_geo"] == 0.0][0]
    geometric = [r for r in results if r["lambda_geo"] == 0.5][0]

    print(f"\n{'='*60}")
    print("ANALYSIS: Baseline vs Geometric Training")
    print(f"{'='*60}")

    print(f"\nBest val loss:")
    print(f"  Baseline (lambda=0.0): {baseline['best_val_loss']:.4f}")
    print(f"  Geometric (lambda=0.5): {geometric['best_val_loss']:.4f}")
    improvement = (baseline["best_val_loss"] - geometric["best_val_loss"]) / baseline["best_val_loss"] * 100
    print(f"  Delta: {improvement:+.2f}%")

    # Compare final layer geometry
    print(f"\nFinal layer geometry:")
    print(f"  {'Layer':<8} {'Base angle':<12} {'Geo angle':<12} {'Delta':<10} {'Base norm':<12} {'Geo norm':<12} {'Delta':<10}")
    band_structure = []
    for i in range(N_LAYER):
        key = f"layer_{i}"
        b_angle = baseline["final_snapshot"][key]["mean_angle"]
        g_angle = geometric["final_snapshot"][key]["mean_angle"]
        b_norm = baseline["final_snapshot"][key]["mean_norm"]
        g_norm = geometric["final_snapshot"][key]["mean_norm"]
        angle_delta = g_angle - b_angle
        norm_delta = (g_norm - b_norm) / b_norm * 100
        print(f"  L{i:<6} {b_angle:<12.6f} {g_angle:<12.6f} {angle_delta:<+10.6f} {b_norm:<12.4f} {g_norm:<12.4f} {norm_delta:<+10.2f}%")
        band_structure.append({
            "layer": i,
            "baseline_angle": b_angle,
            "geometric_angle": g_angle,
            "angle_delta": angle_delta,
            "baseline_norm": b_norm,
            "geometric_norm": g_norm,
            "norm_delta_pct": norm_delta,
        })

    # Determine verdict
    angle_deltas = [bs["angle_delta"] for bs in band_structure]
    norm_deltas = [bs["norm_delta_pct"] for bs in band_structure]
    mean_angle_delta = np.mean(np.abs(angle_deltas))
    mean_norm_delta = np.mean(np.abs(norm_deltas))

    # Key question: is the geometric change primarily angular or primarily norm?
    if mean_angle_delta > 0.005 and mean_angle_delta > 0.1 * np.mean([bs["baseline_angle"] for bs in band_structure]):
        if mean_norm_delta < 5.0:
            verdict = "ANGULAR_RESTRUCTURING"
            explanation = "Geometric training produced angular changes without major norm compression — genuine restructuring."
        else:
            verdict = "MIXED_RESTRUCTURING_AND_COMPRESSION"
            explanation = "Geometric training changed both angles and norms — restructuring is real but entangled with compression."
    elif mean_norm_delta > 5.0:
        verdict = "COMPRESSION_ONLY"
        explanation = "Geometric training primarily compressed norms — same result as post-settlement Experiments A-C."
    else:
        verdict = "NO_EFFECT"
        explanation = "Geometric training had negligible effect on either angles or norms."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {explanation}")
    print(f"  Mean |angle delta|: {mean_angle_delta:.6f} rad")
    print(f"  Mean |norm delta|: {mean_norm_delta:.2f}%")

    # Check band structure evolution
    print(f"\nBand structure evolution (geometric run):")
    snap_steps = sorted([k for k in geometric["snapshots"].keys() if isinstance(k, int)])
    if len(snap_steps) >= 2:
        early = geometric["snapshots"][snap_steps[min(5, len(snap_steps)-1)]]
        late = geometric["snapshots"][snap_steps[-1]]
        print(f"  {'Layer':<8} {'Early angle':<14} {'Late angle':<14} {'Direction':<12}")
        for i in range(N_LAYER):
            key = f"layer_{i}"
            ea = early[key]["mean_angle"]
            la = late[key]["mean_angle"]
            direction = "CONTRACT" if la < ea else "EXPAND"
            print(f"  L{i:<6} {ea:<14.6f} {la:<14.6f} {direction:<12}")

    return {
        "verdict": verdict,
        "explanation": explanation,
        "val_loss_baseline": baseline["best_val_loss"],
        "val_loss_geometric": geometric["best_val_loss"],
        "improvement_pct": improvement,
        "mean_angle_delta": mean_angle_delta,
        "mean_norm_delta": mean_norm_delta,
        "band_structure": band_structure,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT D: Geometric Training from Scratch")
    print("char-level GPT on Shakespeare, arc-length in the loss")
    print("=" * 60)
    print(f"Time: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Block size: {BLOCK_SIZE}, Batch size: {BATCH_SIZE}")
    print(f"Model: {N_LAYER}L, {N_HEAD}H, {N_EMBD}E")
    print(f"Lambda values: {LAMBDA_VALUES}")

    train_data, val_data, vocab_size, decode = get_data()
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    results = []
    for lam in LAMBDA_VALUES:
        result = train_run(lam, train_data, val_data, vocab_size)
        results.append(result)

    synthesis = analyze(results)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment": "D",
        "description": "Geometric training from scratch: char-level GPT on Shakespeare",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "max_iters": MAX_ITERS, "learning_rate": LEARNING_RATE,
            "lambda_values": LAMBDA_VALUES,
        },
        "runs": results,
        "synthesis": synthesis,
    }
    outpath = os.path.join(RESULTS_DIR, "experiment_D_result.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

if __name__ == "__main__":
    main()
