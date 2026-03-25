#!/usr/bin/env python3
"""
Experiment E.4: SGP evolution across training checkpoints.

Rationale: The closure bundle in E.1 found c₁=0 because real-valued centroids
kill the Berry phase — overlaps are always positive, so discrete phase is 
identically zero. This is a measurement limitation, not a topological absence.

Fix: Use the complex-valued projective state machinery from holonomy_topology_probe.py
(R^d → C^{d/2} via to_complex(), then Pancharatnam phase) to measure the sign of
geometric phase (SGP) per concept class at each training checkpoint.

If the geometric run causes SGP sign flips that baseline doesn't → nontrivial topology.
If SGP signs are identical across all checkpoints for both runs → topology genuinely 
absent at this scale → write the metric paper.

Steps:
  1. Re-run D_v3 training but save model state_dicts at each snapshot
  2. At each checkpoint, load model, run concept-class prompts through it
  3. Measure Pancharatnam phase per concept class per layer pair
  4. Track SGP sign evolution

Author: Vybn, March 23 2026
"""

import torch
import torch.nn as nn
import numpy as np
import cmath
import json
import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ─── Phase 1: Training with model checkpoints ───────────────────────────

# Inline the D_v3 training code (tiny GPT-2) but save model checkpoints
BLOCK_SIZE = 256
BATCH_SIZE = 64
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
MAX_ITERS = 3000
LR = 1e-3
SNAPSHOT_INTERVAL = 100
LAMBDA_VALUES = [0.0, 0.5]

# ─── Tiny GPT-2 model (from run_D_v3.py) ────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.n_head = N_HEAD
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        hiddens = [x.detach().cpu()]
        for block in self.blocks:
            x = block(x)
            hiddens.append(x.detach().cpu())
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, hiddens
        return logits, loss

    def get_hidden_states(self, idx):
        """Get hidden states from all layers for a batch."""
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        hiddens = [x]
        for block in self.blocks:
            x = block(x)
            hiddens.append(x)
        return hiddens  # list of (B, T, N_EMBD) tensors, len = N_LAYER + 1


# ─── Complex phase geometry (from holonomy_topology_probe.py) ────────────

def to_complex(real_vec):
    """R^d → C^{d/2}, normalized to unit vector in CP^{d/2-1}."""
    n = len(real_vec) // 2
    cs = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(cs)
    return cs / norm if norm > 1e-15 else cs

def pancharatnam_phase(states):
    """Holonomy of natural connection on CP^{n-1}."""
    n = len(states)
    if n < 3:
        return 0.0
    product = complex(1.0, 0.0)
    for k in range(n):
        inner = np.vdot(states[k], states[(k + 1) % n])
        if abs(inner) < 1e-15:
            return 0.0
        product *= inner / abs(inner)
    return cmath.phase(product)


# ─── Concept classes (same as holonomy_topology_probe.py) ────────────────

CONCEPT_CLASSES = {
    "concrete_transformation": [
        "Two doubled is four",
        "Three tripled is nine",
        "Five plus five equals ten",
        "Ten halved is five",
    ],
    "abstract_epistemic": [
        "The truth of the matter remains uncertain",
        "Knowledge requires justified true belief",
        "The distinction between correlation and causation is subtle",
        "The evidence is consistent with multiple hypotheses",
    ],
    "temporal": [
        "Yesterday was quiet but tomorrow will be different",
        "The seasons change and eventually winter arrives again",
        "Before the meeting she had already decided",
        "Time passed slowly in the waiting room",
    ],
    "spatial_physical": [
        "The ball rolled down the hill and stopped at the bottom",
        "She walked through the door and turned left down the hallway",
        "The river flows from the mountains to the sea",
        "He stacked the books on top of each other on the shelf",
    ],
    "emotional_social": [
        "She felt a surge of pride watching her child succeed",
        "The betrayal left a wound that took years to heal",
        "They laughed together and for a moment nothing else mattered",
        "His anger dissolved into something closer to sadness",
    ],
    "self_referential": [
        "This sentence refers to itself",
        "I am thinking about the fact that I am thinking",
        "The model generates text about generating text",
        "Awareness of awareness is a recursive process",
    ],
}


# ─── Measure SGP for a model checkpoint ──────────────────────────────────

def measure_sgp_at_checkpoint(model, encode_fn, vocab_size, device, layer_pairs=None):
    """
    For each concept class and layer pair, measure:
    - Pancharatnam phase (complex-valued, from R^d → C^{d/2} lift)
    - Sign of geometric phase (SGP)
    - Phase magnitude
    
    Returns dict[class_name][layer_pair_str] = {mean_phase, sgp_sign, phases, ...}
    """
    if layer_pairs is None:
        # For 6-layer model: pairs that span different depths
        layer_pairs = [(0, 3), (0, 6), (2, 5), (3, 6)]
    
    model.eval()
    results = {}
    
    for class_name, prompts in CONCEPT_CLASSES.items():
        class_results = {}
        for in_l, out_l in layer_pairs:
            if out_l > N_LAYER:
                continue
            lp_key = f"L{in_l}->L{out_l}"
            phases = []
            
            for prompt in prompts:
                # Char-level encode (matching training tokenization)
                tokens = encode_fn(prompt)
                if len(tokens) > BLOCK_SIZE:
                    tokens = tokens[:BLOCK_SIZE]
                if len(tokens) < 3:
                    continue  # need at least 3 tokens for phase
                idx = torch.tensor([tokens], device=device)
                
                with torch.no_grad():
                    hiddens = model.get_hidden_states(idx)
                
                h_in = hiddens[in_l][0].cpu().numpy()   # (T, N_EMBD)
                h_out = hiddens[out_l][0].cpu().numpy()  # (T, N_EMBD)
                
                # Lift to complex projective space
                in_states = [to_complex(h_in[i]) for i in range(h_in.shape[0])]
                out_states = [to_complex(h_out[i]) for i in range(h_out.shape[0])]
                
                # Interleaved trajectory for differential phase
                interleaved = []
                for i_s, o_s in zip(in_states, out_states):
                    interleaved.append(i_s)
                    interleaved.append(o_s)
                
                ip = pancharatnam_phase(np.array(in_states))
                tp = pancharatnam_phase(np.array(interleaved))
                diff_phase = tp - ip
                phases.append(diff_phase)
            
            mean_phase = float(np.mean(phases))
            class_results[lp_key] = {
                "mean_phase_rad": mean_phase,
                "mean_phase_deg": float(np.degrees(mean_phase)),
                "sgp_sign": int(np.sign(mean_phase)) if abs(mean_phase) > 1e-10 else 0,
                "std_phase_rad": float(np.std(phases)),
                "individual_phases_rad": [float(p) for p in phases],
                "n_prompts": len(phases),
            }
        
        results[class_name] = class_results
    
    return results


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training {len(LAMBDA_VALUES)} runs, {MAX_ITERS} steps each")
    print(f"Saving model checkpoints every {SNAPSHOT_INTERVAL} steps")
    print(f"Measuring SGP for {len(CONCEPT_CLASSES)} concept classes")
    print()
    
    # Using char-level encoding (same as training)
    
    # Load training data (same as D_v3)
    data_path = Path(__file__).parent.parent / "experiment_D" / "data" / "input.txt"
    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        print("Downloading Shakespeare...")
        import urllib.request
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            str(data_path)
        )
    
    with open(data_path) as f:
        text = f.read()
    
    # Character-level tokenization (same as D_v3)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix]).to(device)
        y = torch.stack([d[i+1:i+1+BLOCK_SIZE] for i in ix]).to(device)
        return x, y
    
    # ─── Geometric loss (from D_v3) ─────────────────────────────────
    def geometric_loss(model, xb, base_loss, lam):
        if lam == 0:
            return base_loss, {}
        _, _, hiddens = model(xb, return_hidden=True)
        geo_penalty = torch.tensor(0.0, device=device)
        for layer_h in hiddens[1:]:  # skip embedding
            h = layer_h.to(device)
            norms = h.norm(dim=-1)
            norm_var = norms.var()
            cos_sim = torch.nn.functional.cosine_similarity(
                h[:, :-1, :].reshape(-1, N_EMBD),
                h[:, 1:, :].reshape(-1, N_EMBD),
                dim=1
            )
            angle_var = cos_sim.var()
            geo_penalty = geo_penalty + norm_var + angle_var
        total = base_loss + lam * geo_penalty
        return total, {"geo_penalty": geo_penalty.item()}
    
    # ─── Run training + SGP measurement ─────────────────────────────
    all_results = []
    checkpoint_dir = tempfile.mkdtemp(prefix="e4_checkpoints_")
    print(f"Checkpoint dir: {checkpoint_dir}")
    
    for lam in LAMBDA_VALUES:
        tag = f"baseline" if lam == 0.0 else f"geo_{lam}"
        print(f"\n{'='*60}")
        print(f"Training: {tag} (lambda={lam})")
        print(f"{'='*60}")
        
        torch.manual_seed(42)
        model = TinyGPT(vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        sgp_timeline = {}  # step -> SGP measurements
        loss_log = []
        
        # Measure SGP at initialization
        print(f"  Step 0: measuring SGP...", flush=True)
        sgp_timeline["0"] = measure_sgp_at_checkpoint(model, encode, vocab_size, device)
        
        for it in range(1, MAX_ITERS + 1):
            model.train()
            xb, yb = get_batch("train")
            _, base_loss = model(xb, targets=yb)
            loss, geo_info = geometric_loss(model, xb, base_loss, lam)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if it % 100 == 0:
                # Eval
                model.eval()
                with torch.no_grad():
                    xv, yv = get_batch("val")
                    _, val_loss = model(xv, targets=yv)
                
                loss_log.append({
                    "step": it,
                    "train_loss": base_loss.item(),
                    "val_loss": val_loss.item(),
                    "geo_penalty": geo_info.get("geo_penalty", 0.0),
                })
                print(f"  Step {it}: train={base_loss.item():.4f} val={val_loss.item():.4f}", end="")
                
                if it % SNAPSHOT_INTERVAL == 0:
                    # Measure SGP
                    print(" [SGP]", end="", flush=True)
                    sgp_timeline[str(it)] = measure_sgp_at_checkpoint(model, encode, vocab_size, device)
                
                print()
        
        # Final SGP measurement
        print(f"  Final: measuring SGP...", flush=True)
        sgp_timeline["final"] = measure_sgp_at_checkpoint(model, encode, vocab_size, device)
        
        all_results.append({
            "tag": tag,
            "lambda_geo": lam,
            "loss_log": loss_log,
            "sgp_timeline": sgp_timeline,
            "best_val_loss": min(e["val_loss"] for e in loss_log) if loss_log else None,
        })
    
    # Cleanup checkpoint dir
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    # ─── Analysis: SGP sign evolution ────────────────────────────────
    print(f"\n{'='*60}")
    print("ANALYSIS: SGP SIGN EVOLUTION")
    print(f"{'='*60}")
    
    analysis = {}
    for run in all_results:
        tag = run["tag"]
        timeline = run["sgp_timeline"]
        steps = sorted([int(s) if s != "final" else 99999 for s in timeline.keys()])
        
        sign_changes = {}
        for class_name in CONCEPT_CLASSES:
            for lp_key in timeline["0"].get(class_name, {}):
                trace_key = f"{class_name}/{lp_key}"
                signs = []
                for step in steps:
                    step_key = str(step) if step != 99999 else "final"
                    sgp = timeline[step_key].get(class_name, {}).get(lp_key, {})
                    signs.append(sgp.get("sgp_sign", 0))
                
                # Count sign flips
                flips = sum(1 for i in range(1, len(signs)) 
                           if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0)
                
                sign_changes[trace_key] = {
                    "signs": signs,
                    "n_flips": flips,
                    "initial_sign": signs[0] if signs else 0,
                    "final_sign": signs[-1] if signs else 0,
                    "sign_changed": signs[0] != signs[-1] if len(signs) >= 2 else False,
                }
        
        analysis[tag] = sign_changes
        
        # Print summary
        total_flips = sum(v["n_flips"] for v in sign_changes.values())
        changed = sum(1 for v in sign_changes.values() if v["sign_changed"])
        total_traces = len(sign_changes)
        print(f"\n  {tag}:")
        print(f"    Total SGP traces: {total_traces}")
        print(f"    Total sign flips: {total_flips}")
        print(f"    Traces where initial ≠ final sign: {changed}/{total_traces}")
        
        # Detail the flips
        for trace_key, info in sign_changes.items():
            if info["n_flips"] > 0:
                print(f"    FLIP: {trace_key}: {info['signs'][:5]}...{info['signs'][-3:]}")
    
    # ─── Verdict ─────────────────────────────────────────────────────
    baseline_flips = sum(v["n_flips"] for v in analysis.get("baseline", {}).values())
    geo_flips = sum(v["n_flips"] for v in analysis.get("geo_0.5", {}).values())
    
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    
    if geo_flips > baseline_flips + 2:  # meaningful excess
        verdict = "NONTRIVIAL: Geometric training induces SGP sign flips absent in baseline"
        print(f"  {verdict}")
        print(f"  Baseline flips: {baseline_flips}")
        print(f"  Geometric flips: {geo_flips}")
        print(f"  → Chern class potentially nonzero. Proceed to full bundle measurement.")
    elif geo_flips == 0 and baseline_flips == 0:
        verdict = "NULL: No sign flips in either run. Topology genuinely absent at this scale."
        print(f"  {verdict}")
        print(f"  → Write the metric paper (Fubini-Study compression + generalization).")
    else:
        verdict = f"AMBIGUOUS: baseline={baseline_flips}, geometric={geo_flips}. Need more data."
        print(f"  {verdict}")
    
    # ─── Save results ────────────────────────────────────────────────
    output = {
        "experiment": "E.4",
        "description": "SGP evolution across training checkpoints using complex-valued projective states",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rationale": "E.1 closure bundle found c1=0 because real centroids kill phase. This uses R^d->C^{d/2} lift + Pancharatnam.",
        "config": {
            "n_layer": N_LAYER, "n_embd": N_EMBD, "max_iters": MAX_ITERS,
            "snapshot_interval": SNAPSHOT_INTERVAL, "lambda_values": LAMBDA_VALUES,
            "concept_classes": list(CONCEPT_CLASSES.keys()),
            "layer_pairs": [[0,3],[0,6],[2,5],[3,6]],
        },
        "runs": all_results,
        "analysis": analysis,
        "verdict": verdict,
    }
    
    outpath = Path(__file__).parent / "results" / "experiment_E4_sgp_evolution.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    return output


if __name__ == "__main__":
    main()
