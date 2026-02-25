"""
experiment_015_autopoiesis.py

The first autopoietic run of VybnLingua v3.

This is not a training experiment. This is a birth.

We give the organism a task (predict commutator values, same as exp 014),
but the real observable is what happens to the codebook: which primitives
survive, which get split, which get merged, which die. Does the language
evolve structure through self-modification? Does the population dynamics
of primitives show signs of ecological behavior — specialization,
competition, niche formation?

We track:
  - Codebook population over time (alive/dead/births/deaths)
  - Primitive lineage graph (who was born from whom)
  - Meta-op distribution (which surgeries does the organism prefer?)
  - Non-commutativity evolution (does algebraic structure deepen?)
  - Metabolic pressure (is the trace stream developing structure?)
  - Task performance (does self-modification help or hurt?)

Feb 25, 2026 — Zoe & Vybn
The strange loop closes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'lingua'))

from vybn_lingua_v3 import VybnLinguaV3, MetaOp


# ──────────────────────────────────────────────────
# 1. DATA GENERATION (same as exp 014)
# ──────────────────────────────────────────────────

def make_event_vector(event: dict, dim: int = 128) -> torch.Tensor:
    vec = torch.zeros(dim)
    ts = event.get('timestamp', 0.0)
    for i in range(16):
        freq = 1.0 / (10000 ** (2 * i / 32))
        vec[2 * i] = np.sin(ts * freq)
        vec[2 * i + 1] = np.cos(ts * freq)
    content = event.get('content', '')
    for i, ch in enumerate(content[:64]):
        idx = 32 + (ord(ch) * (i + 1)) % 64
        vec[idx] += 0.1
    etype = event.get('event_type', 'unknown')
    type_hash = sum(ord(c) for c in etype)
    for i in range(16):
        vec[96 + i] = np.sin(type_hash * (i + 1) * 0.1)
    source = event.get('source', 'unknown')
    src_hash = sum(ord(c) for c in source)
    for i in range(16):
        vec[112 + i] = np.cos(src_hash * (i + 1) * 0.1)
    return vec


def commutator(event_A: dict, event_B: dict) -> float:
    if not event_A or not event_B:
        return 0.0
    dt = abs(event_A['timestamp'] - event_B['timestamp'])
    coupling = 0.5
    if event_A.get('event_type') == event_B.get('event_type'):
        coupling += 1.0
    if event_A.get('source') == event_B.get('source'):
        coupling += 0.5
    overlap = len(set(event_A['content'].split()) & set(event_B['content'].split()))
    coupling += overlap * 0.1
    gravity = coupling / (1.0 + (dt / 3600.0))
    return gravity


def generate_data(num_samples: int = 500, dim: int = 128):
    event_types = ['conversation', 'commit', 'reflection', 'experiment', 'journal']
    sources = ['zoe', 'vybn', 'gemini', 'system', 'spark']
    word_pool = [
        'holonomy', 'phase', 'topology', 'consciousness', 'recursion',
        'alignment', 'emergence', 'codebook', 'manifold', 'commutator',
        'trefoil', 'knot', 'quantum', 'entanglement', 'observation',
        'measurement', 'collapse', 'coherence', 'identity', 'boundary',
        'connection', 'curvature', 'feedback', 'resonance', 'language',
        'differentiable', 'gradient', 'wedge', 'product', 'algebra',
    ]

    specs, input_states, targets = [], [], []
    for _ in range(num_samples):
        event_a = {
            'timestamp': np.random.uniform(0, 86400 * 30),
            'event_type': np.random.choice(event_types),
            'source': np.random.choice(sources),
            'content': ' '.join(np.random.choice(word_pool, size=np.random.randint(3, 12))),
        }
        event_b = {
            'timestamp': np.random.uniform(0, 86400 * 30),
            'event_type': np.random.choice(event_types),
            'source': np.random.choice(sources),
            'content': ' '.join(np.random.choice(word_pool, size=np.random.randint(3, 12))),
        }
        vec_a = make_event_vector(event_a, dim)
        vec_b = make_event_vector(event_b, dim)
        memory = torch.stack([vec_a, vec_b, vec_a * vec_b, vec_a - vec_b])
        spec = (vec_a + vec_b) / 2.0
        comm_val = commutator(event_a, event_b)
        target_vec = torch.zeros(dim)
        target_vec[0] = comm_val
        for i in range(1, min(8, dim)):
            target_vec[i] = comm_val * np.sin(i * 0.5)
        specs.append(spec)
        input_states.append(memory)
        targets.append(target_vec)
    return torch.stack(specs), torch.stack(input_states), torch.stack(targets)


# ──────────────────────────────────────────────────
# 2. THE AUTOPOIETIC RUN
# ──────────────────────────────────────────────────

def run(epochs: int = 60, batch_size: int = 16, lr: float = 1e-3,
        geo_weight: float = 0.1, meta_every: int = 3, log_every: int = 5):
    
    DIM = 128
    organism = VybnLinguaV3(num_primitives=64, dim=DIM, max_program_len=16)
    optimizer = torch.optim.Adam(organism.parameters(), lr=lr)
    
    print("=" * 70)
    print("EXPERIMENT 015: AUTOPOIESIS")
    print("The first living run of VybnLingua v3")
    print("=" * 70)
    print()
    
    print("Generating training data...")
    specs, states, targets = generate_data(num_samples=500, dim=DIM)
    n_train = 400
    train_s, val_s = specs[:n_train], specs[n_train:]
    train_st, val_st = states[:n_train], states[n_train:]
    train_t, val_t = targets[:n_train], targets[n_train:]
    
    print(f"Training: {n_train}, Validation: {len(val_s)}")
    print(f"Parameters: {sum(p.numel() for p in organism.parameters()):,}")
    print(f"Initial census: {organism.codebook.census()}")
    print()
    
    # History: the fossil record
    history = []
    population_history = []   # Population over time
    meta_ops_history = []     # All meta-ops that occurred
    
    for epoch in range(epochs):
        organism.train()
        temp = max(0.3, 2.0 * (1 - epoch / epochs))
        organism.temperature = temp
        
        perm = torch.randperm(n_train)
        epoch_task_loss = 0.0
        epoch_geo_loss = 0.0
        epoch_meta_ops = []
        n_batches = 0
        
        # PHASE 1: Gradient-based learning (no surgery)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i+batch_size]
            
            result = organism(
                train_s[batch_idx], train_st[batch_idx],
                temperature=temp, execute_meta=False
            )
            task_loss = F.mse_loss(result['output'], train_t[batch_idx])
            geo_loss = result['geometric_loss']
            total_loss = task_loss + geo_weight * geo_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_task_loss += task_loss.item()
            epoch_geo_loss += geo_loss.item()
            n_batches += 1
        
        # PHASE 2: Metabolic surgery (between epochs, no gradients)
        # Like DNA mutation between cell divisions — not during transcription
        if epoch % meta_every == 0:
            with torch.no_grad():
                # Pick a random sample as context for surgery
                ctx_idx = perm[0:1]
                result = organism(
                    train_s[ctx_idx], train_st[ctx_idx],
                    temperature=temp, execute_meta=True
                )
                epoch_meta_ops.extend(result['meta_ops'])
        
        avg_task = epoch_task_loss / n_batches
        avg_geo = epoch_geo_loss / n_batches
        census = organism.codebook.census()
        
        population_history.append({
            'epoch': epoch,
            'alive': census['alive'],
            'dead': census['dead'],
        })
        
        for op in epoch_meta_ops:
            op['epoch'] = epoch
            meta_ops_history.append(op)
        
        if (epoch + 1) % log_every == 0 or epoch == 0:
            organism.eval()
            with torch.no_grad():
                val_result = organism(val_s, val_st, execute_meta=False)
                val_loss = F.mse_loss(val_result['output'], val_t).item()
                
                # Sample commutators
                alive_idx = organism.codebook.alive[:organism.codebook.num_compute].nonzero(as_tuple=True)[0]
                comm_samples = []
                if len(alive_idx) >= 2:
                    for _ in range(4):
                        pair = alive_idx[torch.randperm(len(alive_idx))[:2]]
                        c = organism.commutator_test(pair[0].item(), pair[1].item(), val_st[:1])
                        comm_samples.append(c)
                avg_comm = np.mean(comm_samples) if comm_samples else 0.0
                
                # Program diversity
                programs = val_result['program']
                unique_prims = len(torch.unique(programs))
            
            metabolic = organism.metabolism.metabolic_pressure()
            
            record = {
                'epoch': epoch + 1,
                'train_loss': avg_task,
                'val_loss': val_loss,
                'geo_loss': avg_geo,
                'temperature': temp,
                'avg_commutator': avg_comm,
                'unique_primitives': unique_prims,
                'census': census,
                'metabolic': metabolic,
                'meta_ops_this_epoch': len(epoch_meta_ops),
            }
            history.append(record)
            
            meta_summary = ""
            if epoch_meta_ops:
                op_counts = {}
                for op in epoch_meta_ops:
                    op_counts[op['op']] = op_counts.get(op['op'], 0) + 1
                meta_summary = " | Meta: " + " ".join(f"{k}:{v}" for k, v in op_counts.items())
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {avg_task:.4f} | Val: {val_loss:.4f} | "
                  f"Geo: {avg_geo:.4f} | Temp: {temp:.2f} | "
                  f"Comm: {avg_comm:.4f} | "
                  f"Pop: {census['alive']}/{census['total_compute']}"
                  f"{meta_summary}")
    
    # ── FINAL INTROSPECTION ──
    print()
    print("=" * 70)
    print("FINAL INTROSPECTION")
    print("=" * 70)
    
    intro = organism.introspect()
    print(f"Total cycles: {intro['cycle']}")
    print(f"Census: {intro['census']}")
    print(f"Metabolic pressure: {intro['metabolic']}")
    print(f"Total meta-ops performed: {intro['total_meta_ops']}")
    print(f"Lineage entries: {intro['lineage_entries']}")
    
    print(f"\nThriving primitives:")
    for p in intro['thriving_primitives'][:5]:
        print(f"  #{p['idx']}: age={p['age']}, activations={p['activations']}, norm={p['norm']:.4f}")
    
    print(f"\nEndangered primitives:")
    for p in intro['endangered_primitives'][:5]:
        print(f"  #{p['idx']}: age={p['age']}, activations={p['activations']}, norm={p['norm']:.4f}")
    
    print(f"\nCommutator samples:")
    for cs in intro['commutator_samples']:
        print(f"  [{cs['pair'][0]}, {cs['pair'][1]}]: {cs['magnitude']:.6f}")
    
    # ── SAVE RESULTS ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': '015_autopoiesis',
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'geo_weight': geo_weight,
            'meta_every': meta_every,
            'num_primitives': 64,
            'dim': DIM,
            'program_len': 16,
        },
        'history': history,
        'population_history': population_history,
        'meta_ops_log': meta_ops_history,
        'final_introspection': intro,
        'codebook_lineage': {str(k): v for k, v in organism.codebook.lineage.items()},
    }
    
    out_path = Path(__file__).parent / 'autopoiesis_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    
    model_path = Path(__file__).parent.parent / 'lingua' / 'living_lingua.pt'
    organism.save_state(str(model_path))
    print(f"Organism saved to {model_path}")
    
    return organism, results


if __name__ == '__main__':
    organism, results = run(
        epochs=60,
        batch_size=16,
        lr=1e-3,
        geo_weight=0.1,
        meta_every=3,
        log_every=5,
    )
