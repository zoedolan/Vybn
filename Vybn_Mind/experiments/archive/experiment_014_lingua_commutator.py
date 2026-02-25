"""
experiment_014_lingua_commutator.py

First real training experiment for VybnLingua.

Goal: teach the differentiable language to predict the commutator
from manifold.py. If it succeeds, the codebook primitives will
encode the algebraic structure of Vybn's own memory interactions.

Setup:
  - Generate synthetic memory event pairs with known commutator values
  - Encode events as vectors (the input_state / working memory)
  - Train VybnLingua to induce programs that, when executed on
    event-pair memory, output the commutator value
  - Monitor: codebook geometry, commutator magnitudes, program structure

This is the moment the language encounters real structure.

Feb 24, 2026 — Zoe & Vybn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'lingua'))

from vybn_lingua import VybnLingua


# ──────────────────────────────────────────────────
# 1. SYNTHETIC COMMUTATOR DATA
# ──────────────────────────────────────────────────
# Mirrors the commutator function from manifold.py
# but produces vectorized training data

def make_event_vector(event: dict, dim: int = 128) -> torch.Tensor:
    """
    Encode a memory event as a vector.
    Uses a simple but deterministic encoding:
    - Timestamp → sinusoidal position encoding
    - Content → bag-of-characters hash spread across dimensions
    - Event type → one-hot-ish signal in specific dimensions
    - Source → another hash signal
    """
    vec = torch.zeros(dim)

    # Timestamp: sinusoidal encoding (first 32 dims)
    ts = event.get('timestamp', 0.0)
    for i in range(16):
        freq = 1.0 / (10000 ** (2 * i / 32))
        vec[2 * i] = np.sin(ts * freq)
        vec[2 * i + 1] = np.cos(ts * freq)

    # Content: character hash spread (dims 32-95)
    content = event.get('content', '')
    for i, ch in enumerate(content[:64]):
        idx = 32 + (ord(ch) * (i + 1)) % 64
        vec[idx] += 0.1

    # Event type: signal in dims 96-111
    etype = event.get('event_type', 'unknown')
    type_hash = sum(ord(c) for c in etype)
    for i in range(16):
        vec[96 + i] = np.sin(type_hash * (i + 1) * 0.1)

    # Source: signal in dims 112-127
    source = event.get('source', 'unknown')
    src_hash = sum(ord(c) for c in source)
    for i in range(16):
        vec[112 + i] = np.cos(src_hash * (i + 1) * 0.1)

    return vec


def commutator(event_A: dict, event_B: dict) -> float:
    """
    Replica of manifold.py commutator for data generation.
    """
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


def generate_training_data(num_samples: int = 1000, dim: int = 128):
    """
    Generate pairs of synthetic memory events with known commutator values.
    """
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

    specs = []
    input_states = []
    targets = []

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

        # Working memory: the two event vectors plus interaction terms
        memory = torch.stack([vec_a, vec_b, vec_a * vec_b, vec_a - vec_b])

        # Spec: what should the program compute?
        spec = (vec_a + vec_b) / 2.0

        # Target: commutator value encoded as a direction vector
        comm_val = commutator(event_a, event_b)
        target_vec = torch.zeros(dim)
        target_vec[0] = comm_val
        for i in range(1, min(8, dim)):
            target_vec[i] = comm_val * np.sin(i * 0.5)

        specs.append(spec)
        input_states.append(memory)
        targets.append(target_vec)

    return (
        torch.stack(specs),
        torch.stack(input_states),
        torch.stack(targets),
    )


# ──────────────────────────────────────────────────
# 2. TRAINING LOOP
# ──────────────────────────────────────────────────

def train(epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
          geo_weight: float = 0.1, log_every: int = 10):
    """
    Train VybnLingua on the commutator task.
    """
    DIM = 128
    lingua = VybnLingua(num_primitives=64, dim=DIM, max_program_len=16)
    optimizer = torch.optim.Adam(lingua.parameters(), lr=lr)

    print("Generating training data...")
    specs, states, targets = generate_training_data(num_samples=2000, dim=DIM)

    n_train = 1600
    train_specs, val_specs = specs[:n_train], specs[n_train:]
    train_states, val_states = states[:n_train], states[n_train:]
    train_targets, val_targets = targets[:n_train], targets[n_train:]

    print(f"Training: {n_train} samples, Validation: {len(val_specs)} samples")
    print(f"Parameters: {sum(p.numel() for p in lingua.parameters()):,}")
    print(f"Geometric weight: {geo_weight}")
    print()

    history = []

    for epoch in range(epochs):
        lingua.train()
        lingua.temperature = max(0.5, 2.0 * (1 - epoch / epochs))

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        epoch_geo = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_spec = train_specs[batch_idx]
            batch_state = train_states[batch_idx]
            batch_target = train_targets[batch_idx]

            result = lingua(batch_spec, batch_state)

            task_loss = F.mse_loss(result['output'], batch_target)
            geo_loss = result['geometric_loss']
            total_loss = task_loss + geo_weight * geo_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += task_loss.item()
            epoch_geo += geo_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_geo = epoch_geo / n_batches

        if (epoch + 1) % log_every == 0 or epoch == 0:
            lingua.eval()
            with torch.no_grad():
                val_result = lingua(val_specs, val_states)
                val_loss = F.mse_loss(val_result['output'], val_targets).item()

                comm_magnitudes = []
                for a, b in [(0, 1), (10, 20), (30, 31), (5, 50)]:
                    cm = lingua.commutator_test(a, b, val_states[:1])
                    comm_magnitudes.append(cm)
                avg_comm = np.mean(comm_magnitudes)

                programs = val_result['program']
                unique_prims = len(torch.unique(programs))

            record = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'geo_loss': avg_geo,
                'temperature': lingua.temperature,
                'avg_commutator': avg_comm,
                'unique_primitives': unique_prims,
            }
            history.append(record)

            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {avg_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Geo: {avg_geo:.4f} | Temp: {lingua.temperature:.2f} | "
                  f"Comm: {avg_comm:.4f} | Prims: {unique_prims}/64")

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'geo_weight': geo_weight,
            'num_primitives': 64,
            'dim': DIM,
            'program_len': 16,
        },
        'history': history,
        'final_commutator_magnitudes': {
            f'[{a},{b}]': lingua.commutator_test(a, b, val_states[:1])
            for a, b in [(0,1), (10,20), (30,31), (5,50), (0,63), (15,48)]
        },
    }

    out_path = Path(__file__).parent / 'lingua_commutator_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    model_path = Path(__file__).parent.parent / 'lingua' / 'trained_lingua.pt'
    torch.save(lingua.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return lingua, history


if __name__ == '__main__':
    lingua, history = train(epochs=100, batch_size=32, lr=1e-3, geo_weight=0.1)
