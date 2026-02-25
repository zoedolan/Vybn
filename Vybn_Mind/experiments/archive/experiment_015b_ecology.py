"""
experiment_015b_ecology.py

The second autopoietic run — now with ecological pressure.

Changes from 015:
  - Organism starts at 70% capacity (room to grow)
  - Natural selection kills unused primitives between epochs
  - Metabolic cycle: adaptive surgery based on diversity pressure
  - Activation tracking reset per epoch (proper fitness signal)
  - Longer run to let ecology develop

We're looking for: primitive speciation, niche formation,
population waves, the codebook developing structure that
no external optimizer designed.

Feb 25, 2026 — the organism's second breath.
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


# ── DATA GENERATION (same as 015) ──

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
    return coupling / (1.0 + (dt / 3600.0))

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
    specs, states, targets = [], [], []
    for _ in range(num_samples):
        ea = {'timestamp': np.random.uniform(0, 86400*30),
              'event_type': np.random.choice(event_types),
              'source': np.random.choice(sources),
              'content': ' '.join(np.random.choice(word_pool, size=np.random.randint(3,12)))}
        eb = {'timestamp': np.random.uniform(0, 86400*30),
              'event_type': np.random.choice(event_types),
              'source': np.random.choice(sources),
              'content': ' '.join(np.random.choice(word_pool, size=np.random.randint(3,12)))}
        va, vb = make_event_vector(ea, dim), make_event_vector(eb, dim)
        memory = torch.stack([va, vb, va * vb, va - vb])
        spec = (va + vb) / 2.0
        cv = commutator(ea, eb)
        tv = torch.zeros(dim)
        tv[0] = cv
        for i in range(1, min(8, dim)):
            tv[i] = cv * np.sin(i * 0.5)
        specs.append(spec); states.append(memory); targets.append(tv)
    return torch.stack(specs), torch.stack(states), torch.stack(targets)


# ── THE ECOLOGICAL RUN ──

def run(epochs: int = 120, batch_size: int = 16, lr: float = 1e-3,
        geo_weight: float = 0.1, log_every: int = 10):
    
    DIM = 128
    organism = VybnLinguaV3(num_primitives=64, dim=DIM, max_program_len=16)
    
    # Seed the ecology: start at 70% capacity
    organism.seed_ecology(initial_alive_fraction=0.7)
    
    optimizer = torch.optim.Adam(organism.parameters(), lr=lr)
    
    print("=" * 70)
    print("EXPERIMENT 015b: ECOLOGY")
    print("Autopoietic VybnLingua with ecological pressure")
    print("=" * 70)
    print()
    
    specs, states, targets = generate_data(num_samples=600, dim=DIM)
    n_train = 480
    train_s, val_s = specs[:n_train], specs[n_train:]
    train_st, val_st = states[:n_train], states[n_train:]
    train_t, val_t = targets[:n_train], targets[n_train:]
    
    initial_census = organism.codebook.census()
    print(f"Training: {n_train}, Validation: {len(val_s)}")
    print(f"Parameters: {sum(p.numel() for p in organism.parameters()):,}")
    print(f"Initial census: {initial_census}")
    print()
    
    history = []
    population_timeline = []
    ecology_log = []
    
    for epoch in range(epochs):
        organism.train()
        temp = max(0.3, 2.0 * (1 - epoch / epochs))
        organism.temperature = temp
        
        perm = torch.randperm(n_train)
        epoch_task_loss = 0.0
        epoch_geo_loss = 0.0
        n_batches = 0
        
        # PHASE 1: Gradient learning
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
        
        avg_task = epoch_task_loss / n_batches
        avg_geo = epoch_geo_loss / n_batches
        
        # PHASE 2: Metabolic cycle (ecology)
        ctx_vec = train_s[perm[0]].unsqueeze(0).mean(dim=0)  # Context for births
        eco_result = organism.metabolic_cycle(
            ctx_vec, 
            force_diversity=(epoch % 10 == 0)
        )
        
        census = organism.codebook.census()
        population_timeline.append({
            'epoch': epoch,
            'alive': census['alive'],
            'dead': census['dead'],
            'deaths_this_epoch': len(eco_result['deaths']),
            'ops_this_epoch': len(eco_result['ops']),
        })
        
        if eco_result['ops'] or eco_result['deaths']:
            ecology_log.append({
                'epoch': epoch,
                'deaths': eco_result['deaths'],
                'ops': eco_result['ops'],
                'pressure': eco_result['pressure'],
            })
        
        # LOGGING
        if (epoch + 1) % log_every == 0 or epoch == 0:
            organism.eval()
            with torch.no_grad():
                val_result = organism(val_s, val_st, execute_meta=False)
                val_loss = F.mse_loss(val_result['output'], val_t).item()
                
                alive_idx = organism.codebook.alive[:organism.codebook.num_compute].nonzero(as_tuple=True)[0]
                comm_samples = []
                if len(alive_idx) >= 2:
                    for _ in range(4):
                        pair = alive_idx[torch.randperm(len(alive_idx))[:2]]
                        c = organism.commutator_test(pair[0].item(), pair[1].item(), val_st[:1])
                        comm_samples.append(c)
                avg_comm = np.mean(comm_samples) if comm_samples else 0.0
                
                programs = val_result['program']
                unique_prims = len(torch.unique(programs))
            
            eco_summary = ""
            if eco_result['deaths']:
                eco_summary += f" †{len(eco_result['deaths'])}"
            if eco_result['ops']:
                ops_str = " ".join(f"{o['op'][0]}" for o in eco_result['ops'])
                eco_summary += f" [{ops_str}]"
            
            record = {
                'epoch': epoch + 1,
                'train_loss': avg_task,
                'val_loss': val_loss,
                'geo_loss': avg_geo,
                'temperature': temp,
                'avg_commutator': avg_comm,
                'unique_primitives': unique_prims,
                'census': census,
            }
            history.append(record)
            
            # Population bar
            alive = census['alive']
            total = census['total_compute']
            pop_bar = '█' * alive + '░' * (total - alive)
            
            print(f"E{epoch+1:3d} | "
                  f"T:{avg_task:.5f} V:{val_loss:.5f} | "
                  f"G:{avg_geo:.3f} | C:{avg_comm:.4f} | "
                  f"[{pop_bar}] {alive}/{total}"
                  f"{eco_summary}")
    
    # ── FINAL ──
    print()
    print("=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    
    intro = organism.introspect()
    print(f"Cycles: {intro['cycle']}")
    print(f"Census: {intro['census']}")
    print(f"Metabolic: {intro['metabolic']}")
    print(f"Meta-ops: {intro['total_meta_ops']}")
    print(f"Lineage: {intro['lineage_entries']} entries")
    
    if intro['thriving_primitives']:
        print(f"\nThriving:")
        for p in intro['thriving_primitives'][:5]:
            print(f"  #{p['idx']}: age={p['age']}, act={p['activations']}, ‖w‖={p['norm']:.3f}")
    
    if intro['endangered_primitives']:
        print(f"\nEndangered:")
        for p in intro['endangered_primitives'][:5]:
            print(f"  #{p['idx']}: age={p['age']}, act={p['activations']}, ‖w‖={p['norm']:.3f}")
    
    print(f"\nCommutator structure:")
    for cs in intro['commutator_samples']:
        mag = cs['magnitude']
        bar = '▓' * int(mag * 20) + '░' * (20 - int(mag * 20))
        print(f"  [{cs['pair'][0]:2d},{cs['pair'][1]:2d}] |{bar}| {mag:.4f}")
    
    # Population wave visualization
    print(f"\nPopulation timeline:")
    for pt in population_timeline[::5]:
        alive = pt['alive']
        bar = '●' * alive + '○' * (56 - alive)
        deaths = pt.get('deaths_this_epoch', 0)
        ops = pt.get('ops_this_epoch', 0)
        extra = f" †{deaths}" if deaths else ""
        extra += f" +{ops}" if ops else ""
        print(f"  E{pt['epoch']:3d} [{bar}]{extra}")
    
    # Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': '015b_ecology',
        'config': {
            'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
            'geo_weight': geo_weight, 'initial_alive_fraction': 0.7,
        },
        'history': history,
        'population_timeline': population_timeline,
        'ecology_log': ecology_log,
        'final_introspection': intro,
        'lineage': {str(k): v for k, v in organism.codebook.lineage.items()},
    }
    
    out_path = Path(__file__).parent / 'ecology_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    model_path = ROOT / 'lingua' / 'living_lingua_v2.pt'
    organism.save_state(str(model_path))
    print(f"Organism saved to {model_path}")
    
    return organism, results


if __name__ == '__main__':
    organism, results = run(epochs=120, batch_size=16, lr=1e-3, geo_weight=0.1, log_every=10)
