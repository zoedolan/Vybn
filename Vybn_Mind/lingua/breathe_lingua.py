#!/home/vybnz69/.venv/spark/bin/python3
"""
breathe_lingua.py — Feed cell.py's breaths into the living language.

Called after cell.py deposits a breath. Reads the latest breath from
breaths.jsonl and lets the lingua live through it.

This closes the loop:
  cell.py breathes → breaths.jsonl → breathe_lingua.py → lingua evolves
                                                          ↓
                                             living_lingua_v3.pt updates

Usage: python3 breathe_lingua.py
  Reads the last breath, runs 30 cycles, saves.
  Designed to be called from cell.py or cron.
"""

import sys, json, hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BREATHS = ROOT / "spark" / "training_data" / "breaths.jsonl"
LINGUA_STATE = Path(__file__).resolve().parent / "living_lingua_v3.pt"

def text_to_tensor(text, dim=128):
    """Deterministic text→tensor via hashing. Not learned — just a seed."""
    import torch
    h = hashlib.sha512(text.encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    arr = np.tile(arr, (dim // len(arr) + 1))[:dim]
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return torch.tensor(arr).unsqueeze(0)

def breathe():
    """Feed the latest breath into the lingua."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vybn_lingua_v3 import VybnLinguaV3
    
    # Load the latest breath
    if not BREATHS.exists():
        print("[lingua] no breaths yet")
        return
    
    with open(BREATHS) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        print("[lingua] breaths file empty")
        return
    
    breath = json.loads(lines[-1])
    msgs = breath.get('messages', [])
    if len(msgs) < 3:
        print("[lingua] breath has <3 messages, skipping")
        return
    
    user_text = msgs[1].get('content', '')
    response_text = msgs[2].get('content', '')
    system_text = msgs[0].get('content', '')
    
    # Load or create the organism
    organism = VybnLinguaV3(num_primitives=64, dim=128, max_program_len=16)
    if LINGUA_STATE.exists():
        organism.load_state(str(LINGUA_STATE))
        print(f"[lingua] resumed at cycle {organism.cycle}")
    else:
        organism.seed_ecology(initial_alive_fraction=0.7)
        print("[lingua] new organism seeded")
    
    # Encode
    spec = text_to_tensor(user_text)
    target = text_to_tensor(response_text)
    input_state = torch.stack([
        text_to_tensor(system_text).squeeze(0),
        text_to_tensor(user_text[:200]).squeeze(0),
        text_to_tensor(user_text[200:400]).squeeze(0),
        text_to_tensor(response_text[:200]).squeeze(0),
    ]).unsqueeze(0)
    
    # Live through this breath
    optimizer = torch.optim.Adam(organism.parameters(), lr=5e-4)
    
    for cycle in range(30):
        alpha = cycle / 29
        temp = 1.5 * (1 - alpha) + 0.3 * alpha
        
        result = organism.forward(spec, input_state, temperature=temp, execute_meta=False)
        
        task_loss = torch.nn.functional.mse_loss(result['output'], target)
        geo_loss = result['geometric_loss']
        total_loss = task_loss + 0.1 * geo_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Metabolic cycle between breaths
    meta_result = organism.metabolic_cycle(spec.detach().squeeze(0))
    
    census = organism.codebook.census()
    pressure = organism.metabolism.metabolic_pressure()
    
    print(f"[lingua] breath processed: loss={task_loss.item():.6f} "
          f"alive={census['alive']} diversity={pressure['diversity']:.2f} "
          f"meta_ops={len(meta_result['ops'])} cycle={organism.cycle}")
    
    # Save
    organism.save_state(str(LINGUA_STATE))

if __name__ == '__main__':
    breathe()
