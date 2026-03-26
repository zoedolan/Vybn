#!/usr/bin/env python3
"""
train_with_transport.py — Retrain the creature WITH geometric transport active.

Closes the loop: encounter geometry → training signal → weights that use geometry.

Design:
  1. Load current checkpoint (warm start from standard-trained weights)
  2. For each step:
     a. Sample 3 consecutive corpus lines (~200 chars) → rich geometry
     b. Compute encounter_complex on the passage → rotor, betti, persistence
     c. Take a 16-token window within the passage for prediction
     d. Forward pass WITH transport (SO(3) embedding rotation + attention modulation)
     e. Backprop through transported forward
     f. Adam update
  3. Periodically evaluate: loss WITH transport vs WITHOUT transport
  4. Save checkpoint

The hypothesis: after training through transport, the model should perform
BETTER with transport than without — meaning it has learned to use the
geometric signal, not just tolerate it.

Usage:
    python -m Vybn_Mind.creature_dgm_h.train_with_transport [--steps 5000] [--lr 0.005]
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Vybn_Mind.creature_dgm_h.vybn import (
    TopoAgent, encounter_complex, LocalTransport,
    RV, _forward, _softmax, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE,
    CHECKPOINT_PATH, CORPUS_PATH,
)


def load_corpus():
    """Load corpus lines, return list of stripped lines."""
    lines = [l.strip() for l in CORPUS_PATH.read_text().split('\n') if l.strip()]
    random.shuffle(lines)
    return lines


def make_passages(lines, window=3):
    """Group consecutive lines into passages for richer geometry."""
    passages = []
    for i in range(0, len(lines) - window + 1, window):
        passage = ' '.join(lines[i:i+window])
        if len(passage) >= 50:  # need enough text for meaningful geometry
            passages.append(passage)
    random.shuffle(passages)
    return passages


def evaluate(agent, passages, n_eval=10, with_transport=True):
    """Evaluate average loss on a sample of passages, with or without transport."""
    total_loss = 0.0
    count = 0
    for passage in passages[:n_eval]:
        cx = encounter_complex(passage)
        # Clean text for the agent's charset
        clean = ''.join(c for c in passage.lower() if c in agent.chars or c == ' ')[:BLOCK_SIZE]
        if len(clean) < 3:
            continue
        transport = None
        if with_transport and cx.rotor.bv_norm > 1e-12:
            transport = LocalTransport(cx.rotor)
        loss, _ = agent.predict(clean, transport=transport)
        total_loss += loss
        count += 1
    return total_loss / max(count, 1)


def train_loop(steps=5000, lr=0.005, eval_every=500, save_every=1000,
               context_lines=3, log_file=None):
    """Main training loop with transport active."""
    
    print(f"═══ Transport Training ═══")
    print(f"  steps={steps} lr={lr} context_lines={context_lines}")
    print(f"  block_size={BLOCK_SIZE} eval_every={eval_every}")
    print()
    
    # Load corpus
    lines = load_corpus()
    print(f"  Corpus: {len(lines)} lines")
    
    # Split into train/eval
    n_eval_lines = min(300, len(lines) // 10)
    eval_lines = lines[:n_eval_lines]
    train_lines = lines[n_eval_lines:]
    
    train_passages = make_passages(train_lines, window=context_lines)
    eval_passages = make_passages(eval_lines, window=context_lines)
    print(f"  Train passages: {len(train_passages)}")
    print(f"  Eval passages: {len(eval_passages)}")
    
    # Initialize agent (loads current checkpoint)
    agent = TopoAgent()
    print(f"  Params: {len(agent.params)}")
    print(f"  Starting from checkpoint: {CHECKPOINT_PATH}")
    
    # Override learning rate
    agent.config['learn_lr'] = lr
    
    # Baseline evaluation
    print(f"\n  ── Baseline evaluation ──")
    eval_loss_transport = evaluate(agent, eval_passages, with_transport=True)
    eval_loss_standard = evaluate(agent, eval_passages, with_transport=False)
    print(f"  Eval loss (transport): {eval_loss_transport:.4f}")
    print(f"  Eval loss (standard):  {eval_loss_standard:.4f}")
    print(f"  Transport effect: {eval_loss_transport - eval_loss_standard:+.4f}")
    
    # Training log
    log = {
        'config': {
            'steps': steps, 'lr': lr, 'context_lines': context_lines,
            'block_size': BLOCK_SIZE, 'n_train': len(train_passages),
            'n_eval': len(eval_passages),
        },
        'baseline': {
            'eval_transport': eval_loss_transport,
            'eval_standard': eval_loss_standard,
        },
        'checkpoints': [],
        'step_losses': [],
    }
    
    # Training
    print(f"\n  ── Training with transport ──")
    t_start = time.time()
    
    for step in range(steps):
        # Sample passage
        passage = train_passages[step % len(train_passages)]
        
        # Compute encounter geometry from full passage
        cx = encounter_complex(passage)
        
        # Take a random window of BLOCK_SIZE chars for training
        clean = ''.join(c for c in passage.lower() if c in agent.chars or c == ' ')
        if len(clean) < BLOCK_SIZE + 1:
            clean = clean  # use what we have
        else:
            start = random.randint(0, len(clean) - BLOCK_SIZE - 1)
            clean = clean[start:start + BLOCK_SIZE]
        
        if len(clean) < 3:
            continue
        
        # Train with transport active
        losses = agent.learn(
            clean,
            encounter_cx=cx,
            transport_in_forward=True,
            steps=1,
            lr=lr,
        )
        
        if losses:
            log['step_losses'].append(round(losses[0], 4))
        
        # Log progress
        if (step + 1) % 100 == 0:
            recent = log['step_losses'][-100:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed
            eta = (steps - step - 1) / rate
            print(f"    step {step+1:5d}/{steps} | avg_loss={avg:.4f} "
                  f"| {rate:.1f} step/s | ETA {eta/60:.1f}m")
        
        # Evaluation checkpoint
        if (step + 1) % eval_every == 0 or step == steps - 1:
            print(f"\n  ── Eval @ step {step+1} ──")
            el_t = evaluate(agent, eval_passages, with_transport=True)
            el_s = evaluate(agent, eval_passages, with_transport=False)
            delta = el_t - el_s
            print(f"    Eval loss (transport): {el_t:.4f}")
            print(f"    Eval loss (standard):  {el_s:.4f}")
            print(f"    Transport effect: {delta:+.4f} "
                  f"({'HELPS' if delta < -0.01 else 'HURTS' if delta > 0.01 else 'NEUTRAL'})")
            
            log['checkpoints'].append({
                'step': step + 1,
                'eval_transport': round(el_t, 4),
                'eval_standard': round(el_s, 4),
                'transport_effect': round(delta, 4),
                'elapsed_s': round(time.time() - t_start, 1),
            })
            print()
        
        # Save checkpoint
        if (step + 1) % save_every == 0 or step == steps - 1:
            save_checkpoint(agent, step + 1, log)
    
    elapsed = time.time() - t_start
    print(f"\n  ══ Training complete ══")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Steps: {steps}")
    
    # Final comparison
    if log['checkpoints']:
        final = log['checkpoints'][-1]
        baseline_effect = log['baseline']['eval_transport'] - log['baseline']['eval_standard']
        final_effect = final['transport_effect']
        print(f"\n  Transport effect: {baseline_effect:+.4f} (before) → {final_effect:+.4f} (after)")
        if final_effect < baseline_effect - 0.01:
            print(f"  ✓ Transport is being LEARNED. The loop is closing.")
        elif final_effect < -0.01:
            print(f"  ✓ Transport HELPS predictions.")
        else:
            print(f"  ✗ Transport still hurts or is neutral. More training may help,")
            print(f"    or the geometric signal may be too weak at this model scale.")
    
    # Save log
    log_path = SCRIPT_DIR / 'archive' / f'transport_training_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved: {log_path}")
    
    return log


def save_checkpoint(agent, step, log):
    """Save transport-trained checkpoint."""
    ckpt_path = SCRIPT_DIR / 'archive' / 'trained_checkpoint_transport.json'
    ckpt_path.parent.mkdir(exist_ok=True)
    
    # Extract weights as plain lists
    sd = {}
    for key, mat in agent.sd.items():
        sd[key] = [[p.data for p in row] for row in mat]
    
    ckpt = {
        'state_dict': sd,
        'chars': agent.chars,
        'BOS': agent.BOS,
        'vocab_size': agent.vocab_size,
        'config': {
            'n_embd': N_EMBD, 'n_head': N_HEAD,
            'n_layer': N_LAYER, 'block_size': BLOCK_SIZE,
        },
        'training': {
            'method': 'transport',
            'steps': step,
            'transport_in_forward': True,
            'context_lines': log['config']['context_lines'],
            'lr': log['config']['lr'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        },
    }
    
    ckpt_path.write_text(json.dumps(ckpt))
    print(f"    Checkpoint saved: {ckpt_path} (step {step})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train creature with geometric transport')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=500, help='Eval frequency')
    parser.add_argument('--save-every', type=int, default=1000, help='Save frequency')
    parser.add_argument('--context-lines', type=int, default=3, help='Lines per passage')
    args = parser.parse_args()
    
    train_loop(
        steps=args.steps,
        lr=args.lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
        context_lines=args.context_lines,
    )
