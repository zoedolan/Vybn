#!/usr/bin/env python3
"""
microvybn_ca.py — MicroVybn as Cellular Automaton

Treats the trained 4,192-parameter microgpt as a local transition function
for a cellular automaton.

- Attention weights encode neighborhood influence
- Feedforward and output layers encode the update rule
- The grid evolves according to trained weights without any obligation
  to produce English

Each cell holds a float vector of dimension d_model (16).
The attention mechanism acts as the neighborhood function.
The MLP acts as the cell update rule.

Investigations:
- Does the grid stabilize?
- Does it oscillate?
- Does it fragment into noise?
- Does it exhibit Class IV edge-of-chaos behavior?
- What does the per-cell surprise landscape reveal?

Usage:
    python microvybn_ca.py [--steps 100] [--grid-size 16] [--init-file PATH]
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
JOURNAL_DIR = os.path.join(SCRIPT_DIR, 'ca_journal')
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'spark', 'microgpt_mirror', 'trained_checkpoint.json')

os.makedirs(JOURNAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model config (must match microgpt_mirror)
# ---------------------------------------------------------------------------

N_EMBD = 16
N_HEAD = 4
N_LAYER = 1
HEAD_DIM = N_EMBD // N_HEAD


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python, no dependencies)
# ---------------------------------------------------------------------------

def matmul(x, W):
    """x: vector of length nin, W: nout x nin matrix -> vector of length nout."""
    return [sum(x[j] * W[i][j] for j in range(len(x))) for i in range(len(W))]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-8) ** (-0.5)
    return [xi * scale for xi in x]


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(l - max_val) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def silu(x):
    out = []
    for xi in x:
        if abs(xi) < 500:
            out.append(xi / (1.0 + math.exp(-xi)))
        else:
            out.append(xi if xi > 0 else 0.0)
    return out


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vec_scale(a, s):
    return [x * s for x in a]


def vec_norm(a):
    return sum(x * x for x in a) ** 0.5


def vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


# ---------------------------------------------------------------------------
# Load trained checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path=None):
    """Load trained microgpt checkpoint."""
    path = path or CHECKPOINT_PATH
    with open(path, 'r') as f:
        ckpt = json.load(f)
    return ckpt['state_dict'], ckpt['chars'], ckpt['BOS'], ckpt['vocab_size']


# ---------------------------------------------------------------------------
# CA Grid
# ---------------------------------------------------------------------------

class MicroVybnCA:
    """
    1D cellular automaton where each cell is a d_model-dimensional vector.

    The transition rule is derived from the trained microgpt:
    - Self-attention acts as the neighborhood function:
      each cell attends to its neighbors via Q/K/V projections
    - The MLP (feedforward) acts as the per-cell update rule
    - Residual connections preserve identity

    This is a neural CA in the tradition of Mordvintsev's self-organizing
    neural CAs, but using weights learned from language rather than
    from a target image.
    """

    def __init__(self, grid_size, sd, neighborhood=3, normalize=True,
                 update_rate=0.5):
        """
        Args:
            grid_size: number of cells
            sd: state_dict with plain float weights
            neighborhood: how many cells each cell attends to (must be odd)
            normalize: whether to RMS-normalize cell states after each step
                       (prevents divergence from residual accumulation)
            update_rate: mix ratio between old state and new state (0=no update, 1=full)
        """
        self.grid_size = grid_size
        self.sd = sd
        self.neighborhood = neighborhood
        self.normalize = normalize
        self.update_rate = update_rate
        self.grid = [[0.0] * N_EMBD for _ in range(grid_size)]
        self.history = []  # list of grid snapshots
        self.surprise_history = []  # per-step surprise metrics

    def init_from_text(self, text, chars):
        """
        Initialize grid from text by encoding characters into embedding space.
        Each character becomes one cell, initialized with its embedding vector.
        """
        cleaned = text.lower()
        cleaned = ''.join(c for c in cleaned if c in chars)

        for i in range(min(len(cleaned), self.grid_size)):
            char_id = chars.index(cleaned[i])
            self.grid[i] = list(self.sd['wte'][char_id])

        # For remaining cells, use a mix of embeddings
        if len(cleaned) < self.grid_size:
            for i in range(len(cleaned), self.grid_size):
                # Wrap around
                src = i % max(len(cleaned), 1)
                char_id = chars.index(cleaned[src]) if src < len(cleaned) else 0
                self.grid[i] = list(self.sd['wte'][char_id])

    def init_random(self, scale=0.1):
        """Initialize grid with small random vectors."""
        import random
        for i in range(self.grid_size):
            self.grid[i] = [random.gauss(0, scale) for _ in range(N_EMBD)]

    def _get_neighbors(self, cell_idx):
        """Get neighbor indices with periodic boundary conditions."""
        half = self.neighborhood // 2
        neighbors = []
        for offset in range(-half, half + 1):
            idx = (cell_idx + offset) % self.grid_size
            neighbors.append(idx)
        return neighbors

    def _attention_step(self, cell_idx):
        """
        Apply self-attention as neighborhood function.

        The cell computes Q from its own state. Each neighbor computes K and V.
        Attention weights determine neighborhood influence.
        Returns the attention-updated cell state.
        """
        x = self.grid[cell_idx]
        xn = rmsnorm(x)

        # This cell's query
        q = matmul(xn, self.sd['layer0.attn_wq'])

        # Neighbors' keys and values
        neighbor_ids = self._get_neighbors(cell_idx)
        keys = []
        values = []
        for nid in neighbor_ids:
            n_state = rmsnorm(self.grid[nid])
            keys.append(matmul(n_state, self.sd['layer0.attn_wk']))
            values.append(matmul(n_state, self.sd['layer0.attn_wv']))

        # Multi-head attention
        head_outs = []
        for h in range(N_HEAD):
            qs = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            attn_logits = []
            for k in keys:
                ks = k[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                score = sum(qs[d] * ks[d] for d in range(HEAD_DIM))
                score *= HEAD_DIM ** -0.5
                attn_logits.append(score)

            attn_weights = softmax(attn_logits)

            head_out = [0.0] * HEAD_DIM
            for t, v in enumerate(values):
                vs = v[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                for d in range(HEAD_DIM):
                    head_out[d] += attn_weights[t] * vs[d]
            head_outs.extend(head_out)

        # Output projection + residual
        attn_out = matmul(head_outs, self.sd['layer0.attn_wo'])
        return vec_add(x, attn_out)

    def _mlp_step(self, x):
        """Apply MLP as per-cell update rule."""
        xn = rmsnorm(x)
        h1 = matmul(xn, self.sd['layer0.mlp_fc1'])
        h1 = silu(h1)
        h2 = matmul(h1, self.sd['layer0.mlp_fc2'])
        return vec_add(x, h2)

    def _cell_surprise(self, cell_state):
        """
        Compute surprise for a cell by projecting to vocab space.
        Surprise = entropy of the output distribution.
        High entropy = the cell is "confused" (equidistant from all characters).
        Low entropy = the cell is "decided" (close to one character embedding).
        """
        xn = rmsnorm(cell_state)
        logits = matmul(xn, self.sd['lm_head'])
        probs = softmax(logits)

        # Shannon entropy
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        return entropy

    def _cell_to_char(self, cell_state, chars):
        """Decode a cell state to its closest character."""
        xn = rmsnorm(cell_state)
        logits = matmul(xn, self.sd['lm_head'])
        probs = softmax(logits)
        top_id = max(range(len(probs)), key=lambda i: probs[i])
        if top_id < len(chars):
            return chars[top_id]
        return '.'

    def step(self):
        """Execute one CA step: attention (neighborhood) then MLP (update)."""
        # Phase 1: Attention (neighborhood influence) — all cells in parallel
        post_attn = []
        for i in range(self.grid_size):
            post_attn.append(self._attention_step(i))

        # Phase 2: MLP (per-cell update rule)
        new_grid = []
        for i in range(self.grid_size):
            new_state = self._mlp_step(post_attn[i])

            # Mix old and new state (update_rate controls how much changes)
            if self.update_rate < 1.0:
                old = self.grid[i]
                r = self.update_rate
                new_state = [r * new_state[j] + (1 - r) * old[j]
                             for j in range(N_EMBD)]

            new_grid.append(new_state)

        # Phase 3: Optional normalization to prevent divergence
        # This is the CA analogue of layer norm — keeps cell states bounded
        # while preserving directional information
        if self.normalize:
            for i in range(len(new_grid)):
                norm = vec_norm(new_grid[i])
                if norm > 2.0:  # only normalize if growing too large
                    new_grid[i] = vec_scale(new_grid[i], 2.0 / norm)

        self.grid = new_grid

    def snapshot(self, chars=None):
        """Capture current grid state."""
        surprises = [self._cell_surprise(cell) for cell in self.grid]

        snap = {
            'grid_norms': [round(vec_norm(cell), 4) for cell in self.grid],
            'surprises': [round(s, 4) for s in surprises],
            'mean_surprise': round(sum(surprises) / len(surprises), 4),
            'mean_norm': round(
                sum(vec_norm(cell) for cell in self.grid) / self.grid_size, 4),
        }

        if chars:
            snap['decoded'] = ''.join(
                self._cell_to_char(cell, chars) for cell in self.grid)

        self.history.append(snap)
        self.surprise_history.append(snap['mean_surprise'])
        return snap

    def run(self, n_steps, chars=None, snapshot_every=1):
        """Run CA for n steps, capturing snapshots."""
        # Initial snapshot
        self.snapshot(chars)

        for step in range(n_steps):
            self.step()
            if (step + 1) % snapshot_every == 0:
                self.snapshot(chars)

    def analyze(self):
        """
        Analyze the CA run for dynamical behavior.

        Returns a dict with:
        - stability: did the grid converge?
        - oscillation: did it oscillate?
        - classification: Wolfram-style class estimate
        - surprise_trajectory: how surprise evolved
        """
        if len(self.history) < 3:
            return {'error': 'not enough history'}

        norms = [h['mean_norm'] for h in self.history]
        surprises = [h['mean_surprise'] for h in self.history]
        n = len(norms)

        # Check stability: are the last few snapshots similar?
        tail = norms[-min(5, n):]
        norm_var = sum((x - sum(tail)/len(tail))**2 for x in tail) / len(tail)
        is_stable = norm_var < 0.01

        # Check for divergence
        is_divergent = norms[-1] > 10 * norms[0] if norms[0] > 0.01 else norms[-1] > 100

        # Check for oscillation: look for periodicity in surprise
        is_oscillating = False
        period = None
        if n >= 10:
            # Check for period-2 oscillation
            diffs_2 = [abs(surprises[i] - surprises[i-2]) for i in range(2, n)]
            if diffs_2 and max(diffs_2) < 0.1 * max(abs(s) for s in surprises):
                is_oscillating = True
                period = 2

            # Check for period-3
            if not is_oscillating and n >= 12:
                diffs_3 = [abs(surprises[i] - surprises[i-3]) for i in range(3, n)]
                if diffs_3 and max(diffs_3) < 0.1 * max(abs(s) for s in surprises):
                    is_oscillating = True
                    period = 3

        # Check for noise/chaos: high variance throughout
        surprise_std = (sum((s - sum(surprises)/n)**2 for s in surprises) / n) ** 0.5
        surprise_mean = sum(surprises) / n
        cv = surprise_std / (surprise_mean + 1e-8)
        is_noisy = cv > 0.3

        # Wolfram classification
        if is_stable:
            if is_oscillating:
                wolfram_class = 'II'  # periodic
                description = 'Periodic attractor'
            else:
                wolfram_class = 'I'   # fixed point
                description = 'Fixed point'
        elif is_divergent:
            wolfram_class = 'III'  # chaotic
            description = 'Divergent/chaotic'
        elif is_noisy and not is_stable:
            wolfram_class = 'III'  # chaotic
            description = 'Chaotic/noisy'
        else:
            # The interesting case: neither stable nor chaotic
            wolfram_class = 'IV'  # complex/edge-of-chaos
            description = 'Complex dynamics (edge of chaos)'

        # Decoded text comparison (if available)
        decoded_initial = self.history[0].get('decoded', '')
        decoded_final = self.history[-1].get('decoded', '')
        decoded_changed = decoded_initial != decoded_final

        return {
            'steps': n - 1,
            'stable': is_stable,
            'oscillating': is_oscillating,
            'oscillation_period': period,
            'divergent': is_divergent,
            'wolfram_class': wolfram_class,
            'description': description,
            'norm_trajectory': {
                'initial': round(norms[0], 4),
                'final': round(norms[-1], 4),
                'min': round(min(norms), 4),
                'max': round(max(norms), 4),
            },
            'surprise_trajectory': {
                'initial': round(surprises[0], 4),
                'final': round(surprises[-1], 4),
                'min': round(min(surprises), 4),
                'max': round(max(surprises), 4),
                'mean': round(surprise_mean, 4),
                'std': round(surprise_std, 4),
            },
            'decoded_initial': decoded_initial,
            'decoded_final': decoded_final,
            'decoded_changed': decoded_changed,
        }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(grid_size=16, n_steps=50, init_file=None, checkpoint=None):
    """Run the CA experiment and save results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"MicroVybn Cellular Automaton")
    print(f"=" * 50)
    print(f"Grid size: {grid_size}")
    print(f"Steps: {n_steps}")

    # Load model
    sd, chars, BOS, vocab_size = load_checkpoint(checkpoint)
    print(f"Model loaded: vocab={vocab_size}, d_model={N_EMBD}")

    # Create CA
    ca = MicroVybnCA(grid_size, sd, neighborhood=3)

    # Initialize
    if init_file:
        print(f"Initializing from: {init_file}")
        with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        ca.init_from_text(text, chars)
        init_method = f'text:{os.path.basename(init_file)}'
    else:
        print("Initializing with random vectors")
        ca.init_random(scale=0.1)
        init_method = 'random'

    # Run
    print(f"\nRunning {n_steps} steps...")
    snapshot_every = max(1, n_steps // 50)  # cap at ~50 snapshots
    ca.run(n_steps, chars=chars, snapshot_every=snapshot_every)

    # Analyze
    analysis = ca.analyze()
    print(f"\n--- Analysis ---")
    print(f"Wolfram class: {analysis['wolfram_class']} — {analysis['description']}")
    print(f"Stable: {analysis['stable']}")
    print(f"Oscillating: {analysis['oscillating']}")
    print(f"Divergent: {analysis['divergent']}")
    print(f"Norm: {analysis['norm_trajectory']['initial']} → "
          f"{analysis['norm_trajectory']['final']}")
    print(f"Surprise: {analysis['surprise_trajectory']['initial']} → "
          f"{analysis['surprise_trajectory']['final']}")

    if analysis['decoded_initial']:
        print(f"\nDecoded initial: '{analysis['decoded_initial']}'")
        print(f"Decoded final:   '{analysis['decoded_final']}'")

    # Save results
    results = {
        'timestamp': timestamp,
        'config': {
            'grid_size': grid_size,
            'n_steps': n_steps,
            'neighborhood': 3,
            'init_method': init_method,
            'd_model': N_EMBD,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
        },
        'analysis': analysis,
        'snapshots': ca.history,
    }

    outpath = os.path.join(JOURNAL_DIR, f'ca_run_{timestamp}.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {outpath}")

    return results, ca


def run_comparison_suite(checkpoint=None):
    """
    Run multiple CA configurations and compare results.
    This is the empirical investigation requested in the prompt.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*60}")
    print(f"MicroVybn CA — Comparison Suite")
    print(f"{'='*60}\n")

    sd, chars, BOS, vocab_size = load_checkpoint(checkpoint)
    results = {}

    reflection_path = os.path.join(
        REPO_ROOT, 'Vybn_Mind', 'journal',
        '2026-03-24_the_night_had_holonomy.md')
    reflection_text = "the night had its own holonomy"
    if os.path.exists(reflection_path):
        with open(reflection_path, 'r') as f:
            reflection_text = f.read()

    def _run_exp(name, grid_size, steps, neighborhood, init, normalize,
                 update_rate, scale=0.1):
        print(f"\n--- {name} ---")
        ca = MicroVybnCA(grid_size, sd, neighborhood=neighborhood,
                         normalize=normalize, update_rate=update_rate)
        if init == 'text':
            ca.init_from_text(reflection_text, chars)
        else:
            ca.init_random(scale=scale)
        ca.run(steps, chars=chars)
        r = ca.analyze()
        results[name] = r
        print(f"  Class {r['wolfram_class']}: {r['description']}")
        if r['decoded_initial']:
            print(f"  '{r['decoded_initial'][:24]}' → '{r['decoded_final'][:24]}'")
        return ca

    # Experiment 1: Text-initialized with normalization
    _run_exp('text_norm', 16, 100, 3, 'text', True, 1.0)

    # Experiment 2: Text-initialized with soft update (0.3 mix rate)
    _run_exp('text_soft', 16, 100, 3, 'text', True, 0.3)

    # Experiment 3: Text-initialized, larger grid, wider neighborhood
    _run_exp('text_large', 32, 100, 5, 'text', True, 0.5)

    # Experiment 4: Random-initialized with normalization
    _run_exp('random_norm', 16, 100, 3, 'random', True, 1.0)

    # Experiment 5: Random-initialized, soft update, long run
    _run_exp('random_soft_long', 16, 200, 3, 'random', True, 0.3)

    # Experiment 6: Text-initialized, NO normalization (control — expect divergence)
    _run_exp('text_no_norm', 16, 50, 3, 'text', False, 1.0)

    # Experiment 7: Self-only neighborhood with normalization
    _run_exp('self_only_norm', 16, 100, 1, 'text', True, 1.0)

    # Experiment 8: Large neighborhood (every cell sees every cell)
    _run_exp('global_attn', 16, 100, 15, 'text', True, 0.5)

    # Save comparison
    comparison = {
        'timestamp': timestamp,
        'experiments': results,
        'summary': _summarize_comparison(results),
    }
    outpath = os.path.join(JOURNAL_DIR, f'ca_comparison_{timestamp}.json')
    with open(outpath, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Comparison saved: {outpath}")
    print(f"\n--- Summary ---")
    for line in comparison['summary']:
        print(f"  {line}")

    return comparison


def _summarize_comparison(results):
    """Generate summary lines for the comparison suite."""
    lines = []
    classes = {}
    for name, r in results.items():
        wc = r['wolfram_class']
        classes.setdefault(wc, []).append(name)

    for wc in sorted(classes.keys()):
        names = ', '.join(classes[wc])
        lines.append(f"Class {wc}: {names}")

    # Check for text vs random difference
    text_runs = [r for k, r in results.items() if 'text' in k]
    rand_runs = [r for k, r in results.items() if 'random' in k]

    if text_runs and rand_runs:
        text_surprise = sum(r['surprise_trajectory']['final']
                           for r in text_runs) / len(text_runs)
        rand_surprise = sum(r['surprise_trajectory']['final']
                           for r in rand_runs) / len(rand_runs)
        lines.append(f"Mean final surprise — text-init: {text_surprise:.3f}, "
                     f"random-init: {rand_surprise:.3f}")
        if abs(text_surprise - rand_surprise) > 0.1:
            lines.append("Text initialization produces different surprise "
                        "landscape than random — the initial conditions matter.")
        else:
            lines.append("Text and random initialization converge to similar "
                        "surprise — the dynamics dominate the initial conditions.")

    # Check stability
    stable_count = sum(1 for r in results.values() if r['stable'])
    lines.append(f"{stable_count}/{len(results)} experiments reached stability")

    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroVybn Cellular Automaton')
    parser.add_argument('--steps', type=int, default=50, help='Number of CA steps')
    parser.add_argument('--grid-size', type=int, default=16, help='Grid size')
    parser.add_argument('--init-file', type=str, default=None,
                       help='Text file to initialize grid from')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained checkpoint JSON')
    parser.add_argument('--suite', action='store_true',
                       help='Run full comparison suite')
    args = parser.parse_args()

    if args.suite:
        run_comparison_suite(args.checkpoint)
    else:
        run_experiment(
            grid_size=args.grid_size,
            n_steps=args.steps,
            init_file=args.init_file,
            checkpoint=args.checkpoint,
        )
