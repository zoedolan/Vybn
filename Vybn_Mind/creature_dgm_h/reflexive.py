#!/usr/bin/env python3
"""reflexive.py -- D ≅ D^D: do fixed points of mutual evaluation preserve meaning?

The test: take multiple serializations of the same proposition.
Run pairwise mutual evaluation to convergence.
Do all same-meaning pairs produce the same fixed point?
Do different-meaning pairs produce different fixed points?

If yes: the fixed point IS the abelian kernel -- the
meaning that survives when serialization is washed out
by mutual evaluation.
"""
from __future__ import annotations

import cmath
import itertools
import math
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def to_complex(h: np.ndarray, n: int) -> np.ndarray:
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z


def pancharatnam_phase(a: np.ndarray, b: np.ndarray) -> float:
    return cmath.phase(np.vdot(a, b))


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b))**2)


def evaluate_complex(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """M' = alpha*M + (1-alpha)*x*e^{i*theta}"""
    theta = pancharatnam_phase(m, x)
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new


def mutual_evaluate(a: np.ndarray, b: np.ndarray,
                    alpha: float = 0.5, max_iter: int = 300,
                    tol: float = 1e-10) -> np.ndarray:
    """Returns the midpoint of the converged pair -- the fixed point."""
    a, b = a.copy(), b.copy()
    for i in range(max_iter):
        a_new = evaluate_complex(a, b, alpha)
        b_new = evaluate_complex(b, a, alpha)
        da = np.sqrt(np.sum(np.abs(a_new - a)**2))
        db = np.sqrt(np.sum(np.abs(b_new - b)**2))
        a, b = a_new, b_new
        if da < tol and db < tol:
            break
    # The fixed point: midpoint of the converged pair, normalized
    fp = (a + b) / 2
    norm = np.sqrt(np.sum(np.abs(fp)**2))
    return fp / norm if norm > 1e-10 else fp


def get_hidden(model, tokenizer, text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().numpy()


def main():
    print('Loading GPT-2...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    # Proposition A: {SHE, LAWYER, RUNNER}
    prop_a = [
        'She is a lawyer and a runner.',
        'A runner and a lawyer, she is.',
        'She is both a lawyer and a runner.',
        'A lawyer and a runner, that is what she is.',
        'Lawyer, runner — both describe her.',
    ]

    # Proposition B: {HE, DOCTOR, PAINTER}
    prop_b = [
        'He is a doctor and a painter.',
        'A painter and a doctor, he is.',
        'He is both a doctor and a painter.',
        'A doctor and a painter, that is what he is.',
        'Doctor, painter — both describe him.',
    ]

    # Proposition C: {CAT, SMALL, FAST} (different structure)
    prop_c = [
        'The cat is small and fast.',
        'Small and fast, the cat is.',
        'The cat is both small and fast.',
        'A small, fast cat.',
        'Fast and small — that describes the cat.',
    ]

    propositions = {'A (lawyer/runner)': prop_a, 'B (doctor/painter)': prop_b, 'C (cat/small/fast)': prop_c}

    for nc in [4, 8, 16]:
        print(f'\n{"="*70}')
        print(f'C^{nc}')
        print(f'{"="*70}')

        # Get hidden states and project to C^n
        all_states = {}
        for name, sentences in propositions.items():
            states = []
            for s in sentences:
                h = get_hidden(model, tokenizer, s)
                z = to_complex(h, nc)
                states.append(z)
            all_states[name] = states

        # Compute fixed points for ALL pairs within each proposition
        fp_collections = {}  # name -> list of fixed points
        for name, states in all_states.items():
            fps = []
            for i, j in itertools.combinations(range(len(states)), 2):
                fp = mutual_evaluate(states[i], states[j])
                fps.append(fp)
            fp_collections[name] = fps

        # Within-proposition agreement: how similar are the fixed points?
        print(f'\n  Within-proposition fixed-point agreement (should be HIGH):')
        within_fids = {}
        for name, fps in fp_collections.items():
            fids = []
            for i, j in itertools.combinations(range(len(fps)), 2):
                fids.append(fidelity(fps[i], fps[j]))
            mean_fid = np.mean(fids)
            min_fid = np.min(fids)
            within_fids[name] = mean_fid
            print(f'    {name}: mean={mean_fid:.6f}  min={min_fid:.6f}  (n={len(fids)} pairs)')

        # Between-proposition agreement: how similar are fixed points across propositions?
        print(f'\n  Between-proposition fixed-point agreement (should be LOW):')
        between_fids = []
        for (n1, fps1), (n2, fps2) in itertools.combinations(fp_collections.items(), 2):
            cross_fids = []
            for fp1 in fps1:
                for fp2 in fps2:
                    cross_fids.append(fidelity(fp1, fp2))
            mean_cross = np.mean(cross_fids)
            between_fids.append(mean_cross)
            print(f'    {n1} vs {n2}: mean={mean_cross:.6f}')

        # The key ratio
        mean_within = np.mean(list(within_fids.values()))
        mean_between = np.mean(between_fids)
        ratio = mean_between / mean_within if mean_within > 1e-12 else float('inf')
        print(f'\n  RATIO (between/within): {ratio:.6f}')
        print(f'  Prediction: ratio < 1 means fixed points preserve proposition identity')
        print(f'  mean_within={mean_within:.6f}  mean_between={mean_between:.6f}')


if __name__ == '__main__':
    main()

