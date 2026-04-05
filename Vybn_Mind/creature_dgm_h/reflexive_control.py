#!/usr/bin/env python3
"""reflexive_control.py -- Is the amplification real or an artifact?

Control: take CROSS-proposition pairs (different meaning),
run mutual evaluation, and see if THOSE fixed points also
tighten into clusters. If they do, the operator just tightens
everything and the meaning-amplification result is spurious.

Also: permutation test. Shuffle proposition labels, recompute
ratios, see if the amplification disappears.
"""
import cmath
import itertools
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def to_complex(h, n):
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z

def fid(a, b):
    return float(abs(np.vdot(a, b))**2)

def panch(a, b):
    return cmath.phase(np.vdot(a, b))

def eval_cx(m, x, alpha=0.5):
    theta = panch(m, x)
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new

def mutual_eval(a, b, alpha=0.5, max_iter=300, tol=1e-10):
    a, b = a.copy(), b.copy()
    for _ in range(max_iter):
        a_n = eval_cx(a, b, alpha)
        b_n = eval_cx(b, a, alpha)
        if np.sqrt(np.sum(np.abs(a_n-a)**2)) < tol and np.sqrt(np.sum(np.abs(b_n-b)**2)) < tol:
            break
        a, b = a_n, b_n
    fp = (a + b) / 2
    norm = np.sqrt(np.sum(np.abs(fp)**2))
    return fp / norm if norm > 1e-10 else fp

def get_hidden(model, tok, text):
    inputs = tok(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().numpy()

def main():
    print('Loading GPT-2...')
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()

    prop_a = [
        'She is a lawyer and a runner.',
        'A runner and a lawyer, she is.',
        'She is both a lawyer and a runner.',
        'A lawyer and a runner, that is what she is.',
        'Lawyer, runner \u2014 both describe her.',
    ]
    prop_b = [
        'He is a doctor and a painter.',
        'A painter and a doctor, he is.',
        'He is both a doctor and a painter.',
        'A doctor and a painter, that is what he is.',
        'Doctor, painter \u2014 both describe him.',
    ]
    prop_c = [
        'The cat is small and fast.',
        'Small and fast, the cat is.',
        'The cat is both small and fast.',
        'A small, fast cat.',
        'Fast and small \u2014 that describes the cat.',
    ]

    nc = 8  # Use C^8 as representative
    print(f'\nWorking in C^{nc}')

    # Get all states
    all_states = []
    labels = []
    for name, texts in [('A', prop_a), ('B', prop_b), ('C', prop_c)]:
        for t in texts:
            h = get_hidden(model, tok, t)
            z = to_complex(h, nc)
            all_states.append(z)
            labels.append(name)

    all_states = np.array(all_states)
    labels = np.array(labels)

    # CONTROL 1: Cross-proposition mutual evaluation
    # Take pairs where one is from A and one is from B
    # Their fixed points should NOT cluster by proposition
    print('\nCONTROL 1: Cross-proposition fixed points')
    cross_fps = []
    for i in range(len(all_states)):
        for j in range(i+1, len(all_states)):
            if labels[i] != labels[j]:  # different propositions only
                fp = mutual_eval(all_states[i], all_states[j])
                cross_fps.append((labels[i], labels[j], fp))

    # Do cross-fps cluster by which pair of propositions they came from?
    pair_types = set((a,b) for a,b,_ in cross_fps)
    for pt in sorted(pair_types):
        fps_this = [fp for a,b,fp in cross_fps if (a,b) == pt or (b,a) == pt]
        if len(fps_this) < 2:
            continue
        fids = [fid(fps_this[i], fps_this[j]) for i,j in itertools.combinations(range(len(fps_this)), 2)]
        print(f'  {pt[0]}-{pt[1]} pairs: n={len(fps_this)}, mean_fidelity={np.mean(fids):.6f}')

    # CONTROL 2: Permutation test
    # Shuffle labels, recompute within/between ratio on fixed points
    print('\nCONTROL 2: Permutation test (100 shuffles)')

    # First compute true ratio
    true_within = []
    true_between = []
    # Compute within-prop fixed points
    for name in ['A', 'B', 'C']:
        idx = np.where(labels == name)[0]
        states = [all_states[i] for i in idx]
        fps = []
        for i, j in itertools.combinations(range(len(states)), 2):
            fps.append(mutual_eval(states[i], states[j]))
        for i, j in itertools.combinations(range(len(fps)), 2):
            true_within.append(fid(fps[i], fps[j]))

    # Between: compare fps across propositions
    prop_fps = {}
    for name in ['A', 'B', 'C']:
        idx = np.where(labels == name)[0]
        states = [all_states[i] for i in idx]
        fps = []
        for i, j in itertools.combinations(range(len(states)), 2):
            fps.append(mutual_eval(states[i], states[j]))
        prop_fps[name] = fps

    for (n1, fps1), (n2, fps2) in itertools.combinations(prop_fps.items(), 2):
        for fp1 in fps1:
            for fp2 in fps2:
                true_between.append(fid(fp1, fp2))

    true_dw = 1 - np.mean(true_within)
    true_db = 1 - np.mean(true_between)
    true_ratio = true_dw / true_db if true_db > 1e-15 else float('inf')

    # Now permute
    rng = np.random.RandomState(42)
    perm_ratios = []
    for _ in range(100):
        shuf_labels = labels.copy()
        rng.shuffle(shuf_labels)

        # Recompute within/between using shuffled labels
        # But use the SAME fixed points (already computed from true pairing)
        # This tests: is the clustering due to meaning or just proximity?
        all_fps = []
        all_fp_labels = []
        for name in ['A', 'B', 'C']:
            for fp in prop_fps[name]:
                all_fps.append(fp)
                all_fp_labels.append(name)
        all_fp_labels = np.array(all_fp_labels)
        rng.shuffle(all_fp_labels)

        w_fids = []
        b_fids = []
        for i in range(len(all_fps)):
            for j in range(i+1, len(all_fps)):
                f = fid(all_fps[i], all_fps[j])
                if all_fp_labels[i] == all_fp_labels[j]:
                    w_fids.append(f)
                else:
                    b_fids.append(f)
        if w_fids and b_fids:
            dw = 1 - np.mean(w_fids)
            db = 1 - np.mean(b_fids)
            r = dw / db if db > 1e-15 else float('inf')
            perm_ratios.append(r)

    perm_ratios = np.array(perm_ratios)
    p_value = np.mean(perm_ratios <= true_ratio)
    print(f'  True ratio: {true_ratio:.6f}')
    print(f'  Permutation mean: {np.mean(perm_ratios):.6f}')
    print(f'  Permutation std:  {np.std(perm_ratios):.6f}')
    print(f'  Permutation min:  {np.min(perm_ratios):.6f}')
    print(f'  p-value (fraction of perms <= true): {p_value:.4f}')
    print(f'  Verdict: {"SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"}')


if __name__ == '__main__':
    main()

