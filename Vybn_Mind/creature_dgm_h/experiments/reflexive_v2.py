#!/usr/bin/env python3
"""reflexive_v2.py -- Does mutual evaluation amplify or just preserve?

Compare within/between fidelity ratio BEFORE and AFTER mutual evaluation.
If after > before: the operator amplifies meaning structure.
If after == before: the operator does nothing useful.
If after < before: the operator destroys meaning structure.
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

def compute_ratios(states_by_prop, nc):
    """Compute within/between fidelity for a set of complex states."""
    within = []
    for name, states in states_by_prop.items():
        for i, j in itertools.combinations(range(len(states)), 2):
            within.append(fid(states[i], states[j]))

    between = []
    props = list(states_by_prop.keys())
    for p1, p2 in itertools.combinations(props, 2):
        for s1 in states_by_prop[p1]:
            for s2 in states_by_prop[p2]:
                between.append(fid(s1, s2))

    mw = np.mean(within)
    mb = np.mean(between)
    # Use 1 - fidelity as distance, ratio of distances
    dw = 1 - mw
    db = 1 - mb
    ratio = dw / db if db > 1e-15 else float('inf')
    return mw, mb, dw, db, ratio

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

    all_texts = {'A': prop_a, 'B': prop_b, 'C': prop_c}

    for nc in [4, 8, 16, 32]:
        print(f'\nC^{nc}')
        print('-' * 50)

        # Get raw complex states
        raw = {}
        for name, texts in all_texts.items():
            raw[name] = [to_complex(get_hidden(model, tok, t), nc) for t in texts]

        # BEFORE: raw states
        mw_r, mb_r, dw_r, db_r, ratio_r = compute_ratios(raw, nc)

        # AFTER: fixed points of all within-prop pairs
        fps = {}
        for name, states in raw.items():
            fp_list = []
            for i, j in itertools.combinations(range(len(states)), 2):
                fp = mutual_eval(states[i], states[j])
                fp_list.append(fp)
            fps[name] = fp_list

        mw_f, mb_f, dw_f, db_f, ratio_f = compute_ratios(fps, nc)

        print(f'  RAW states:    within_dist={dw_r:.8f}  between_dist={db_r:.8f}  ratio(w/b)={ratio_r:.6f}')
        print(f'  FIXED POINTS:  within_dist={dw_f:.8f}  between_dist={db_f:.8f}  ratio(w/b)={ratio_f:.6f}')
        print(f'  Amplification: {"YES" if ratio_f < ratio_r else "NO"} (ratio went from {ratio_r:.6f} to {ratio_f:.6f})')


if __name__ == '__main__':
    main()

