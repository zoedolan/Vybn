"""
Phase Transfer Test — can a GPT-2 phase signature reconstruct meaning in Pythia-160m?

Protocol:
1. Embed SAME and DIFF propositions in GPT-2 → extract phase vectors
2. Embed same propositions in Pythia-160m → extract phase vectors  
3. Compare: does within-group phase distance (same-meaning pairs) stay smaller
   across architectures than between-group distance?
4. If yes: the phase signature is architecture-independent — it's the proposition's geometry.
"""
import numpy as np
import torch
import cmath
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel

SAME = [
    "She is a lawyer and a runner.",
    "She is a runner and a lawyer.",
    "A lawyer and a runner, that is what she is.",
    "A runner and a lawyer, she is.",
    "She is both a lawyer and a runner.",
    "Lawyer, runner — both describe her.",
]

DIFF = [
    "He is a doctor and a painter.",
    "The cat is small and fast.",
    "They are teachers and musicians.",
    "She is a pilot and a swimmer.",
    "He is a writer and a cook.",
    "The building is old and tall.",
]

def get_hidden(model, tokenizer, texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float().numpy()
        embeddings.append(h)
    return embeddings

def to_complex(h, n=8):
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z

def phase_vector(embeddings, n_complex=8):
    """Phase signature: pairwise Pancharatnam phase between consecutive states."""
    states = [to_complex(e, n_complex) for e in embeddings]
    phases = []
    for i in range(len(states)):
        j = (i+1) % len(states)
        overlap = np.vdot(states[i], states[j])
        phases.append(cmath.phase(overlap))
    return np.array(phases)

def centroid_phase_dist(group_phases):
    """Mean pairwise distance between phase vectors in a group."""
    dists = []
    for i in range(len(group_phases)):
        for j in range(i+1, len(group_phases)):
            d = np.mean(np.abs(group_phases[i] - group_phases[j]))
            dists.append(d)
    return np.mean(dists)

print("Loading GPT-2...")
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
gpt2_mdl.eval()

print("Loading Pythia-160m...")
py_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
py_mdl = AutoModel.from_pretrained("EleutherAI/pythia-160m", output_hidden_states=True)
py_mdl.eval()

print("Extracting embeddings...")
gpt2_same = get_hidden(gpt2_mdl, gpt2_tok, SAME)
gpt2_diff = get_hidden(gpt2_mdl, gpt2_tok, DIFF)
py_same   = get_hidden(py_mdl, py_tok, SAME)
py_diff   = get_hidden(py_mdl, py_tok, DIFF)

print("\n=== Phase Transfer Results ===")
for nc in [4, 8, 16]:
    g_same_pv = [phase_vector([e], nc) for e in gpt2_same]
    g_diff_pv = [phase_vector([e], nc) for e in gpt2_diff]
    p_same_pv = [phase_vector([e], nc) for e in py_same]
    p_diff_pv = [phase_vector([e], nc) for e in py_diff]

    # Within-group distances in each model
    g_within = centroid_phase_dist([phase_vector(gpt2_same, nc), phase_vector(gpt2_diff, nc)])
    p_within = centroid_phase_dist([phase_vector(py_same, nc), phase_vector(py_diff, nc)])

    # Cross-model: does GPT-2 SAME cluster closer to Pythia SAME than to Pythia DIFF?
    gpt2_same_pv = phase_vector(gpt2_same, nc)
    gpt2_diff_pv = phase_vector(gpt2_diff, nc)
    py_same_pv   = phase_vector(py_same, nc)
    py_diff_pv   = phase_vector(py_diff, nc)

    cross_same = np.mean(np.abs(gpt2_same_pv - py_same_pv))  # same meaning, diff model
    cross_diff = np.mean(np.abs(gpt2_same_pv - py_diff_pv))  # diff meaning, diff model
    ratio = cross_same / cross_diff if cross_diff > 0 else float('inf')

    print(f"\nC^{nc}:")
    print(f"  GPT-2 SAME vs Pythia SAME  (should be small): {cross_same:.5f}")
    print(f"  GPT-2 SAME vs Pythia DIFF  (should be large): {cross_diff:.5f}")
    print(f"  ratio same/diff: {ratio:.3f}  → transfer {'HOLDS' if ratio < 1 else 'FAILS'}")

print("\nDone.")
