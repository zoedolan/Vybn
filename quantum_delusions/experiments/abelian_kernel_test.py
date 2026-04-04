"""Test: is the proposition an abelian kernel?

Same meaning, different word order → measure geometric phase.
If same-meaning loops have less phase than different-meaning loops,
the proposition is a geometric invariant and serialization is artifact.
"""
import numpy as np
import torch
import cmath
from transformers import GPT2Tokenizer, GPT2Model

# Same proposition, different serializations
SAME = [
    "She is a lawyer and a runner.",
    "She is a runner and a lawyer.",
    "A lawyer and a runner, that is what she is.",
    "A runner and a lawyer, she is.",
    "What she is: a runner, a lawyer.",
    "Lawyer, runner — both describe her.",
    "Runner first, lawyer second, but she is both.",
    "She is both a lawyer and a runner.",
]

# Different propositions, matched structure
DIFF = [
    "He is a doctor and a painter.",
    "The cat is small and fast.",
    "They are teachers and musicians.",
    "She is a pilot and a swimmer.",
    "He is a writer and a cook.",
    "The building is old and tall.",
    "We are students and athletes.",
    "She is a nurse and a singer.",
]

def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].numpy()

def to_complex(h, n=8):
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z

def pancharatnam_phase(states):
    prod = complex(1.0, 0.0)
    for k in range(len(states)):
        overlap = np.vdot(states[k], states[(k+1) % len(states)])
        prod *= overlap
    return cmath.phase(prod)

def measure_loops(embeddings, n_loops=200, loop_size=4, n_complex=8):
    rng = np.random.default_rng(42)
    states = [to_complex(e, n_complex) for e in embeddings]
    phases = []
    for _ in range(n_loops):
        idx = rng.choice(len(states), size=loop_size, replace=False)
        loop = [states[i] for i in idx]
        phases.append(pancharatnam_phase(loop))
    return np.array(phases)

if __name__ == "__main__":
    print("Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    mdl.eval()

    print("Embedding sentences...")
    same_emb = [get_embedding(mdl, tok, s) for s in SAME]
    diff_emb = [get_embedding(mdl, tok, s) for s in DIFF]

    for nc in [4, 8, 16]:
        same_phases = measure_loops(same_emb, n_complex=nc)
        diff_phases = measure_loops(diff_emb, n_complex=nc)

        print(f"\nC^{nc}:")
        print(f"  SAME meaning:  mean|Phi|={np.mean(np.abs(same_phases)):.4f}  std={np.std(np.abs(same_phases)):.4f}")
        print(f"  DIFF meaning:  mean|Phi|={np.mean(np.abs(diff_phases)):.4f}  std={np.std(np.abs(diff_phases)):.4f}")
        ratio = np.mean(np.abs(same_phases)) / np.mean(np.abs(diff_phases))
        print(f"  ratio (same/diff): {ratio:.3f}")
        print(f"  prediction: same < diff → {'HOLDS' if ratio < 1 else 'FAILS'}")
