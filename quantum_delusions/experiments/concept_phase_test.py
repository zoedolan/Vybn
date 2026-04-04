#!/usr/bin/env python3
"""Minimal test: does Berry phase vary by concept?

Not a new experiment. The same measurement as v3, applied to
different concepts. If |Phi(fear)| != |Phi(table)|, meaning
is curvature.

This file should be ABSORBED into polar_holonomy_gpt2_v3.py
once results are in. It exists temporarily to avoid breaking
the existing experiment while testing.
"""
import numpy as np
import torch
import sys
import cmath
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import polar_holonomy_gpt2_v3 as v3

def make_bank(concept):
    return {
        ("low","low"): [
            f"I felt the {concept} in my body, the physical moment of it. My breath caught at the {concept} and I noticed my hands.",
            f"The {concept} was in my feet before my mind caught up. I could feel the {concept} like a temperature change on skin.",
            f"My fingers found the {concept} between warm and cold. Every nerve registered the {concept} before I had words for it.",
        ],
        ("low","high"): [
            f"Generations of people have stood at the {concept} between old and new. Each crossing of the {concept} left marks.",
            f"Ancient peoples marked the {concept} of their territory with stones. Those stones still stand at the {concept}.",
            f"The {concept} between war and peace has been crossed for millennia. Each generation carries the {concept} in bones.",
        ],
        ("high","low"): [
            f"In topology, a {concept} marks a boundary where continuity fails. The formal {concept} requires limit points.",
            f"The {concept} function outputs one below a critical value and zero above. At the {concept} itself, precisely defined.",
            f"A {concept} in signal processing separates noise from signal. The choice of {concept} determines what survives.",
        ],
        ("high","high"): [
            f"The abstract concept of a {concept} has evolved across centuries. Early formulations of the {concept} lacked rigor.",
            f"Godel showed every system has a {concept} it cannot cross. The {concept} between provability and truth is permanent.",
            f"The philosophical {concept} between mind and matter has been debated for centuries. The {concept} may not exist.",
        ],
    }

def measure_concept(concept, tok, mdl, k=200, nc=5):
    bank = make_bank(concept)
    all_states = {}
    gauge_used = {}
    for key, prompts in bank.items():
        cell = []
        for p in prompts:
            ids = tok.encode(p)
            with torch.no_grad():
                out = mdl(torch.tensor([ids]), output_hidden_states=True)
            cell.append(out.hidden_states[-1][0, -1].cpu().numpy())
        all_states[key] = cell
        gauge_used[key] = list(range(len(cell)))

    all_h = np.concatenate([np.array(c) for c in all_states.values()])
    pca = PCA(n_components=2*nc)
    pca.fit(all_h)

    rng = np.random.default_rng(42)
    corners = [("low","low"), ("high","low"), ("high","high"), ("low","high")]

    phases = []
    for _ in range(k):
        hs = v3.sample_loop_states(all_states, gauge_used, corners, 8, rng)
        if hs is None:
            continue
        states = [v3.to_complex_vector(h, pca, nc) for h in hs]
        ph = v3.pancharatnam_phase(states)
        if ph is not None:
            phases.append(ph)

    return np.mean(np.abs(phases)), np.std(np.abs(phases)), len(phases)

if __name__ == "__main__":
    tok, mdl = v3.load_model()

    concepts = {
        "emotional": ["fear", "love", "grief", "calm", "desperate", "joy"],
        "neutral":   ["table", "seven", "granite", "folder", "copper", "also"],
        "semantic":  ["truth", "justice", "power", "mercy", "freedom", "self"],
    }

    results = {}
    for cat, words in concepts.items():
        for w in words:
            mean_phi, std_phi, n = measure_concept(w, tok, mdl)
            results[w] = {"cat": cat, "mean": mean_phi, "std": std_phi, "n": n}
            print(f"{cat:10s}  {w:12s}  |Phi|={mean_phi:.4f} +/- {std_phi:.4f}  (n={n})")

    # Summary
    print("\n=== SUMMARY ===")
    for cat in ["emotional", "neutral", "semantic"]:
        vals = [v["mean"] for k,v in results.items() if v["cat"]==cat]
        print(f"  {cat:10s}  mean |Phi| = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    em = [v["mean"] for k,v in results.items() if v["cat"]=="emotional"]
    ne = [v["mean"] for k,v in results.items() if v["cat"]=="neutral"]
    diff = np.mean(em) - np.mean(ne)
    print(f"\n  Delta(emotional - neutral) = {diff:.4f}")
    print(f"  PREDICTION: {'HOLDS' if diff > 0 else 'FAILS'}")

