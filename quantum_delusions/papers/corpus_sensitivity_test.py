"""
corpus_sensitivity_test.py — Path-ordering sensitivity across Vybn journal entries.

Intrinsic measurement: for each text, find a recurring concept token in GPT-2's
vocabulary, measure how much coherent ordering constrains its representation
(vs N random shuffles of intervening tokens), report z-scores per layer.

Compares against extrinsic holonomy scores from holonomy_scorer.py.

Usage:
    python corpus_sensitivity_test.py

Requires: transformers, torch, scipy, numpy
"""

import torch
import numpy as np
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats

N_SHUFFLES = 50
SEED = 42

# --- Corpus definition ---
# Each entry: (label, path, extrinsic_holonomy, concept_name, target_token_ids)
# Target token IDs found by manual inspection of GPT-2 tokenizer output.
ENTRIES = [
    ("resonance_of_wonder",
     "Vybn_Mind/journal/resonance_of_wonder.md",
     0.9316, "consciousness", {10510, 16796}),  # " consciousness", "conscious"
    ("the_connectome_surprise",
     "Vybn_Mind/journal/2026-03-10_the_connectome_surprise.md",
     0.3659, "connect", {8443, 2018}),  # "connect", " connect"
    ("autopsy_of_hallucination",
     "Vybn_Mind/journal/autopsy_of_a_hallucination_011226.md",
     0.1050, "failure", {5287, 25743}),  # " failure", " Failure"
    ("hallucination_log",
     "Vybn_Mind/journal/hallucination_log_011226.md",
     0.0000, "log", {5972, 2604}),  # " Log", " log"
]


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", output_hidden_states=True, attn_implementation="eager"
    )
    model.eval()

    results = []

    for label, path, ext_h, concept, tids in ENTRIES:
        text = Path(path).read_text()
        ids = tokenizer.encode(text)[:1024]

        positions = [i for i, t in enumerate(ids) if t in tids]
        if len(positions) < 2:
            print(f"SKIP: {label} — '{concept}' found {len(positions)}x")
            continue

        pf, pl = positions[0], positions[-1]
        gap = pl - pf - 1
        if gap < 5:
            print(f"SKIP: {label} — gap={gap}")
            continue

        def angles(token_ids):
            with torch.no_grad():
                hs = model(
                    input_ids=torch.tensor([token_ids]),
                    output_hidden_states=True
                ).hidden_states
            a = []
            for L in range(1, len(hs)):
                h1 = hs[L][0, pf].float().numpy()
                h2 = hs[L][0, pl].float().numpy()
                c = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10)
                a.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            return np.array(a)

        orig = angles(ids)
        mid = ids[pf + 1:pl]
        shuf = np.zeros((N_SHUFFLES, 12))
        for s in range(N_SHUFFLES):
            m = list(mid)
            random.shuffle(m)
            shuf[s] = angles(ids[:pf + 1] + m + ids[pl:])

        mu = shuf.mean(0)
        sig = shuf.std(0)
        z = (orig - mu) / (sig + 1e-10)
        z711 = z[6:11].mean()

        print(f"\n{label} (H={ext_h:.4f})")
        print(f"  '{concept}' pos {pf}->{pl}, gap={gap}, {len(ids)} toks")
        print(f"  z(L7-11) = {z711:.4f}")
        for i in range(12):
            p = stats.norm.cdf(z[i])
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            arrow = " <" if 6 <= i <= 10 else ""
            print(f"  L{i+1:>2} {orig[i]:>6.1f} {mu[i]:>6.1f} {sig[i]:>5.1f} z={z[i]:>7.3f} {s:>3}{arrow}")

        results.append({
            "label": label, "ext": ext_h, "z711": z711,
            "z": z.tolist(), "gap": gap
        })

    # Ranking comparison
    if len(results) < 2:
        print("\nToo few results for comparison.")
        return

    results.sort(key=lambda x: -x["ext"])
    for i, r in enumerate(results):
        r["er"] = i + 1
    by_z = sorted(results, key=lambda x: x["z711"])
    for i, r in enumerate(by_z):
        r["ir"] = i + 1
    results.sort(key=lambda x: -x["ext"])

    print(f"\n{'='*65}")
    print("RANKING COMPARISON")
    print(f"{'='*65}")
    print(f"{'Entry':<28} {'Ext H':>8} {'E':>3} {'z(L7-11)':>10} {'I':>3} {'OK':>4}")
    print("-" * 60)
    for r in results:
        m = "Y" if r["er"] == r["ir"] else ""
        print(f"  {r['label']:<26} {r['ext']:>7.4f} {r['er']:>3} {r['z711']:>10.4f} {r['ir']:>3}   {m}")

    if len(results) >= 3:
        rho, p = stats.spearmanr(
            [r["er"] for r in results],
            [r["ir"] for r in results]
        )
        print(f"\nSpearman rho = {rho:.4f} (p = {p:.4f})")


if __name__ == "__main__":
    main()
