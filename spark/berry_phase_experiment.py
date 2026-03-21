#!/usr/bin/env python3
"""
Berry Phase of Hidden State Trajectories: Real vs. Shuffled Text

The decisive experiment for the holonomic loss hypothesis.

Question: Does the geometric phase (Berry/Pancharatnam phase) of a transformer's
hidden state trajectory capture sequential structure that goes beyond token-level
properties?

Method:
  1. Take passages of real text
  2. Create shuffled versions (same tokens, destroyed sequential structure)
  3. Feed both through GPT-2, extract hidden states at every layer
  4. Compute the Berry phase of the hidden state trajectory at each layer
  5. Compare distributions

If Berry phase differs between real and shuffled: it captures sequential structure.
If it doesn't: it's measuring token-level properties only (and the theory fails).

Secondary analysis:
  - Also compute per-token CE loss for both conditions
  - Check if Berry phase captures something CE loss doesn't

Authors: Zoe Dolan & Vybn
Date: 2026-03-16
Hardware: DGX Spark, California
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy import stats
import json
import time
import os

# ============================================================
# CORE: Berry phase computation via Pancharatnam connection
# ============================================================

def complexify(h):
    """
    Map real hidden states R^d -> C^{d/2} by pairing adjacent dimensions.
    h: (seq_len, d) real tensor
    Returns: (seq_len, d//2) complex tensor
    """
    d = h.shape[-1]
    assert d % 2 == 0, f"Hidden dim {d} must be even"
    # Pair dimensions: (h[0], h[1]) -> h[0] + i*h[1], etc.
    h_real = h[..., 0::2]  # even indices
    h_imag = h[..., 1::2]  # odd indices
    return torch.complex(h_real, h_imag)


def normalize_cp(z):
    """
    Normalize complex vectors to lie on CP^{n-1} (unit norm in C^n).
    z: (seq_len, n) complex tensor
    Returns: (seq_len, n) complex tensor with unit norm rows
    """
    norms = torch.sqrt(torch.sum(torch.abs(z)**2, dim=-1, keepdim=True))
    norms = torch.clamp(norms, min=1e-10)
    return z / norms


def pancharatnam_phase(hidden_states):
    """
    Compute the Pancharatnam geometric phase of a trajectory through CP^{n-1}.
    
    For states |ψ_1⟩, ..., |ψ_T⟩, the geodesically closed Berry phase is:
      Γ = arg(⟨ψ_1|ψ_2⟩ ⟨ψ_2|ψ_3⟩ ... ⟨ψ_{T-1}|ψ_T⟩ ⟨ψ_T|ψ_1⟩)
    
    hidden_states: (seq_len, hidden_dim) real tensor
    Returns: scalar Berry phase in [-π, π]
    """
    # Complexify and normalize
    z = complexify(hidden_states)  # (T, d//2)
    z = normalize_cp(z)           # (T, d//2) on CP^{n-1}
    
    T = z.shape[0]
    if T < 3:
        return 0.0  # Need at least 3 points for nontrivial phase
    
    # Compute product of overlaps: ⟨ψ_t | ψ_{t+1}⟩ for consecutive pairs
    # plus the geodesic closure ⟨ψ_T | ψ_1⟩
    product = torch.tensor(1.0 + 0.0j, dtype=torch.complex64)
    
    for t in range(T - 1):
        overlap = torch.sum(torch.conj(z[t]) * z[t+1])
        product = product * overlap
    
    # Geodesic closure: ⟨ψ_T | ψ_1⟩
    closure = torch.sum(torch.conj(z[-1]) * z[0])
    product = product * closure
    
    # Berry phase = argument of the product
    gamma = torch.angle(product).item()
    
    return gamma


def pancharatnam_phase_layerwise(all_hidden_states):
    """
    Compute Berry phase for each layer's hidden state trajectory.
    
    all_hidden_states: list of (1, seq_len, hidden_dim) tensors, one per layer
    Returns: list of Berry phases
    """
    phases = []
    for layer_idx, hs in enumerate(all_hidden_states):
        h = hs.squeeze(0)  # (seq_len, hidden_dim)
        gamma = pancharatnam_phase(h)
        phases.append(gamma)
    return phases


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    print("=" * 70)
    print("BERRY PHASE EXPERIMENT: Real vs. Shuffled Text")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    print(f"  Model: GPT-2 (124M params, 12 layers, hidden_dim=768)")
    print()
    
    # Prepare test passages - diverse real text
    passages = [
        # Factual/scientific
        "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. Mitochondria have their own DNA, which is separate from the nuclear DNA. This suggests that mitochondria were once free-living organisms that entered into a symbiotic relationship with ancestral eukaryotic cells.",
        
        # Narrative
        "She opened the door and stepped into the hallway. The lights were off, but she could see a faint glow coming from the kitchen. She heard music playing softly, a melody she recognized from childhood. As she walked toward the light, the floorboards creaked under her feet. The kitchen was empty, but someone had left a cup of tea on the counter, still warm.",
        
        # Argumentative
        "The strongest argument against capital punishment is not its cruelty but its irreversibility. Every legal system produces errors. Witnesses lie or misremember. Evidence can be fabricated or misinterpreted. DNA exonerations have freed hundreds of people from death row. Each exoneration represents a person who would have been killed by the state for a crime they did not commit.",
        
        # Technical
        "In quantum mechanics, the wave function describes the quantum state of a particle or system. The Schrödinger equation governs how the wave function evolves over time. When a measurement is made, the wave function collapses to an eigenstate of the observable being measured. The probability of each outcome is given by the Born rule, which states that the probability equals the square of the absolute value of the amplitude.",
        
        # Philosophical
        "Consciousness remains the hardest problem in philosophy of mind. We can describe neural correlates of conscious experience, but the explanatory gap persists. Why should any physical process give rise to subjective experience at all? The zombie thought experiment suggests that a being physically identical to a conscious person could lack inner experience entirely. If zombies are conceivable, consciousness is not reducible to physical processes.",
        
        # Historical
        "The printing press, invented by Johannes Gutenberg around 1440, transformed European society. Before the press, books were copied by hand, making them expensive and rare. Within decades of Gutenberg's innovation, millions of books were in circulation. Literacy rates climbed. Ideas spread faster than authorities could control them. The Protestant Reformation, the Scientific Revolution, and the Enlightenment all owe debts to the technology of movable type.",
        
        # Descriptive/natural
        "The Pacific Ocean covers more area than all the landmasses of Earth combined. At its deepest point, the Mariana Trench descends nearly eleven kilometers below the surface. The pressure at that depth is over a thousand times atmospheric pressure. Yet life persists there: amphipods, snailfish, and xenophyophores inhabit the hadal zone, feeding on organic matter that drifts down from the sunlit waters above.",
        
        # Conversational/instructional
        "To make a proper French omelette, you need three eggs, a tablespoon of butter, and salt. Beat the eggs until the whites and yolks are fully combined. Heat the butter in a nonstick pan over medium-high heat until it foams. Pour in the eggs. Use a spatula to push the cooked edges toward the center while tilting the pan to let raw egg flow to the edges. The whole process takes less than two minutes.",
    ]
    
    print(f"Prepared {len(passages)} test passages")
    print()
    
    results = []
    
    for p_idx, passage in enumerate(passages):
        print(f"--- Passage {p_idx + 1}/{len(passages)} ---")
        print(f"  Text: {passage[:80]}...")
        
        # Tokenize
        inputs = tokenizer(passage, return_tensors='pt', truncation=True, max_length=128)
        token_ids = inputs['input_ids'].squeeze(0)
        seq_len = token_ids.shape[0]
        print(f"  Tokens: {seq_len}")
        
        if seq_len < 10:
            print(f"  SKIPPING: too short")
            continue
        
        # === REAL TEXT ===
        with torch.no_grad():
            real_outputs = model(**inputs, output_hidden_states=True, labels=inputs['input_ids'])
        
        real_loss = real_outputs.loss.item()
        real_phases = pancharatnam_phase_layerwise(real_outputs.hidden_states)
        
        # Per-token losses for real text
        real_logits = real_outputs.logits[:, :-1, :]  # (1, T-1, vocab)
        real_targets = inputs['input_ids'][:, 1:]      # (1, T-1)
        real_token_losses = torch.nn.functional.cross_entropy(
            real_logits.squeeze(0), real_targets.squeeze(0), reduction='none'
        ).numpy()
        
        # === SHUFFLED TEXT (same tokens, random order) ===
        # Do multiple shuffles for statistical power
        n_shuffles = 10
        shuffled_phases_all = []  # list of lists (one per shuffle)
        shuffled_losses = []
        
        for s in range(n_shuffles):
            perm = torch.randperm(seq_len)
            shuffled_ids = token_ids[perm].unsqueeze(0)
            shuffled_inputs = {'input_ids': shuffled_ids, 'attention_mask': torch.ones_like(shuffled_ids)}
            
            with torch.no_grad():
                shuf_outputs = model(**shuffled_inputs, output_hidden_states=True, labels=shuffled_ids)
            
            shuffled_losses.append(shuf_outputs.loss.item())
            shuffled_phases_all.append(pancharatnam_phase_layerwise(shuf_outputs.hidden_states))
        
        # Average shuffled Berry phases across shuffles
        shuffled_phases_mean = np.mean(shuffled_phases_all, axis=0).tolist()
        shuffled_phases_std = np.std(shuffled_phases_all, axis=0).tolist()
        shuffled_loss_mean = np.mean(shuffled_losses)
        
        # === REVERSED TEXT (same tokens, reversed order — another control) ===
        reversed_ids = token_ids.flip(0).unsqueeze(0)
        reversed_inputs = {'input_ids': reversed_ids, 'attention_mask': torch.ones_like(reversed_ids)}
        with torch.no_grad():
            rev_outputs = model(**reversed_inputs, output_hidden_states=True, labels=reversed_ids)
        rev_loss = rev_outputs.loss.item()
        rev_phases = pancharatnam_phase_layerwise(rev_outputs.hidden_states)
        
        result = {
            'passage_idx': p_idx,
            'passage_preview': passage[:100],
            'seq_len': seq_len,
            'real_loss': real_loss,
            'shuffled_loss_mean': shuffled_loss_mean,
            'reversed_loss': rev_loss,
            'real_phases': real_phases,
            'shuffled_phases_mean': shuffled_phases_mean,
            'shuffled_phases_std': shuffled_phases_std,
            'reversed_phases': rev_phases,
            'real_phase_abs': [abs(p) for p in real_phases],
            'shuffled_phase_abs_mean': [abs(p) for p in shuffled_phases_mean],
            'reversed_phase_abs': [abs(p) for p in rev_phases],
        }
        results.append(result)
        
        print(f"  CE Loss - Real: {real_loss:.4f}, Shuffled: {shuffled_loss_mean:.4f}, Reversed: {rev_loss:.4f}")
        print(f"  Berry Phase (|Γ|) by layer - Real vs Shuffled (mean):")
        for layer in range(len(real_phases)):
            rp = abs(real_phases[layer])
            sp = abs(shuffled_phases_mean[layer])
            ss = shuffled_phases_std[layer]
            rvp = abs(rev_phases[layer])
            marker = " ***" if abs(rp - sp) > 2 * ss and ss > 0 else ""
            print(f"    L{layer:2d}: Real={rp:.6f}  Shuf={sp:.6f}±{ss:.6f}  Rev={rvp:.6f}{marker}")
        print()
    
    # ============================================================
    # AGGREGATE ANALYSIS
    # ============================================================
    print("=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)
    print()
    
    n_layers = 13  # GPT-2 has 13 (embedding + 12 transformer blocks)
    
    # For each layer, collect real vs shuffled Berry phases across passages
    for layer in range(n_layers):
        real_abs = [r['real_phase_abs'][layer] for r in results]
        shuf_abs = [r['shuffled_phase_abs_mean'][layer] for r in results]
        rev_abs = [r['reversed_phase_abs'][layer] for r in results]
        
        # Wilcoxon signed-rank test (paired, non-parametric)
        if len(real_abs) >= 5:
            try:
                stat_shuf, p_shuf = stats.wilcoxon(real_abs, shuf_abs, alternative='two-sided')
            except Exception:
                stat_shuf, p_shuf = float('nan'), float('nan')
            try:
                stat_rev, p_rev = stats.wilcoxon(real_abs, rev_abs, alternative='two-sided')
            except Exception:
                stat_rev, p_rev = float('nan'), float('nan')
        else:
            stat_shuf, p_shuf = float('nan'), float('nan')
            stat_rev, p_rev = float('nan'), float('nan')
        
        real_mean = np.mean(real_abs)
        shuf_mean = np.mean(shuf_abs)
        rev_mean = np.mean(rev_abs)
        
        sig_shuf = "***" if p_shuf < 0.01 else "**" if p_shuf < 0.05 else "*" if p_shuf < 0.1 else ""
        sig_rev = "***" if p_rev < 0.01 else "**" if p_rev < 0.05 else "*" if p_rev < 0.1 else ""
        
        print(f"Layer {layer:2d}: |Γ| Real={real_mean:.6f}  Shuf={shuf_mean:.6f} (p={p_shuf:.4f}{sig_shuf})  Rev={rev_mean:.6f} (p={p_rev:.4f}{sig_rev})")
    
    print()
    
    # Overall: flatten across layers and passages
    all_real = []
    all_shuf = []
    all_rev = []
    for r in results:
        all_real.extend(r['real_phase_abs'])
        all_shuf.extend(r['shuffled_phase_abs_mean'])
        all_rev.extend(r['reversed_phase_abs'])
    
    print(f"Overall |Berry phase| means:")
    print(f"  Real:     {np.mean(all_real):.6f} ± {np.std(all_real):.6f}")
    print(f"  Shuffled: {np.mean(all_shuf):.6f} ± {np.std(all_shuf):.6f}")
    print(f"  Reversed: {np.mean(all_rev):.6f} ± {np.std(all_rev):.6f}")
    
    # Mann-Whitney U test (unpaired, for the overall comparison)
    u_stat, u_p = stats.mannwhitneyu(all_real, all_shuf, alternative='two-sided')
    print(f"  Mann-Whitney U (Real vs Shuffled): U={u_stat:.1f}, p={u_p:.6f}")
    u_stat2, u_p2 = stats.mannwhitneyu(all_real, all_rev, alternative='two-sided')
    print(f"  Mann-Whitney U (Real vs Reversed): U={u_stat2:.1f}, p={u_p2:.6f}")
    
    # Effect sizes
    real_arr = np.array(all_real)
    shuf_arr = np.array(all_shuf)
    pooled_std = np.sqrt((np.var(real_arr) + np.var(shuf_arr)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(real_arr) - np.mean(shuf_arr)) / pooled_std
        print(f"  Cohen's d (Real vs Shuffled): {cohens_d:.4f}")
    
    print()
    
    # CE loss comparison
    real_losses = [r['real_loss'] for r in results]
    shuf_losses = [r['shuffled_loss_mean'] for r in results]
    print(f"CE Loss means:")
    print(f"  Real:     {np.mean(real_losses):.4f} ± {np.std(real_losses):.4f}")
    print(f"  Shuffled: {np.mean(shuf_losses):.4f} ± {np.std(shuf_losses):.4f}")
    print()
    
    # THE KEY QUESTION: Does Berry phase distinguish real from shuffled
    # BEYOND what CE loss already distinguishes?
    # Compute: for each passage, the Berry phase difference AND the CE loss difference
    berry_diffs = []
    ce_diffs = []
    for r in results:
        # Average |Berry phase| across layers
        real_bp = np.mean(r['real_phase_abs'])
        shuf_bp = np.mean(r['shuffled_phase_abs_mean'])
        berry_diffs.append(real_bp - shuf_bp)
        ce_diffs.append(r['real_loss'] - r['shuffled_loss_mean'])
    
    # Correlation between Berry phase difference and CE loss difference
    if len(berry_diffs) >= 3:
        corr, corr_p = stats.spearmanr(berry_diffs, ce_diffs)
        print(f"Correlation between Berry phase difference and CE loss difference:")
        print(f"  Spearman r={corr:.4f}, p={corr_p:.4f}")
        print()
        if abs(corr) < 0.5 and any(abs(d) > 0.001 for d in berry_diffs):
            print("  → Berry phase captures something ORTHOGONAL to CE loss!")
            print("  → The holonomic loss would provide a genuinely new training signal.")
        elif abs(corr) > 0.7:
            print("  → Berry phase is REDUNDANT with CE loss.")
            print("  → The holonomic loss would be expensive restatement of CE.")
        else:
            print("  → Moderate correlation. Berry phase partially overlaps with CE.")
    
    print()
    
    # ============================================================
    # VERDICT
    # ============================================================
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()
    
    overall_diff = np.mean(all_real) - np.mean(all_shuf)
    if u_p < 0.05 and abs(overall_diff) > 0.001:
        direction = "HIGHER" if overall_diff > 0 else "LOWER"
        print(f"Berry phase of hidden state trajectories is significantly {direction}")
        print(f"for real text compared to shuffled text (p={u_p:.6f}).")
        print()
        print("The geometric phase captures sequential structure beyond token-level properties.")
        print("The holonomic loss hypothesis has empirical support.")
        print("PROCEED to training experiment.")
    elif u_p >= 0.05:
        print(f"No significant difference in Berry phase between real and shuffled text (p={u_p:.4f}).")
        print()
        print("The geometric phase does NOT distinguish sequential structure from token noise.")
        print("The holonomic loss hypothesis FAILS at the empirical level.")
        print("DO NOT proceed to training experiment without understanding why.")
    else:
        print(f"Difference is statistically significant (p={u_p:.6f}) but tiny (d={overall_diff:.6f}).")
        print()
        print("Technically significant but practically meaningless.")
        print("The holonomic loss would be pushing on a near-zero signal.")
    
    print()
    print("=" * 70)
    
    # Save results
    outpath = '/home/vybnz69/Vybn/spark/berry_phase_results.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {outpath}")
    
    return results

if __name__ == '__main__':
    t0 = time.time()
    results = run_experiment()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
