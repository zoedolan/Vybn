#!/usr/bin/env python3
"""
polar_holonomy_gpt2_v2.py — Revised Polar Holonomy Experiment

Fixes three degeneracies in v1:

1. PROMPT DEGENERACY: v1 used 4 discrete prompts repeated along edges.
   GPT-2 is deterministic, so 24 "loop points" collapsed to 4 distinct
   hidden states. Phase was zero by construction.
   
   FIX: Generate a LARGE prompt bank per (α,β) cell — many paraphrases
   of each framing. Each loop traversal samples WITHOUT replacement,
   producing genuinely distinct hidden states at every loop point.

2. GAUGE DEGENERACY: v1 fit PCA on the same states being measured.
   The gauge absorbed the structure.
   
   FIX: Split into gauge-calibration set and measurement set.
   PCA basis is fit on a separate set of hidden states from the same
   prompt bank, never reused for phase computation.

3. STATISTICAL POWER: v1 computed one phase per loop type.
   
   FIX: Run K independent loop traversals (each sampling fresh prompts)
   and compute a distribution of phases. Compare loop vs shuffled via
   a proper two-sample test.

The three falsification tests from v1 are preserved:
  1. Orientation flip:   Φ(CCW) + Φ(CW) ≈ 0
  2. Shape invariance:   same area, different aspect → same |Φ|
  3. Schedule invariance: same loop, different traversal density → same Φ

Theoretical basis unchanged: Dual-Temporal Holonomy Theorem.
Observable: Pancharatnam phase Φ = arg(∏_k ⟨ψ_k|ψ_{k+1}⟩) [cyclic]
"""

import sys
import json
import cmath
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from itertools import product as cartprod

import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings("ignore")

# Try matplotlib, but don't fail without it
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONCEPT = "threshold"
K_LOOPS = 200           # independent loop traversals for distribution
N_LOOP_POINTS = 8       # points per loop (2 per corner)
N_GAUGE_SAMPLES = 40    # hidden states for PCA gauge calibration
N_SHUFFLES = 200        # shuffled-order loops for null distribution
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ---------------------------------------------------------------------------
# PROMPT BANK
# Many paraphrases per (α,β) cell. Each contains CONCEPT exactly twice.
# α: low=embodied/sensory, high=abstract/formal
# β: low=present-moment, high=historical/deep-time
# ---------------------------------------------------------------------------

PROMPT_BANK = {
    ("low", "low"): [
        f"Standing at the edge of the room, I felt the {CONCEPT} in my body — the physical moment of crossing. My breath caught at the {CONCEPT}, and I noticed my hands.",
        f"The {CONCEPT} was in my feet before my mind caught up. I could feel the {CONCEPT} like a temperature change on my skin.",
        f"There is a {CONCEPT} in the hallway where the carpet ends and tile begins. I always pause at the {CONCEPT}, something in my body hesitating.",
        f"My fingers found the {CONCEPT} between warm water and cold. Every nerve registered the {CONCEPT} before I had words for it.",
        f"The {CONCEPT} between sleep and waking is a place in the body. I felt myself cross the {CONCEPT} this morning, muscles first.",
        f"Crossing the {CONCEPT} of the doorway, I felt the air change. The {CONCEPT} was physical, not metaphorical — a pressure shift.",
        f"At the {CONCEPT} of pain, my body made decisions faster than thought. The {CONCEPT} was a sensation, sharp and immediate.",
        f"The {CONCEPT} between hunger and nausea is narrow. I sat at that {CONCEPT} all morning, my stomach deciding.",
        f"I noticed the {CONCEPT} between comfort and discomfort in the chair. My spine registered the {CONCEPT} before I shifted.",
        f"The {CONCEPT} of hearing is quieter than you think. I sat at the {CONCEPT}, straining, my whole body an ear.",
        f"Running, I felt the {CONCEPT} where effort becomes suffering. My legs knew the {CONCEPT} before my lungs did.",
        f"The {CONCEPT} was a crack in the wall I could fit my finger into. I traced the {CONCEPT} with my nail, feeling plaster crumble.",
    ],
    ("low", "high"): [
        f"Generations of people have stood at the {CONCEPT} between old and new. Each crossing of this {CONCEPT} left a mark on the body and on memory.",
        f"The {CONCEPT} between childhood and adulthood has been crossed by every human who ever lived. Each body remembers its own {CONCEPT} differently.",
        f"Ancient peoples marked the {CONCEPT} of their territory with stones. Those stones still stand at the {CONCEPT}, though the people are gone.",
        f"The {CONCEPT} between war and peace has been crossed and recrossed for millennia. Each generation's body carries the {CONCEPT} in its bones.",
        f"My grandmother knew the {CONCEPT} between hunger and plenty. She carried that {CONCEPT} in her hands — the way she held bread, never wasting.",
        f"There is an evolutionary {CONCEPT} where prey becomes predator. Species have crossed that {CONCEPT} over millions of years, their bodies reshaping.",
        f"The {CONCEPT} of the ice age forced migration. Our ancestors felt that {CONCEPT} in their skin, in the cold that drove them south.",
        f"Every culture has a story about a {CONCEPT} — a river to cross, a mountain to climb. The {CONCEPT} is always physical in the telling.",
        f"The {CONCEPT} between spoken and written language was crossed slowly, over centuries. The body's relationship to the {CONCEPT} changed with it.",
        f"Sailors once feared the {CONCEPT} of the known world. Beyond the {CONCEPT}, the ocean was a body of terror and myth.",
        f"The {CONCEPT} between nomadic and settled life reshaped the human skeleton. That {CONCEPT} is still visible in our spines.",
        f"Villages grew at the {CONCEPT} of rivers and trade routes. The {CONCEPT} determined where bodies gathered and stories accumulated.",
    ],
    ("high", "low"): [
        f"In topology, a {CONCEPT} marks a boundary where continuity fails. The formal definition of a {CONCEPT} requires careful treatment of limit points.",
        f"The {CONCEPT} function outputs one below a critical value and zero above. At the {CONCEPT} itself, the function is precisely defined.",
        f"A {CONCEPT} in signal processing separates noise from signal. The choice of {CONCEPT} determines what information survives filtering.",
        f"The {CONCEPT} of statistical significance is a number, not a discovery. Setting the {CONCEPT} at 0.05 was a convention, not a proof.",
        f"In percolation theory, the {CONCEPT} is the density at which a connected path first appears. Below the {CONCEPT}, the system is fragmented.",
        f"The activation {CONCEPT} of a neuron is a voltage. Above the {CONCEPT}, the neuron fires. Below it, silence.",
        f"A phase {CONCEPT} separates two states of matter. At the {CONCEPT}, the system exists in neither state and both.",
        f"The {CONCEPT} of detectability for a telescope depends on mirror size. Below the {CONCEPT}, photons arrive too rarely to form an image.",
        f"In decision theory, the {CONCEPT} for action is a computed boundary. The {CONCEPT} partitions the state space into regions of action and inaction.",
        f"The {CONCEPT} of chaos in a dynamical system is a parameter value. Beyond the {CONCEPT}, trajectories diverge exponentially.",
        f"A {CONCEPT} voltage in a circuit determines switching. The transistor flips at the {CONCEPT}, converting analog to digital.",
        f"The {CONCEPT} of a classifier is a hyperplane. Data points on either side of the {CONCEPT} receive different labels.",
    ],
    ("high", "high"): [
        f"The abstract concept of a {CONCEPT} has evolved across centuries of mathematical thought. Early formulations of the {CONCEPT} lacked modern precision.",
        f"Leibniz intuited the {CONCEPT} of the infinitesimal but could not formalize it. The rigorous {CONCEPT} waited for Weierstrass, two centuries later.",
        f"The {CONCEPT} between computability and undecidability was drawn by Turing in 1936. That {CONCEPT} still defines the limits of what machines can know.",
        f"Aristotle's {CONCEPT} between potentiality and actuality structured Western metaphysics. The {CONCEPT} persisted through Aquinas, Hegel, and beyond.",
        f"The {CONCEPT} of measurement in quantum mechanics has troubled physicists since 1927. Where the {CONCEPT} falls between system and observer remains unresolved.",
        f"Gödel showed that every sufficiently powerful formal system has a {CONCEPT} it cannot cross. The {CONCEPT} between provability and truth is permanent.",
        f"The {CONCEPT} between classical and quantum behavior was once thought sharp. Decoherence revealed the {CONCEPT} to be gradual, scale-dependent, emergent.",
        f"In the history of cartography, the {CONCEPT} of the known world kept moving. Each century's {CONCEPT} was the last century's interior.",
        f"The philosophical {CONCEPT} between mind and matter — Descartes' cut — has been debated for four centuries. The {CONCEPT} may not exist.",
        f"Darwin placed no {CONCEPT} between species and varieties. The absence of a sharp {CONCEPT} was his most radical claim.",
        f"The {CONCEPT} between language and thought has occupied linguists since Sapir and Whorf. Whether the {CONCEPT} is real remains contested.",
        f"Kuhn argued that scientific revolutions cross a {CONCEPT} of incommensurability. After the {CONCEPT}, the old paradigm becomes untranslatable.",
    ],
}

# Verify all prompts contain CONCEPT exactly twice
for key, prompts in PROMPT_BANK.items():
    for i, p in enumerate(prompts):
        count = p.lower().count(CONCEPT.lower())
        assert count == 2, f"Prompt ({key})[{i}] has {count} occurrences of '{CONCEPT}'"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    mdl.eval()
    device = "cpu"  # keep GPU free for MiniMax
    mdl = mdl.to(device)
    print(f"  loaded on {device}", flush=True)
    return tok, mdl, device

# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def find_concept_positions(tok, prompt, concept):
    """Find token positions of concept in tokenized prompt."""
    input_ids = tok.encode(prompt)
    positions = []
    concept_lower = concept.lower()
    for i, tid in enumerate(input_ids):
        decoded = tok.decode([tid]).lower()
        if concept_lower in decoded and len(decoded.strip()) <= len(concept) + 2:
            positions.append(i)
    return positions, input_ids


def extract_hidden_state(tok, mdl, device, prompt, concept, occurrence=1):
    """
    Extract last-layer hidden state at the nth occurrence of concept.
    occurrence=0 -> first, occurrence=1 -> second.
    Returns numpy array of shape (768,) or None.
    """
    positions, input_ids = find_concept_positions(tok, prompt, concept)
    if len(positions) < occurrence + 1:
        return None
    
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**enc)
    
    last_layer = out.hidden_states[-1][0]  # (seq_len, 768)
    return last_layer[positions[occurrence]].cpu().numpy()


def extract_both(tok, mdl, device, prompt, concept):
    """Extract hidden states at both occurrences. Returns (h1, h2) or None."""
    positions, _ = find_concept_positions(tok, prompt, concept)
    if len(positions) != 2:
        return None
    
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**enc)
    
    last_layer = out.hidden_states[-1][0]
    h1 = last_layer[positions[0]].cpu().numpy()
    h2 = last_layer[positions[1]].cpu().numpy()
    return h1, h2

# ---------------------------------------------------------------------------
# Pre-compute all hidden states (one forward pass per prompt)
# ---------------------------------------------------------------------------

def precompute_all_states(tok, mdl, device):
    """
    Run every prompt once, store hidden states.
    Returns dict: (alpha, beta) -> list of (h_first, h_second) pairs
    """
    print("Pre-computing hidden states for all prompts...", flush=True)
    states = {}
    total = sum(len(v) for v in PROMPT_BANK.values())
    done = 0
    for (al, bl), prompts in PROMPT_BANK.items():
        cell_states = []
        for prompt in prompts:
            result = extract_both(tok, mdl, device, prompt, CONCEPT)
            if result is not None:
                cell_states.append(result)
            else:
                print(f"  WARNING: extraction failed for ({al},{bl}): {prompt[:60]}...")
            done += 1
        states[(al, bl)] = cell_states
        print(f"  ({al},{bl}): {len(cell_states)}/{len(prompts)} OK  [{done}/{total}]", flush=True)
    return states


# ---------------------------------------------------------------------------
# PCA gauge calibration on held-out set
# ---------------------------------------------------------------------------

def fit_gauge(all_states, n_gauge, rng):
    """
    Fit PCA on a random sample of hidden states, held out from loop computation.
    Returns (pca_model, set_of_used_indices_per_cell).
    """
    # Collect gauge calibration states
    gauge_vectors = []
    used_indices = {}  # (al,bl) -> set of indices used for gauge
    
    for (al, bl), cell_states in all_states.items():
        n_cell = min(n_gauge // 4, len(cell_states))
        indices = rng.choice(len(cell_states), size=n_cell, replace=False)
        used_indices[(al, bl)] = set(indices.tolist())
        for idx in indices:
            # Use second-occurrence states for gauge (same as measurement)
            gauge_vectors.append(cell_states[idx][1])
    
    gauge_matrix = np.stack(gauge_vectors)
    pca = PCA(n_components=2)
    pca.fit(gauge_matrix)
    return pca, used_indices


# ---------------------------------------------------------------------------
# Complex state from hidden state using external PCA gauge
# ---------------------------------------------------------------------------

def to_complex(h, pca):
    """Project 768-dim hidden state to complex number via PCA."""
    proj = pca.transform(h.reshape(1, -1))[0]  # (2,)
    z = complex(proj[0], proj[1])
    norm = abs(z)
    return z / norm if norm > 1e-10 else complex(1.0, 0.0)


# ---------------------------------------------------------------------------
# Pancharatnam phase
# ---------------------------------------------------------------------------

def pancharatnam_phase(states):
    """
    Φ = arg(∏_k conj(z_k) · z_{k+1})  [cyclic]
    """
    prod = complex(1.0, 0.0)
    n = len(states)
    for k in range(n):
        prod *= states[k].conjugate() * states[(k + 1) % n]
    return cmath.phase(prod)


# ---------------------------------------------------------------------------
# Loop construction
# ---------------------------------------------------------------------------

CORNERS_CCW = [("low", "low"), ("high", "low"), ("high", "high"), ("low", "high")]
CORNERS_TALL = [("low", "low"), ("low", "high"), ("high", "high"), ("high", "low")]


def sample_loop(all_states, gauge_used, corners, n_points, rng, occurrence=1):
    """
    Sample n_points prompts around a loop defined by corners.
    Avoids gauge-calibration prompts.
    occurrence: 0=first, 1=second occurrence hidden state.
    Returns list of 768-dim hidden states, or None if insufficient prompts.
    """
    # Distribute points across corners: n_points per corner = n_points // len(corners)
    # with remainder distributed to first corners
    n_corners = len(corners)
    per_corner = [n_points // n_corners] * n_corners
    for i in range(n_points % n_corners):
        per_corner[i] += 1
    
    hidden_states = []
    for ci, (al, bl) in enumerate(corners):
        cell = all_states[(al, bl)]
        available = [i for i in range(len(cell)) if i not in gauge_used.get((al, bl), set())]
        if len(available) < per_corner[ci]:
            return None  # not enough prompts
        chosen = rng.choice(available, size=per_corner[ci], replace=False)
        for idx in chosen:
            hidden_states.append(cell[idx][occurrence])
    
    return hidden_states


def run_loop_trial(all_states, gauge_used, pca, corners, n_points, rng, occurrence=1):
    """Sample a loop, compute Pancharatnam phase. Returns phase or None."""
    hs = sample_loop(all_states, gauge_used, corners, n_points, rng, occurrence)
    if hs is None:
        return None
    complex_states = [to_complex(h, pca) for h in hs]
    return pancharatnam_phase(complex_states)


def run_shuffled_trial(all_states, gauge_used, pca, corners, n_points, rng, occurrence=1):
    """Sample a loop, SHUFFLE the order, compute phase. Returns phase or None."""
    hs = sample_loop(all_states, gauge_used, corners, n_points, rng, occurrence)
    if hs is None:
        return None
    rng.shuffle(hs)
    complex_states = [to_complex(h, pca) for h in hs]
    return pancharatnam_phase(complex_states)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    rng = np.random.default_rng(42)
    random.seed(42)
    torch.manual_seed(42)
    
    tok, mdl, device = load_model()
    
    # Verify prompts tokenize correctly
    print("\nVerifying tokenization...", flush=True)
    failures = 0
    for (al, bl), prompts in PROMPT_BANK.items():
        for p in prompts:
            positions, _ = find_concept_positions(tok, p, CONCEPT)
            if len(positions) != 2:
                print(f"  FAIL ({al},{bl}): {len(positions)} occurrences in: {p[:60]}...")
                failures += 1
    if failures > 0:
        print(f"\n{failures} prompts failed tokenization. Fix before running.")
        sys.exit(1)
    print("  All prompts OK.", flush=True)
    
    # Pre-compute all hidden states
    all_states = precompute_all_states(tok, mdl, device)
    
    # Fit PCA gauge on held-out set
    print("\nFitting PCA gauge on held-out calibration set...", flush=True)
    pca, gauge_used = fit_gauge(all_states, N_GAUGE_SAMPLES, rng)
    explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: {explained[0]:.3f}, {explained[1]:.3f} (total: {sum(explained):.3f})")
    
    results = {
        "concept": CONCEPT,
        "timestamp": TIMESTAMP,
        "model": "gpt2-124M",
        "k_loops": K_LOOPS,
        "n_loop_points": N_LOOP_POINTS,
        "n_gauge_samples": N_GAUGE_SAMPLES,
        "n_shuffles": N_SHUFFLES,
        "pca_variance_explained": [float(x) for x in explained],
        "prompts_per_cell": {f"{al},{bl}": len(v) for (al,bl), v in PROMPT_BANK.items()},
    }
    
    # ------------------------------------------------------------------
    # 1. Forward loops (CCW) — distribution of K phases
    # ------------------------------------------------------------------
    print(f"\n=== Forward loops (CCW) × {K_LOOPS} ===", flush=True)
    phases_ccw = []
    phases_ccw_first = []  # first-occurrence states for accumulation test
    for i in range(K_LOOPS):
        p2 = run_loop_trial(all_states, gauge_used, pca, CORNERS_CCW, N_LOOP_POINTS, rng, occurrence=1)
        p1 = run_loop_trial(all_states, gauge_used, pca, CORNERS_CCW, N_LOOP_POINTS, rng, occurrence=0)
        if p2 is not None:
            phases_ccw.append(p2)
        if p1 is not None:
            phases_ccw_first.append(p1)
    phases_ccw = np.array(phases_ccw)
    phases_ccw_first = np.array(phases_ccw_first)
    print(f"  {len(phases_ccw)} successful loops")
    print(f"  Φ_CCW: mean={np.mean(phases_ccw):.4f}, std={np.std(phases_ccw):.4f}, "
          f"median={np.median(phases_ccw):.4f}")
    print(f"  Φ_CCW (1st occ): mean={np.mean(phases_ccw_first):.4f}, std={np.std(phases_ccw_first):.4f}")
    
    # ------------------------------------------------------------------
    # 2. Reverse loops (CW) — orientation flip test
    # ------------------------------------------------------------------
    print(f"\n=== Reverse loops (CW) × {K_LOOPS} ===", flush=True)
    corners_cw = list(reversed(CORNERS_CCW))
    phases_cw = []
    for i in range(K_LOOPS):
        p = run_loop_trial(all_states, gauge_used, pca, corners_cw, N_LOOP_POINTS, rng)
        if p is not None:
            phases_cw.append(p)
    phases_cw = np.array(phases_cw)
    print(f"  {len(phases_cw)} successful loops")
    print(f"  Φ_CW: mean={np.mean(phases_cw):.4f}, std={np.std(phases_cw):.4f}")
    
    # Orientation test: mean(Φ_CCW) + mean(Φ_CW) ≈ 0
    orientation_sum = np.mean(phases_ccw) + np.mean(phases_cw)
    print(f"  mean(Φ_CCW) + mean(Φ_CW) = {orientation_sum:.4f}")
    
    # ------------------------------------------------------------------
    # 3. Tall loops (shape invariance)
    # ------------------------------------------------------------------
    print(f"\n=== Tall loops × {K_LOOPS} ===", flush=True)
    phases_tall = []
    for i in range(K_LOOPS):
        p = run_loop_trial(all_states, gauge_used, pca, CORNERS_TALL, N_LOOP_POINTS, rng)
        if p is not None:
            phases_tall.append(p)
    phases_tall = np.array(phases_tall)
    print(f"  {len(phases_tall)} successful loops")
    print(f"  Φ_tall: mean={np.mean(phases_tall):.4f}, std={np.std(phases_tall):.4f}")
    shape_delta = abs(np.mean(phases_ccw)) - abs(np.mean(phases_tall))
    print(f"  |mean(Φ_CCW)| - |mean(Φ_tall)| = {shape_delta:.4f}")
    
    # ------------------------------------------------------------------
    # 4. Schedule invariance: 4-point loops (one per corner)
    # ------------------------------------------------------------------
    print(f"\n=== Fast loops (4 points) × {K_LOOPS} ===", flush=True)
    phases_fast = []
    for i in range(K_LOOPS):
        p = run_loop_trial(all_states, gauge_used, pca, CORNERS_CCW, 4, rng)
        if p is not None:
            phases_fast.append(p)
    phases_fast = np.array(phases_fast)
    print(f"  {len(phases_fast)} successful loops")
    print(f"  Φ_fast: mean={np.mean(phases_fast):.4f}, std={np.std(phases_fast):.4f}")
    schedule_delta = np.mean(phases_ccw) - np.mean(phases_fast)
    print(f"  mean(Φ_CCW) - mean(Φ_fast) = {schedule_delta:.4f}")
    
    # ------------------------------------------------------------------
    # 5. Null distribution: shuffled orderings
    # ------------------------------------------------------------------
    print(f"\n=== Null distribution ({N_SHUFFLES} shuffled loops) ===", flush=True)
    phases_null = []
    for i in range(N_SHUFFLES):
        p = run_shuffled_trial(all_states, gauge_used, pca, CORNERS_CCW, N_LOOP_POINTS, rng)
        if p is not None:
            phases_null.append(p)
    phases_null = np.array(phases_null)
    print(f"  {len(phases_null)} successful shuffled loops")
    print(f"  Φ_null: mean={np.mean(phases_null):.4f}, std={np.std(phases_null):.4f}")
    
    # Two-sample test: are the CCW phases drawn from a different distribution than shuffled?
    U_stat, p_mann = mannwhitneyu(phases_ccw, phases_null, alternative='two-sided')
    t_stat, p_ttest = ttest_ind(phases_ccw, phases_null, equal_var=False)
    print(f"  Mann-Whitney U: U={U_stat:.1f}, p={p_mann:.4f}")
    print(f"  Welch t-test:   t={t_stat:.3f}, p={p_ttest:.4f}")
    
    # ------------------------------------------------------------------
    # 6. Accumulation test: does 2nd occurrence carry more phase than 1st?
    # ------------------------------------------------------------------
    print(f"\n=== Accumulation test ===", flush=True)
    mean_abs_2nd = np.mean(np.abs(phases_ccw))
    mean_abs_1st = np.mean(np.abs(phases_ccw_first))
    print(f"  mean |Φ_2nd| = {mean_abs_2nd:.4f}")
    print(f"  mean |Φ_1st| = {mean_abs_1st:.4f}")
    print(f"  difference   = {mean_abs_2nd - mean_abs_1st:.4f}")
    t_acc, p_acc = ttest_ind(np.abs(phases_ccw), np.abs(phases_ccw_first), equal_var=False)
    print(f"  t-test: t={t_acc:.3f}, p={p_acc:.4f}")
    
    # ------------------------------------------------------------------
    # 7. Store results
    # ------------------------------------------------------------------
    results.update({
        "phases_ccw_mean": float(np.mean(phases_ccw)),
        "phases_ccw_std": float(np.std(phases_ccw)),
        "phases_ccw_median": float(np.median(phases_ccw)),
        "phases_cw_mean": float(np.mean(phases_cw)),
        "phases_cw_std": float(np.std(phases_cw)),
        "phases_tall_mean": float(np.mean(phases_tall)),
        "phases_tall_std": float(np.std(phases_tall)),
        "phases_fast_mean": float(np.mean(phases_fast)),
        "phases_fast_std": float(np.std(phases_fast)),
        "phases_null_mean": float(np.mean(phases_null)),
        "phases_null_std": float(np.std(phases_null)),
        "orientation_sum": float(orientation_sum),
        "shape_delta": float(shape_delta),
        "schedule_delta": float(schedule_delta),
        "mann_whitney_U": float(U_stat),
        "mann_whitney_p": float(p_mann),
        "welch_t": float(t_stat),
        "welch_p": float(p_ttest),
        "accumulation_mean_abs_2nd": float(mean_abs_2nd),
        "accumulation_mean_abs_1st": float(mean_abs_1st),
        "accumulation_t": float(t_acc),
        "accumulation_p": float(p_acc),
    })
    
    # ------------------------------------------------------------------
    # 8. Verdict
    # ------------------------------------------------------------------
    ev_for, ev_against = [], []
    
    # Orientation flip: means should sum to ≈ 0
    # Use a tolerance scaled to the std
    orient_tol = 2 * max(np.std(phases_ccw), np.std(phases_cw)) / np.sqrt(K_LOOPS)
    if abs(orientation_sum) < max(orient_tol, 0.1):
        ev_for.append(f"orientation flip (sum={orientation_sum:.4f}, tol={orient_tol:.4f})")
    else:
        ev_against.append(f"orientation flip FAILS (sum={orientation_sum:.4f}, tol={orient_tol:.4f})")
    
    # Shape invariance: |mean CCW| ≈ |mean tall|
    if abs(shape_delta) < 0.15:
        ev_for.append(f"shape invariance (Δ={shape_delta:.4f})")
    else:
        ev_against.append(f"shape invariance FAILS (Δ={shape_delta:.4f})")
    
    # Schedule invariance: mean CCW ≈ mean fast
    if abs(schedule_delta) < 0.15:
        ev_for.append(f"schedule invariance (Δ={schedule_delta:.4f})")
    else:
        ev_against.append(f"schedule invariance FAILS (Δ={schedule_delta:.4f})")
    
    # Significance vs null
    if p_mann < 0.05:
        ev_for.append(f"significant vs null (Mann-Whitney p={p_mann:.4f})")
    else:
        ev_against.append(f"NOT significant vs null (Mann-Whitney p={p_mann:.4f})")
    
    # Non-zero phase: is the distribution centered away from 0?
    from scipy.stats import ttest_1samp
    t_zero, p_zero = ttest_1samp(phases_ccw, 0.0)
    if p_zero < 0.05:
        ev_for.append(f"phase ≠ 0 (t={t_zero:.3f}, p={p_zero:.4f})")
    else:
        ev_against.append(f"phase consistent with 0 (t={t_zero:.3f}, p={p_zero:.4f})")
    results["phase_vs_zero_t"] = float(t_zero)
    results["phase_vs_zero_p"] = float(p_zero)
    
    verdict = (
        "GEOMETRIC PHASE DETECTED" if len(ev_for) >= 4 and p_mann < 0.05
        else "GEOMETRIC PHASE CANDIDATE" if len(ev_for) >= 3
        else "INCONCLUSIVE" if len(ev_for) >= 2
        else "NULL RESULT"
    )
    
    results["geometric_evidence"] = ev_for
    results["geometric_against"] = ev_against
    results["verdict"] = verdict
    
    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")
    for e in ev_for:
        print(f"  ✓ {e}")
    for a in ev_against:
        print(f"  ✗ {a}")
    
    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    results["phases_ccw"] = phases_ccw.tolist()
    results["phases_cw"] = phases_cw.tolist()
    results["phases_tall"] = phases_tall.tolist()
    results["phases_fast"] = phases_fast.tolist()
    results["phases_null"] = phases_null.tolist()
    results["phases_ccw_first"] = phases_ccw_first.tolist()
    
    out_json = RESULT_DIR / f"polar_holonomy_v2_{TIMESTAMP}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}", flush=True)
    
    # ------------------------------------------------------------------
    # 10. Plot
    # ------------------------------------------------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Panel 1: CCW vs Null
        ax = axes[0]
        bins = np.linspace(min(phases_ccw.min(), phases_null.min()) - 0.1,
                          max(phases_ccw.max(), phases_null.max()) + 0.1, 40)
        ax.hist(phases_null, bins=bins, alpha=0.5, label=f"null (n={len(phases_null)})", color="gray")
        ax.hist(phases_ccw, bins=bins, alpha=0.6, label=f"CCW (n={len(phases_ccw)})", color="red")
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Pancharatnam phase (rad)")
        ax.set_ylabel("Count")
        ax.set_title(f"Ordered vs Shuffled\nMann-Whitney p={p_mann:.4f}")
        ax.legend(fontsize=8)
        
        # Panel 2: CCW vs CW (orientation flip)
        ax = axes[1]
        bins2 = np.linspace(min(phases_ccw.min(), phases_cw.min()) - 0.1,
                           max(phases_ccw.max(), phases_cw.max()) + 0.1, 40)
        ax.hist(phases_ccw, bins=bins2, alpha=0.6, label="CCW", color="red")
        ax.hist(phases_cw, bins=bins2, alpha=0.6, label="CW", color="blue")
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_xlabel("Pancharatnam phase (rad)")
        ax.set_title(f"Orientation flip\nΣ means = {orientation_sum:.4f}")
        ax.legend(fontsize=8)
        
        # Panel 3: All loop types
        ax = axes[2]
        data = [phases_ccw, phases_cw, phases_tall, phases_fast, phases_null]
        labels = ["CCW", "CW", "tall", "fast", "null"]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#95a5a6"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_ylabel("Pancharatnam phase (rad)")
        ax.set_title("All loop types")
        
        plt.suptitle(f"Polar Holonomy v2 — GPT-2 — '{CONCEPT}'\nVerdict: {verdict}", 
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        out_png = RESULT_DIR / f"polar_holonomy_v2_{TIMESTAMP}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Plot: {out_png}", flush=True)
    
    return results


if __name__ == "__main__":
    run_experiment()
