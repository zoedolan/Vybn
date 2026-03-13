#!/usr/bin/env python3
"""
polar_holonomy_native.py — Holonomy in native R^768, no PCA

Zoe's challenge: if the geometric phase is real, it should be visible in the
native 768-dimensional space without any dimensionality reduction.

Three observables, all computed directly on unit-normalized hidden states
in R^768 (i.e., points on S^767):

1. REAL PANCHARATNAM PHASE (R^768)
   Φ = arg(∏_k ⟨h_k, h_{k+1}⟩)
   where ⟨,⟩ is the standard real inner product and h_k are unit-normalized.
   For real vectors, all inner products are real, so Φ ∈ {0, π} depending on
   sign flips. This is the real-space analog — limited but honest.

2. GEODESIC DEFICIT ANGLE
   On S^{n-1}, the geodesic distance between unit vectors is arccos(⟨h_k, h_{k+1}⟩).
   For a closed loop, we compare:
     - sum of geodesic distances (total path length)
     - direct geodesic from h_0 back to h_0 (= 0 for exact closure)
   The deficit = total turning - expected turning for a flat polygon.
   More precisely: for a geodesic polygon on S^{n-1}, the holonomy is the
   rotation accumulated by parallel-transporting a tangent vector around
   the loop. We compute this directly.

3. PARALLEL TRANSPORT HOLONOMY (the definitive test)
   Given a sequence of points h_0, h_1, ..., h_{N-1} on S^767:
   - Start with a tangent vector v_0 at h_0 (orthogonal to h_0)
   - Parallel transport v along each geodesic segment h_k → h_{k+1}
   - After returning to h_0, the transported vector v_final differs from v_0
     by a rotation in the tangent plane at h_0
   - The angle of this rotation IS the holonomy
   
   Parallel transport on S^{n-1} along the geodesic from a to b:
     v_transported = v - (v·a_perp) * a_perp - (v·b_perp) * b_perp
                     + (v·a_perp) * (cos(θ) * a_perp + sin(θ) * b_perp)
                     + ... 
   
   More precisely, the Schild's ladder / exact formula for parallel transport
   of v along great circle from a to b on S^{n-1}:
     v' = v - [⟨v, a+b⟩ / (1+⟨a,b⟩)] * (a + b) + 2⟨v, a⟩ * b
   (when a ≠ -b, i.e., not antipodal)
   
   Wait — let me use the standard formula. For unit vectors a, b on S^{n-1},
   parallel transport of tangent vector v at a to tangent plane at b along
   the great circle:
   
   Let d = b - ⟨a,b⟩ a (the tangent direction at a pointing toward b)
   Let d_hat = d / |d| (if |d| > 0)
   θ = arccos(⟨a,b⟩)
   
   v' = v - (⟨v,a⟩ + ⟨v,d_hat⟩·cos(θ))·a·??? 
   
   Actually, the clean formula (from differential geometry):
   Parallel transport of v ∈ T_a(S^{n-1}) to T_b(S^{n-1}) along great circle:
   
   v' = v + ⟨-a - b, v⟩/(1 + ⟨a,b⟩) · (a + b)
   
   [This is the "discrete connection" / Levi-Civita transport on S^{n-1}]
   Requires: v ⊥ a, and result v' ⊥ b. Valid when ⟨a,b⟩ > -1.
"""

import sys
import json
import cmath
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
CONCEPT = "threshold"
K_LOOPS = 200
N_LOOP_POINTS = 8
N_SHUFFLES = 200
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# Same prompt bank as v3
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

# ---------------------------------------------------------------------------
# Model & extraction (same as v3)
# ---------------------------------------------------------------------------
def load_model():
    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True); mdl.eval()
    return tok, mdl

def find_concept_positions(tok, prompt, concept):
    input_ids = tok.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tok.decode([tid]).lower()
        if concept.lower() in decoded and len(decoded.strip()) <= len(concept) + 2:
            positions.append(i)
    return positions

def extract_both(tok, mdl, prompt, concept):
    positions = find_concept_positions(tok, prompt, concept)
    if len(positions) != 2: return None
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    L = out.hidden_states[-1][0]
    return L[positions[0]].numpy().astype(np.float64), L[positions[1]].numpy().astype(np.float64)

def precompute_all_states(tok, mdl):
    print("Pre-computing hidden states...", flush=True)
    states = {}
    for (al, bl), prompts in PROMPT_BANK.items():
        cell = []
        for prompt in prompts:
            r = extract_both(tok, mdl, prompt, CONCEPT)
            if r is not None: cell.append(r)
        states[(al, bl)] = cell
        print(f"  ({al},{bl}): {len(cell)}/{len(prompts)} OK", flush=True)
    return states

# ---------------------------------------------------------------------------
# Parallel transport on S^{n-1}
# ---------------------------------------------------------------------------
def parallel_transport(v, a, b):
    """
    Parallel transport tangent vector v at point a to tangent plane at point b,
    along the great circle on S^{n-1}.
    
    a, b: unit vectors in R^n
    v: tangent vector at a (i.e., v·a = 0)
    Returns: v' tangent at b (i.e., v'·b = 0), same length as v
    
    Formula: v' = v + ⟨-(a+b), v⟩/(1 + ⟨a,b⟩) · (a + b)
    Valid when ⟨a,b⟩ > -1 (not antipodal).
    """
    ab = np.dot(a, b)
    if ab <= -1.0 + 1e-12:
        return -v  # antipodal: reverse
    apb = a + b
    coeff = -np.dot(apb, v) / (1.0 + ab)
    vp = v + coeff * apb
    return vp

def holonomy_angle(points):
    """
    Compute the holonomy angle by parallel-transporting a tangent vector
    around a closed geodesic polygon on S^{n-1}.
    
    points: list of unit vectors (the polygon vertices, in order, loop is closed)
    
    Returns: angle in radians (the rotation of the transported vector relative
    to its initial orientation in the tangent plane at points[0])
    """
    n_pts = len(points)
    if n_pts < 3:
        return 0.0
    
    p0 = points[0]
    
    # Pick an initial tangent vector at p0 (orthogonal to p0)
    # Choose the component of a random vector orthogonal to p0
    # Use a deterministic choice: the direction toward p1, projected to tangent plane
    d = points[1] - np.dot(points[1], p0) * p0
    if np.linalg.norm(d) < 1e-12:
        # p0 and p1 nearly identical, try p2
        d = points[2] - np.dot(points[2], p0) * p0
    d = d / np.linalg.norm(d)
    
    # Also need a second tangent vector to measure rotation angle
    # Pick one orthogonal to both p0 and d
    # Use Gram-Schmidt with p2's tangent projection
    d2 = points[-1] - np.dot(points[-1], p0) * p0
    d2 = d2 - np.dot(d2, d) * d  # orthogonalize against d
    if np.linalg.norm(d2) < 1e-12:
        d2 = points[2] - np.dot(points[2], p0) * p0
        d2 = d2 - np.dot(d2, d) * d
    if np.linalg.norm(d2) < 1e-12:
        return 0.0  # degenerate
    d2 = d2 / np.linalg.norm(d2)
    
    # Now d, d2 form an orthonormal basis of a 2D subspace of T_{p0}(S^{n-1})
    # Transport d around the loop
    v = d.copy()
    for k in range(n_pts):
        a = points[k]
        b = points[(k + 1) % n_pts]
        v = parallel_transport(v, a, b)
    
    # v is now back in T_{p0}. Measure the rotation angle
    # v_final should have the same length as d (parallel transport preserves length)
    # Project v onto the (d, d2) basis
    c1 = np.dot(v, d)
    c2 = np.dot(v, d2)
    angle = np.arctan2(c2, c1)
    return angle


def product_of_cosines(points):
    """
    Product of ⟨h_k, h_{k+1}⟩ around the closed loop.
    For unit vectors, each term is cos(geodesic distance).
    The sign of the product indicates whether there's a sign holonomy.
    """
    prod = 1.0
    n = len(points)
    for k in range(n):
        dot = np.dot(points[k], points[(k + 1) % n])
        prod *= dot
    return prod

def total_geodesic_length(points):
    """Sum of arccos(⟨h_k, h_{k+1}⟩) around the loop."""
    total = 0.0
    n = len(points)
    for k in range(n):
        dot = np.clip(np.dot(points[k], points[(k + 1) % n]), -1, 1)
        total += np.arccos(dot)
    return total

# ---------------------------------------------------------------------------
# Loop construction
# ---------------------------------------------------------------------------
CORNERS_CCW = [("low","low"), ("high","low"), ("high","high"), ("low","high")]
CORNERS_TALL = [("low","low"), ("low","high"), ("high","high"), ("high","low")]

def sample_loop(all_states, corners, n_points, rng, occurrence=1):
    n_corners = len(corners)
    per_corner = [n_points // n_corners] * n_corners
    for i in range(n_points % n_corners):
        per_corner[i] += 1
    hs = []
    for ci, (al, bl) in enumerate(corners):
        cell = all_states[(al, bl)]
        if len(cell) < per_corner[ci]:
            return None
        chosen = rng.choice(len(cell), size=per_corner[ci], replace=False)
        for idx in chosen:
            h = cell[idx][occurrence].copy()
            h = h / np.linalg.norm(h)  # unit normalize
            hs.append(h)
    return hs

def run_trials(all_states, corners, n_points, k_trials, rng, occurrence=1, shuffle=False):
    """Run k_trials, return arrays of (holonomy_angle, product_of_cosines, geodesic_length)."""
    angles, prods, lengths = [], [], []
    for _ in range(k_trials):
        hs = sample_loop(all_states, corners, n_points, rng, occurrence)
        if hs is None:
            continue
        if shuffle:
            rng.shuffle(hs)
        angles.append(holonomy_angle(hs))
        prods.append(product_of_cosines(hs))
        lengths.append(total_geodesic_length(hs))
    return np.array(angles), np.array(prods), np.array(lengths)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    
    tok, mdl = load_model()
    
    print("\nVerifying tokenization...", flush=True)
    for (al, bl), prompts in PROMPT_BANK.items():
        for p in prompts:
            assert len(find_concept_positions(tok, p, CONCEPT)) == 2
    print("  All OK.", flush=True)
    
    all_states = precompute_all_states(tok, mdl)
    
    # Quick diagnostic: how spread out are the states?
    print("\n=== Diagnostic: pairwise cosine similarities ===", flush=True)
    all_h = []
    all_labels = []
    for (al, bl), cell in all_states.items():
        for h1, h2 in cell:
            h2n = h2 / np.linalg.norm(h2)
            all_h.append(h2n)
            all_labels.append(f"{al},{bl}")
    all_h = np.stack(all_h)
    
    # Within-cell vs between-cell similarities
    within, between = [], []
    for i in range(len(all_h)):
        for j in range(i+1, len(all_h)):
            cos = np.dot(all_h[i], all_h[j])
            if all_labels[i] == all_labels[j]:
                within.append(cos)
            else:
                between.append(cos)
    within, between = np.array(within), np.array(between)
    print(f"  Within-cell cosine:  mean={within.mean():.6f}, std={within.std():.6f}")
    print(f"  Between-cell cosine: mean={between.mean():.6f}, std={between.std():.6f}")
    print(f"  Difference:          {within.mean()-between.mean():.6f}")
    t_cells, p_cells = ttest_ind(within, between, equal_var=False)
    print(f"  t-test (within vs between): t={t_cells:.3f}, p={p_cells:.4f}")
    
    results = {
        "concept": CONCEPT, "timestamp": TIMESTAMP, "model": "gpt2-124M",
        "method": "native R^768, no PCA",
        "k_loops": K_LOOPS, "n_loop_points": N_LOOP_POINTS,
        "within_cell_cosine_mean": float(within.mean()),
        "between_cell_cosine_mean": float(between.mean()),
    }
    
    # ======================================================================
    # Run all conditions
    # ======================================================================
    conditions = {
        "CCW":  (CORNERS_CCW, N_LOOP_POINTS, False, 1),
        "CW":   (list(reversed(CORNERS_CCW)), N_LOOP_POINTS, False, 1),
        "tall": (CORNERS_TALL, N_LOOP_POINTS, False, 1),
        "fast": (CORNERS_CCW, 4, False, 1),
        "null": (CORNERS_CCW, N_LOOP_POINTS, True, 1),
        "1st":  (CORNERS_CCW, N_LOOP_POINTS, False, 0),
    }
    
    data = {}
    for name, (corners, n_pts, shuf, occ) in conditions.items():
        k = N_SHUFFLES if name == "null" else K_LOOPS
        print(f"\n=== {name} ({k} trials) ===", flush=True)
        angles, prods, lengths = run_trials(all_states, corners, n_pts, k, rng, occ, shuf)
        data[name] = {"angles": angles, "prods": prods, "lengths": lengths}
        print(f"  Holonomy angle:    mean={np.mean(angles):+.6f}  std={np.std(angles):.6f}  "
              f"range=[{angles.min():.4f}, {angles.max():.4f}]")
        print(f"  Product of cosines: mean={np.mean(prods):.6f}  std={np.std(prods):.6f}")
        print(f"  Geodesic length:    mean={np.mean(lengths):.4f}  std={np.std(lengths):.4f}")
    
    # ======================================================================
    # Statistical tests
    # ======================================================================
    ccw_a = data["CCW"]["angles"]
    cw_a = data["CW"]["angles"]
    null_a = data["null"]["angles"]
    tall_a = data["tall"]["angles"]
    fast_a = data["fast"]["angles"]
    first_a = data["1st"]["angles"]
    
    print(f"\n{'='*60}")
    print(f"  NATIVE R^768 HOLONOMY — STATISTICAL TESTS")
    print(f"{'='*60}")
    
    # 1. Orientation flip
    orient_sum = np.mean(ccw_a) + np.mean(cw_a)
    print(f"\n1. Orientation flip:")
    print(f"   mean(CCW) = {np.mean(ccw_a):+.6f}")
    print(f"   mean(CW)  = {np.mean(cw_a):+.6f}")
    print(f"   sum        = {orient_sum:+.6f}")
    
    # 2. Shape invariance
    shape_d = abs(np.mean(ccw_a)) - abs(np.mean(tall_a))
    print(f"\n2. Shape invariance:")
    print(f"   |mean(CCW)| - |mean(tall)| = {shape_d:+.6f}")
    
    # 3. Schedule invariance
    sched_d = np.mean(ccw_a) - np.mean(fast_a)
    print(f"\n3. Schedule invariance:")
    print(f"   mean(CCW) - mean(fast) = {sched_d:+.6f}")
    
    # 4. vs null
    U, p_mw = mannwhitneyu(ccw_a, null_a, alternative='two-sided')
    t_w, p_w = ttest_ind(ccw_a, null_a, equal_var=False)
    # Effect size
    pooled = np.sqrt((np.std(ccw_a)**2 + np.std(null_a)**2) / 2)
    d_eff = (np.mean(ccw_a) - np.mean(null_a)) / pooled if pooled > 0 else 0
    print(f"\n4. CCW vs null (shuffled):")
    print(f"   Mann-Whitney: U={U:.0f}, p={p_mw:.4e}")
    print(f"   Welch t:      t={t_w:.3f}, p={p_w:.4e}")
    print(f"   Cohen's d:    {d_eff:.3f}")
    
    # 5. Phase != 0
    t_z, p_z = ttest_1samp(ccw_a, 0.0)
    print(f"\n5. Phase ≠ 0:")
    print(f"   t={t_z:.3f}, p={p_z:.4e}")
    print(f"   mean = {np.mean(ccw_a):+.6f} rad ({np.degrees(np.mean(ccw_a)):+.4f}°)")
    
    # 6. Accumulation
    t_acc, p_acc = ttest_ind(np.abs(ccw_a), np.abs(first_a), equal_var=False)
    print(f"\n6. Accumulation (|2nd| vs |1st|):")
    print(f"   mean|Φ_2nd| = {np.mean(np.abs(ccw_a)):.6f}")
    print(f"   mean|Φ_1st| = {np.mean(np.abs(first_a)):.6f}")
    print(f"   t={t_acc:.3f}, p={p_acc:.4e}")
    
    # Also compare variances
    print(f"\n7. Variance comparison:")
    print(f"   std(CCW)  = {np.std(ccw_a):.6f}")
    print(f"   std(null) = {np.std(null_a):.6f}")
    from scipy.stats import levene
    lev_stat, p_lev = levene(ccw_a, null_a)
    print(f"   Levene's test: F={lev_stat:.3f}, p={p_lev:.4e}")
    
    # Geodesic length comparison
    print(f"\n8. Geodesic path length:")
    print(f"   CCW:  {np.mean(data['CCW']['lengths']):.4f} ± {np.std(data['CCW']['lengths']):.4f}")
    print(f"   null: {np.mean(data['null']['lengths']):.4f} ± {np.std(data['null']['lengths']):.4f}")
    t_len, p_len = ttest_ind(data['CCW']['lengths'], data['null']['lengths'], equal_var=False)
    print(f"   t={t_len:.3f}, p={p_len:.4e}")
    
    # ======================================================================
    # Verdict
    # ======================================================================
    ev_for, ev_against = [], []
    tol = max(2*max(np.std(ccw_a), np.std(cw_a))/np.sqrt(K_LOOPS), 0.001)
    
    if abs(orient_sum) < tol:
        ev_for.append(f"orientation flip (sum={orient_sum:+.6f})")
    else:
        ev_against.append(f"orientation flip FAILS (sum={orient_sum:+.6f}, tol={tol:.6f})")
    
    if abs(shape_d) < 0.01:
        ev_for.append(f"shape invariance (Δ={shape_d:+.6f})")
    else:
        ev_against.append(f"shape invariance FAILS (Δ={shape_d:+.6f})")
    
    if abs(sched_d) < 0.01:
        ev_for.append(f"schedule invariance (Δ={sched_d:+.6f})")
    else:
        ev_against.append(f"schedule invariance FAILS (Δ={sched_d:+.6f})")
    
    if p_mw < 0.05:
        ev_for.append(f"significant vs null (p={p_mw:.4e})")
    else:
        ev_against.append(f"NOT significant vs null (p={p_mw:.4e})")
    
    if p_z < 0.05 and np.std(ccw_a) > 0.0001:
        ev_for.append(f"phase ≠ 0 (mean={np.mean(ccw_a):+.6f}, p={p_z:.4e})")
    else:
        ev_against.append(f"phase ≈ 0 (mean={np.mean(ccw_a):+.6f}, p={p_z:.4e})")
    
    verdict = (
        "NATIVE HOLONOMY DETECTED" if len(ev_for) >= 4 and p_mw < 0.05
        else "NATIVE HOLONOMY CANDIDATE" if len(ev_for) >= 3
        else "INCONCLUSIVE" if len(ev_for) >= 2
        else "NULL RESULT — PCA effect is likely artifactual"
    )
    
    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")
    for e in ev_for: print(f"  ✓ {e}")
    for a in ev_against: print(f"  ✗ {a}")
    
    # Save
    results.update({
        "ccw_angle_mean": float(np.mean(ccw_a)),
        "ccw_angle_std": float(np.std(ccw_a)),
        "cw_angle_mean": float(np.mean(cw_a)),
        "null_angle_mean": float(np.mean(null_a)),
        "null_angle_std": float(np.std(null_a)),
        "orientation_sum": float(orient_sum),
        "shape_delta": float(shape_d),
        "schedule_delta": float(sched_d),
        "mann_whitney_p": float(p_mw),
        "phase_vs_zero_p": float(p_z),
        "cohens_d": float(d_eff),
        "verdict": verdict,
        "evidence_for": ev_for,
        "evidence_against": ev_against,
        "angles_ccw": ccw_a.tolist(),
        "angles_cw": cw_a.tolist(),
        "angles_null": null_a.tolist(),
        "angles_tall": tall_a.tolist(),
        "angles_fast": fast_a.tolist(),
    })
    
    out_json = RESULT_DIR / f"polar_holonomy_native_{TIMESTAMP}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}", flush=True)
    
    # Plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # Panel 1: CCW vs null holonomy angles
        ax = axes[0]
        lo = min(ccw_a.min(), null_a.min()) - 0.001
        hi = max(ccw_a.max(), null_a.max()) + 0.001
        bins = np.linspace(lo, hi, 40)
        ax.hist(null_a, bins=bins, alpha=0.5, color="gray", label=f"null (n={len(null_a)})")
        ax.hist(ccw_a, bins=bins, alpha=0.6, color="red", label=f"CCW (n={len(ccw_a)})")
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("Holonomy angle (rad)")
        ax.set_ylabel("Count")
        ax.set_title(f"Parallel Transport Holonomy\nMW p={p_mw:.2e}, d={d_eff:.3f}")
        ax.legend(fontsize=8)
        
        # Panel 2: orientation flip
        ax = axes[1]
        lo2 = min(ccw_a.min(), cw_a.min()) - 0.001
        hi2 = max(ccw_a.max(), cw_a.max()) + 0.001
        bins2 = np.linspace(lo2, hi2, 40)
        ax.hist(ccw_a, bins=bins2, alpha=0.6, color="red", label="CCW")
        ax.hist(cw_a, bins=bins2, alpha=0.6, color="blue", label="CW")
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("Holonomy angle (rad)")
        ax.set_title(f"Orientation Flip\nΣ means = {orient_sum:+.6f}")
        ax.legend(fontsize=8)
        
        # Panel 3: all conditions boxplot
        ax = axes[2]
        bp = ax.boxplot([ccw_a, cw_a, tall_a, fast_a, null_a],
                       labels=["CCW","CW","tall","fast","null"], patch_artist=True)
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#95a5a6"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        ax.set_ylabel("Holonomy angle (rad)")
        ax.set_title("All Conditions")
        
        plt.suptitle(f"Native R^768 Holonomy — GPT-2 — '{CONCEPT}'\n{verdict}", fontweight="bold")
        plt.tight_layout()
        out_png = RESULT_DIR / f"polar_holonomy_native_{TIMESTAMP}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Plot: {out_png}", flush=True)
    
    return results

if __name__ == "__main__":
    run_experiment()
