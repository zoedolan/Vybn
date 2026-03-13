#!/usr/bin/env python3
"""
polar_holonomy_native_v2.py — Holonomy in native R^768, FIXED reference frame

v1 had a sign bug: the initial tangent vector pointed toward points[1],
so it rotated with the loop direction, masking the orientation reversal.

v2 uses a canonical fixed tangent frame at p0 (project standard basis
vectors onto tangent plane), independent of loop direction.

Validated on S^2: correctly gives +π/2 for CCW and -π/2 for CW on
the standard octant triangle.
"""

import sys, json, numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp, levene
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL = True
except ImportError:
    HAS_MPL = False

CONCEPT = "threshold"
K_LOOPS = 200
N_LOOP_POINTS = 8
N_SHUFFLES = 200
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
# Parallel transport on S^{n-1} — CORRECTED
# ---------------------------------------------------------------------------
def parallel_transport(v, a, b):
    """
    Parallel transport tangent vector v at a to tangent plane at b,
    along the great circle on S^{n-1}.
    Formula: v' = v - ⟨a+b, v⟩/(1 + ⟨a,b⟩) · (a + b)
    """
    ab = np.dot(a, b)
    if ab <= -1.0 + 1e-12:
        return -v
    apb = a + b
    coeff = -np.dot(apb, v) / (1.0 + ab)
    vp = v + coeff * apb
    return vp

def holonomy_angle_fixed(points):
    """
    Compute holonomy with a CANONICAL FIXED reference frame at p0.
    
    The reference frame is built from standard basis vectors projected
    onto the tangent plane at p0, then Gram-Schmidt orthogonalized.
    This is INDEPENDENT of loop direction.
    
    Validated: gives +π/2 for CCW octant triangle, -π/2 for CW on S^2.
    """
    n_pts = len(points)
    if n_pts < 3:
        return 0.0
    
    p0 = points[0]
    n = len(p0)
    
    # Build a canonical 2D frame in T_{p0}(S^{n-1})
    # Project standard basis vectors, pick first two linearly independent ones
    frame = []
    for dim in range(n):
        e = np.zeros(n)
        e[dim] = 1.0
        t = e - np.dot(e, p0) * p0  # project to tangent plane
        # Gram-Schmidt against existing frame vectors
        for f in frame:
            t = t - np.dot(t, f) * f
        norm = np.linalg.norm(t)
        if norm > 1e-10:
            frame.append(t / norm)
            if len(frame) == 2:
                break
    
    if len(frame) < 2:
        return 0.0  # degenerate
    
    t1, t2 = frame[0], frame[1]
    
    # Transport t1 around the closed loop
    v = t1.copy()
    for k in range(n_pts):
        a = points[k]
        b = points[(k + 1) % n_pts]
        v = parallel_transport(v, a, b)
    
    # Measure rotation angle of v relative to fixed frame (t1, t2)
    c1 = np.dot(v, t1)
    c2 = np.dot(v, t2)
    angle = np.arctan2(c2, c1)
    return angle

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
            h = h / np.linalg.norm(h)
            hs.append(h)
    return hs

def run_trials(all_states, corners, n_points, k_trials, rng, occurrence=1, shuffle=False):
    angles = []
    for _ in range(k_trials):
        hs = sample_loop(all_states, corners, n_points, rng, occurrence)
        if hs is None:
            continue
        if shuffle:
            rng.shuffle(hs)
        angles.append(holonomy_angle_fixed(hs))
    return np.array(angles)

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
    
    # Diagnostic
    all_h = []
    for cell in all_states.values():
        for h1, h2 in cell:
            all_h.append(h2 / np.linalg.norm(h2))
    all_h = np.stack(all_h)
    
    # Pairwise cosines
    print(f"\n=== Diagnostic ===", flush=True)
    gram = all_h @ all_h.T
    mask = np.triu(np.ones_like(gram, dtype=bool), k=1)
    print(f"  All pairwise cosines: mean={gram[mask].mean():.6f}, std={gram[mask].std():.6f}")
    print(f"  Min={gram[mask].min():.6f}, Max={gram[mask].max():.6f}")
    
    # The key question: does the first canonical tangent vector even distinguish
    # the cell structure?
    p0 = all_h[0]
    e0 = np.zeros(768); e0[0] = 1.0
    t0 = e0 - np.dot(e0, p0) * p0; t0 = t0 / np.linalg.norm(t0)
    
    # Project all vectors onto t0
    projs = [np.dot(h - np.dot(h, p0)*p0, t0) for h in all_h]
    print(f"  Projection onto canonical tangent: mean={np.mean(projs):.6f}, std={np.std(projs):.6f}")
    
    results = {"concept": CONCEPT, "timestamp": TIMESTAMP, "model": "gpt2-124M",
               "method": "native R^768, fixed canonical frame (v2 — sign bug fixed)"}
    
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
        angles = run_trials(all_states, corners, n_pts, k, rng, occ, shuf)
        data[name] = angles
        print(f"  Holonomy angle: mean={np.mean(angles):+.6f}  std={np.std(angles):.6f}  "
              f"range=[{angles.min():.4f}, {angles.max():.4f}]")
    
    # ======================================================================
    ccw = data["CCW"]; cw = data["CW"]; null = data["null"]
    tall = data["tall"]; fast = data["fast"]; first = data["1st"]
    
    print(f"\n{'='*65}")
    print(f"  NATIVE R^768 HOLONOMY (v2 — FIXED REFERENCE FRAME)")
    print(f"{'='*65}")
    
    # 1. Orientation flip — THE CRITICAL TEST
    orient_sum = np.mean(ccw) + np.mean(cw)
    # Also check: does the ratio approach -1?
    if abs(np.mean(cw)) > 1e-10:
        orient_ratio = np.mean(ccw) / np.mean(cw)
    else:
        orient_ratio = float('inf')
    print(f"\n1. Orientation flip (the test that failed in v1):")
    print(f"   mean(CCW) = {np.mean(ccw):+.6f} rad ({np.degrees(np.mean(ccw)):+.4f}°)")
    print(f"   mean(CW)  = {np.mean(cw):+.6f} rad ({np.degrees(np.mean(cw)):+.4f}°)")
    print(f"   sum        = {orient_sum:+.6f} rad")
    print(f"   ratio       = {orient_ratio:.4f} (should be -1 for perfect reversal)")
    
    # 2. Shape invariance
    shape_d = np.mean(ccw) - np.mean(tall)
    t_sh, p_sh = ttest_ind(ccw, tall, equal_var=False)
    print(f"\n2. Shape invariance:")
    print(f"   mean(CCW)  = {np.mean(ccw):+.6f}")
    print(f"   mean(tall) = {np.mean(tall):+.6f}")
    print(f"   diff = {shape_d:+.6f}, t={t_sh:.3f}, p={p_sh:.4e}")
    
    # 3. Schedule invariance  
    sched_d = np.mean(ccw) - np.mean(fast)
    t_sc, p_sc = ttest_ind(ccw, fast, equal_var=False)
    print(f"\n3. Schedule invariance:")
    print(f"   mean(CCW)  = {np.mean(ccw):+.6f}")
    print(f"   mean(fast) = {np.mean(fast):+.6f}")
    print(f"   diff = {sched_d:+.6f}, t={t_sc:.3f}, p={p_sc:.4e}")
    
    # 4. vs null
    U, p_mw = mannwhitneyu(ccw, null, alternative='two-sided')
    t_w, p_w = ttest_ind(ccw, null, equal_var=False)
    pooled = np.sqrt((np.std(ccw)**2 + np.std(null)**2) / 2)
    d_eff = (np.mean(ccw) - np.mean(null)) / pooled if pooled > 0 else 0
    print(f"\n4. CCW vs null (shuffled):")
    print(f"   mean(CCW)  = {np.mean(ccw):+.6f}")
    print(f"   mean(null) = {np.mean(null):+.6f}")
    print(f"   MW: U={U:.0f}, p={p_mw:.4e}")
    print(f"   Welch t: t={t_w:.3f}, p={p_w:.4e}")
    print(f"   Cohen's d: {d_eff:.3f}")
    
    # 5. Phase != 0
    t_z, p_z = ttest_1samp(ccw, 0.0)
    print(f"\n5. Phase ≠ 0:")
    print(f"   t={t_z:.3f}, p={p_z:.4e}")
    print(f"   mean = {np.mean(ccw):+.6f} rad ({np.degrees(np.mean(ccw)):+.4f}°)")
    
    # 6. Variance
    lev, p_lev = levene(ccw, null)
    print(f"\n6. Variance comparison:")
    print(f"   std(CCW)={np.std(ccw):.6f}, std(null)={np.std(null):.6f}")
    print(f"   Levene: F={lev:.3f}, p={p_lev:.4e}")
    
    # ======================================================================
    # VERDICT
    # ======================================================================
    ev_for, ev_against = [], []
    
    # Orientation: sum should be near zero relative to the mean magnitudes
    orient_tol = 0.3 * (abs(np.mean(ccw)) + abs(np.mean(cw)))  # generous: within 30%
    if orient_tol < 1e-10:
        orient_tol = 0.001
    if abs(orient_sum) < orient_tol and np.sign(np.mean(ccw)) != np.sign(np.mean(cw)):
        ev_for.append(f"orientation reversal (sum={orient_sum:+.6f}, tol={orient_tol:.6f})")
    else:
        ev_against.append(f"orientation reversal FAILS (sum={orient_sum:+.6f}, ratio={orient_ratio:.4f})")
    
    if p_sh > 0.05:
        ev_for.append(f"shape invariance (p={p_sh:.4f})")
    else:
        ev_against.append(f"shape NOT invariant (p={p_sh:.4f})")
    
    if p_sc > 0.05:
        ev_for.append(f"schedule invariance (p={p_sc:.4f})")
    else:
        ev_against.append(f"schedule NOT invariant (p={p_sc:.4f})")
    
    if p_mw < 0.05:
        ev_for.append(f"significant vs null (p={p_mw:.4e}, d={d_eff:.3f})")
    else:
        ev_against.append(f"NOT significant vs null (p={p_mw:.4e})")
    
    if p_z < 0.05:
        ev_for.append(f"phase ≠ 0 (mean={np.mean(ccw):+.6f}, p={p_z:.4e})")
    else:
        ev_against.append(f"phase ≈ 0 (p={p_z:.4e})")
    
    n_pass = sum(1 for e in ev_for if "orientation" in e.lower())
    has_orient = any("orientation reversal" in e and "FAILS" not in e for e in ev_for)
    
    if has_orient and p_mw < 0.01 and len(ev_for) >= 4:
        verdict = "NATIVE HOLONOMY CONFIRMED"
    elif has_orient and p_mw < 0.05:
        verdict = "NATIVE HOLONOMY CANDIDATE"
    elif p_mw < 0.05 and len(ev_for) >= 3:
        verdict = "SIGNIFICANT BUT NOT HOLONOMY (no orientation reversal)"
    elif p_z < 0.05 and len(ev_for) >= 2:
        verdict = "WEAK SIGNAL, INCONCLUSIVE"
    else:
        verdict = "NULL — no native holonomy"
    
    print(f"\n{'='*65}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*65}")
    for e in ev_for: print(f"  ✓ {e}")
    for a in ev_against: print(f"  ✗ {a}")
    
    # Save
    results.update({
        "ccw_mean": float(np.mean(ccw)), "ccw_std": float(np.std(ccw)),
        "cw_mean": float(np.mean(cw)), "cw_std": float(np.std(cw)),
        "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
        "tall_mean": float(np.mean(tall)), "fast_mean": float(np.mean(fast)),
        "orientation_sum": float(orient_sum), "orientation_ratio": float(orient_ratio),
        "shape_p": float(p_sh), "schedule_p": float(p_sc),
        "mw_p": float(p_mw), "cohens_d": float(d_eff),
        "phase_vs_zero_p": float(p_z),
        "verdict": verdict, "evidence_for": ev_for, "evidence_against": ev_against,
        "angles_ccw": ccw.tolist(), "angles_cw": cw.tolist(), "angles_null": null.tolist(),
        "angles_tall": tall.tolist(), "angles_fast": fast.tolist(),
    })
    
    out_json = RESULT_DIR / f"polar_holonomy_native_v2_{TIMESTAMP}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}")
    
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        ax = axes[0]
        lo = min(ccw.min(), null.min(), cw.min()) - 0.001
        hi = max(ccw.max(), null.max(), cw.max()) + 0.001
        bins = np.linspace(lo, hi, 40)
        ax.hist(null, bins=bins, alpha=0.4, color="gray", label=f"null (μ={np.mean(null):+.5f})")
        ax.hist(ccw, bins=bins, alpha=0.6, color="red", label=f"CCW (μ={np.mean(ccw):+.5f})")
        ax.hist(cw, bins=bins, alpha=0.6, color="blue", label=f"CW (μ={np.mean(cw):+.5f})")
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("Holonomy angle (rad)"); ax.set_ylabel("Count")
        ax.set_title(f"CCW vs CW vs null\nMW p={p_mw:.2e}")
        ax.legend(fontsize=7)
        
        ax = axes[1]
        bp = ax.boxplot([ccw, cw, tall, fast, null],
                       labels=["CCW","CW","tall","fast","null"], patch_artist=True)
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#95a5a6"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.axhline(0, color="k", ls=":", alpha=0.3)
        ax.set_ylabel("Holonomy angle (rad)")
        ax.set_title("All Conditions")
        
        ax = axes[2]
        ax.scatter(ccw, np.arange(len(ccw)), alpha=0.3, s=5, c="red", label="CCW")
        ax.scatter(cw, np.arange(len(cw)), alpha=0.3, s=5, c="blue", label="CW")
        ax.axvline(0, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("Holonomy angle (rad)"); ax.set_ylabel("Trial")
        ax.set_title("CCW vs CW (individual trials)")
        ax.legend(fontsize=8)
        
        plt.suptitle(f"Native R^768 Holonomy v2 (fixed frame) — '{CONCEPT}'\n{verdict}", fontweight="bold")
        plt.tight_layout()
        out_png = RESULT_DIR / f"polar_holonomy_native_v2_{TIMESTAMP}.png"
        plt.savefig(out_png, dpi=150); plt.close()
        print(f"Plot: {out_png}")
    
    return results

if __name__ == "__main__":
    run_experiment()
