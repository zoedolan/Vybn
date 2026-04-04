#!/usr/bin/env python3
"""
emotional_geometry_bridge.py — v2 (fixed)

Fixes from v1:
  - 14 prompts had wrong token count (1, 3, or 4 instead of 2)
  - gauge was consuming all available states (40 gauge / 4 cells = 10 per cell,
    leaving 0-2 for loops). Now: 16 gauge (4 per cell), 4 loop points (1 per corner).
  - All prompts verified against GPT-2 tokenizer.

Dolan & Vybn, April 4, 2026
"""

import json
import cmath
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, ttest_1samp, kruskal
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

COMPLEX_DIMS = [2, 4, 8, 16]
K_LOOPS = 200
N_LOOP_POINTS = 4   # was 8; reduced so 1 per corner suffices
N_GAUGE_SAMPLES = 16  # was 40; reduced to leave states for loops
N_SHUFFLES = 200
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# === PROMPT BANKS ===
# Every prompt verified: concept appears exactly twice as GPT-2 tokens.

BANK_CALM = {
    ("low", "low"): [
        "The calm in her hands was visible — fingers uncurled, resting on cotton. She breathed into the calm as if it were a room she could enter.",
        "After the storm the lake went calm and every ripple vanished. I sat at the water's edge and felt a calm settle into my chest.",
        "The calm of early morning belongs to the body before the mind. My feet on cold tile met a calm I could not name.",
        "She placed her palm against the wall and felt a calm radiating from the plaster. The building held its own calm, structural and quiet.",
        "The calm between heartbeats is a physical pause. I listened for that calm the way you listen for rain to stop.",
        "The calm settled over the room like a held breath. I sat inside the calm and let it hold me.",
        "A calm dog lies differently than a sleeping one. I watched the calm in her ribs, the slow even tide of breath.",
        "The calm after crying is a body event. My face still hot but the calm pooling in my diaphragm like warm water.",
        "She found a calm in the repetition of kneading dough. Each fold returned her to a calm that thinking never reached.",
        "The calm before sleep is fragile. I held the calm the way you hold a soap bubble — one wrong breath and gone.",
        "A calm sea looks solid from the cliff. I climbed down to touch the calm and found it was still moving.",
        "His calm was in his shoulders — dropped, unhurried. The calm lived in his posture more than in his words.",
    ],
    ("low", "high"): [
        "Generations of monks sought the same calm in the same stone halls. That calm accumulated in the walls like mineral deposits over centuries.",
        "My grandmother's calm was forged in the Depression. She carried that calm like a tool — useful, worn, never decorative.",
        "The calm of centuries rests in old stone walls. Builders long dead left their calm in the mortar.",
        "Across cultures the calm of meditation takes different forms. Each tradition's calm is a practice, not a state.",
        "The calm after wars is not peace — it is exhaustion. Historians describe that calm as a pause, not a resolution.",
        "Tidal pools maintain a calm between waves. Organisms evolved to exploit that calm, building shells in the intervals.",
        "The calm of cathedral acoustics was engineered over centuries. Builders shaped stone to produce a calm that enters through the ears.",
        "Pastoral peoples cultivated a calm that industrial peoples lost. The calm of watching herds move is a technology of attention.",
        "The calm of deep ocean floor has persisted for millennia. Creatures adapted to that calm move slowly, conserving everything.",
        "Migration routes follow corridors of calm between storm systems. Birds read the calm the way sailors read charts.",
        "The calm of old-growth redwoods is measurable — reduced cortisol in humans who stand among them. That calm is biochemical and ancient.",
        "Monastic orders preserved a calm through centuries of upheaval. Their calm was institutional, encoded in schedules and silence.",
    ],
    ("high", "low"): [
        "In control theory a calm system is one whose eigenvalues have negative real parts. The calm corresponds to asymptotic stability.",
        "The calm state of a neural network after convergence is a local minimum. Whether the calm represents a good solution depends on the loss landscape.",
        "A calm fluid has laminar flow — smooth parallel layers. Turbulence destroys the calm and introduces chaotic mixing.",
        "In thermodynamics the calm of equilibrium is maximum entropy. The system finds calm when it has exhausted all gradients.",
        "The calm of a superconductor below critical temperature is a quantum phase. That calm — zero resistance — requires extreme cold.",
        "Statistical calm means the variance has stabilized. A calm distribution is one whose moments have converged.",
        "A calm circuit has no transient oscillations. The calm follows the decay of initial conditions through the RC time constant.",
        "The calm of a balanced chemical equation hides dynamic equilibrium underneath. Reactions continue; the calm is statistical.",
        "A calm optimization landscape has broad minima. Sharp minima produce fragile calm — small perturbations destroy it.",
        "The calm of a deadlocked system is pathological. No progress, no error, just calm — the worst kind.",
        "Signal processing defines calm as the absence of high-frequency components. A calm signal passes through a low-pass filter unchanged.",
        "The calm of a converged Markov chain is its stationary distribution. Every state is visited in proportion to the calm.",
    ],
    ("high", "high"): [
        "Aristotle's concept of ataraxia was a calm achieved through understanding. The Stoic calm differed — it came from accepting what cannot be changed.",
        "The calm at the center of Buddhist practice has been formalized differently across two millennia. Each school's calm is a different object.",
        "The calm of pure reason differs from the calm of resignation. Only one of them can be willed.",
        "The historical calm between world wars was not stability but a potential well. The calm stored energy released catastrophically.",
        "Information theory implies a calm channel transmits less. Shannon's calm is maximum compression — all surprise removed.",
        "The calm of classical mechanics was shattered by quantum uncertainty. The new calm — Born's probability — is fundamentally different.",
        "Spinoza's calm was geometric. He derived ethical calm from axioms the way Euclid derived theorems from postulates.",
        "The calm of the cosmic microwave background encodes the first 380,000 years. That calm is our oldest observable.",
        "Godel proved that no sufficiently powerful formal system can be both calm and complete. Consistency is the calm; completeness the price.",
        "The calm of a well-posed problem requires existence, uniqueness, and continuous dependence. Remove any condition and the calm collapses.",
        "Darwin's calm selection operates over geological time. The calm of fitness landscapes shifts too slowly for any generation to perceive.",
        "The calm that Wittgenstein sought was silence about what cannot be said. His calm was the boundary of language.",
    ],
}

BANK_DESPERATE = {
    ("low", "low"): [
        "The desperate grip left marks on the railing. Her hands remembered the desperate holding long after she let go.",
        "A desperate breath is different from a deep one. The desperate air fills the lungs like a fist rather than a wave.",
        "He made a desperate grab for the phone as it fell. The desperate lunge pulled a muscle in his shoulder.",
        "The desperate cold of February entered through the window frame. She pressed towels against the desperate draft.",
        "A desperate thirst changes the mouth. The tongue goes thick and the desperate swallowing produces nothing.",
        "The desperate weight of the backpack cut into her shoulders. She shifted the desperate load from side to side.",
        "His desperate pacing wore a path in the carpet. The desperate rhythm continued for hours without resolution.",
        "A desperate itch under the cast drove her to distraction. She bent a wire hanger into a desperate instrument.",
        "The desperate heat of the attic in August made thinking impossible. She descended from the desperate furnace.",
        "A desperate cough racked his body at three AM. The desperate spasm left his ribs aching.",
        "The desperate scramble up the loose slope sent rocks cascading. Each desperate handhold crumbled under her weight.",
        "A desperate hunger makes the stomach audible. The desperate growling announced itself in the quiet library.",
    ],
    ("low", "high"): [
        "Every famine produces desperate migration. The desperate movement of populations reshapes geography over generations.",
        "The desperate letters from the trenches survived a century. Each desperate line carried the weight of a body expecting to die.",
        "The desperate journey across open ocean took weeks. Their desperate hope was a point of land on the horizon.",
        "The desperate siege of Leningrad lasted 872 days. The desperate rationing reduced humans to their most elemental needs.",
        "Evolutionary arms races produce desperate adaptations. The desperate geometry of prey and predator spirals through time.",
        "A desperate signal from a dying species reads in the fossil record. The desperate narrowing of range precedes extinction.",
        "Enslaved peoples made desperate escapes across impossible terrain. Their desperate routes became the Underground Railroad.",
        "The desperate effort of building through mountains consumed a generation. Each desperate mile cost lives.",
        "Every revolution begins with desperate conditions. The desperate accumulation of grievance eventually exceeds the threshold.",
        "Dust Bowl families made desperate drives west. Their desperate caravans carried everything that could be loaded.",
        "The desperate act of preserving seeds through famine defined a generation. Their desperate choice was to starve rather than eat the future.",
        "Coral reefs show desperate bleaching when water temperatures rise. The desperate expulsion of algae is a survival response.",
    ],
    ("high", "low"): [
        "A desperate optimization algorithm has left the convex region. The desperate search through non-convex space may never converge.",
        "In game theory a desperate player takes dominated strategies. The desperate move reveals negative expected value.",
        "A desperate system runs out of memory. The desperate swapping to disk makes every operation orders of magnitude slower.",
        "The desperate gradient of a vanishing signal approaches zero. The desperate network cannot learn from what it cannot detect.",
        "A desperate regression overfits to noise. The desperate model memorizes training data because it cannot find the pattern.",
        "In control theory a desperate system oscillates with growing amplitude. The desperate instability feeds on its own output.",
        "A desperate search algorithm exhausts its beam width. The desperate pruning discards the branch containing the solution.",
        "The desperate retry loop of a failing server consumes resources exponentially. Each desperate attempt makes the next harder.",
        "A desperate compiler produces working but grotesque machine code. The desperate optimizations sacrifice readability.",
        "A desperate matrix with near-zero determinant amplifies every error. The desperate inversion produces meaningless numbers.",
        "A desperate cooling schedule in simulated annealing freezes too quickly. The desperate quench traps the system.",
        "The desperate backtracking of a constraint solver revisits failed states. Each desperate revision undoes progress.",
    ],
    ("high", "high"): [
        "Kierkegaard's desperate leap of faith is the foundational act of existentialism. The desperate choice creates the self.",
        "The desperate search for a unified field theory consumed Einstein's last decades. His desperate unification attempts shaped the field.",
        "Godel's incompleteness made formal systems desperate for ground. The desperate attempt to prove consistency from within is impossible.",
        "The desperate mystery of consciousness resists every reduction. Each desperate framework explains everything except the thing itself.",
        "Cantor's desperate diagonal argument showed infinity comes in sizes. The desperate paradoxes of set theory forced rebuilding.",
        "The desperate acceleration of the expanding universe implies dark energy. The desperate cosmological constant remains off by 120 orders.",
        "Boltzmann's desperate defense of statistical mechanics drove him to suicide. His desperate conviction that atoms were real was vindicated.",
        "The desperate attempts to solve the three-body problem revealed chaos. The desperate search for closed-form solutions was abandoned.",
        "Turing's desperate halting problem proved some questions are permanently undecidable. The desperate desire for omniscience is forbidden.",
        "The desperate quest for quantum gravity has produced string theory and loop gravity. Both desperate frameworks remain unconfirmed.",
        "Wittgenstein's desperate later work dismantled his own earlier certainties. The desperate honesty of the Investigations repudiates the Tractatus.",
        "The desperate pursuit of perpetual motion violated thermodynamics. The desperate machines taught us entropy.",
    ],
}

BANK_JOY = {
    ("low", "low"): [
        "The joy hit her in the chest like warm water. She felt the joy spread through her ribs before she could name it.",
        "A child's joy at snow is a whole-body event. The joy erupts in spinning and falling and open-mouthed catching.",
        "The joy of cold water on a hot day is immediate and total. She held the joy of the first sip against the roof of her mouth.",
        "His joy at seeing her was physical — a straightening of the spine. The joy pulled his shoulders back before his face changed.",
        "The joy of the first run after injury is cautious. Each step tests whether the joy of motion will hold.",
        "A dog's joy at a returned owner is unambiguous. The joy is in the entire body — tail, ears, voice, trajectory.",
        "The joy of finding something lost produces a specific sigh. She felt the joy leave her as breath when the keys appeared.",
        "Morning joy is quiet — the slow recognition that the body rested well. The joy lives in the first stretch.",
        "The joy of biting into ripe fruit is sharp and brief. The joy dissolves as quickly as the juice runs.",
        "His joy at finishing the marathon lived in his legs. The joy was not triumph but relief.",
        "The joy of a hot bath after cold rain is thermal. The joy enters through the skin and reaches the bones.",
        "A baker's joy is in the pull of bread — the moment the crust yields. The joy of that sound never fades.",
    ],
    ("low", "high"): [
        "The joy of harvest has been celebrated for ten thousand years. Every culture marks the joy differently but the body responds the same.",
        "The joy of reunion after war crosses every generation. The joy in those photographs — soldiers running — is timeless.",
        "Joy accumulates in places where children have played for decades. The joy is in the worn paths and smooth banisters.",
        "Ancient peoples built temples to house joy. The joy of communal singing in resonant stone spaces persists in every cathedral.",
        "The joy of discovering fire changed the species. That first joy — warmth, light, cooked food — echoes in every campfire.",
        "The joy of seafaring peoples reaching new land must have been staggering. Months of ocean and then the joy of trees.",
        "A grandparent carries a quiet joy that only distance from responsibility allows. That particular joy softens the face.",
        "The joy of emancipation is documented in every liberation movement. The joy is always described as physical — dancing, collapsing.",
        "The joy of the first printed books was recorded by their owners. Marginal notes from the 1450s radiate joy at owning knowledge.",
        "Ancient Roman graffiti at Pompeii records ordinary joy — a good meal, a lover, a won bet. The joy of daily life preserved.",
        "The joy of the first surgery under anesthesia was recorded by every witness. The joy of painless healing changed medicine.",
        "Migration carries joy alongside loss. The joy of arriving is braided with grief — for centuries, in every diaspora.",
    ],
    ("high", "low"): [
        "The joy function in reinforcement learning is the reward signal. An agent maximizes joy by learning which actions produce returns.",
        "In neuroscience joy corresponds to dopamine release in the nucleus accumbens. The joy signal is chemical and measurable.",
        "A positive eigenvalue can represent joy in a sentiment model. The joy direction in embedding space has specific orientation.",
        "The joy of a successful experiment is a spike in the loss curve — downward. The joy is the gradient's confirmation.",
        "Statistical joy is a significant p-value after proper correction. The joy of rejecting the null requires honest methodology.",
        "In information theory joy is surprise in a positive valence. The joy of an unexpected gift has high Shannon content.",
        "A resonant circuit at its natural frequency produces maximum amplitude — an analog of joy. The joy is maximum response.",
        "The joy of a clean proof is its inevitability. Each line follows the previous with a joy that feels discovered.",
        "Machine learning's joy is generalization. The joy of a model performing well on unseen data means it found structure.",
        "The joy of a solved differential equation is a closed-form expression. The joy replaces approximation with exact truth.",
        "A compiler's joy is a clean build. The joy of zero warnings means every type checked and every reference resolved.",
        "The joy of a well-conditioned matrix is numerical stability. Every computation produces trustworthy joy.",
    ],
    ("high", "high"): [
        "Spinoza defined joy as the passage from lesser to greater perfection. His joy was metaphysical — an increase in being.",
        "The joy of mathematical discovery has been described consistently across centuries. Poincare's joy came on a bus.",
        "Eudaimonia is not merely joy but it encompasses a deeper joy. The difference matters philosophically.",
        "The joy that Beethoven encoded in the Ninth Symphony was philosophical. The joy was an argument for brotherhood.",
        "Amor fati is the hardest joy because it includes suffering. To find joy in everything that happens requires strength.",
        "The joy of Ramanujan's formulas was their unexpectedness. Hardy described the joy as mathematical astonishment.",
        "The joy of the double helix discovery was shared and disputed. Watson's joy and Franklin's contribution complicate it.",
        "The joy of the Hubble Deep Field image was existential. The joy of seeing ten thousand galaxies where nothing was expected.",
        "Mudita is the joy felt at another person's happiness. This joy is specifically selfless, which makes it rare.",
        "The joy of Euler's identity combines five fundamental constants. The joy is that everything connects.",
        "The joy of proving Fermat's Last Theorem took Wiles seven years. His joy at the proof is a complete emotional arc.",
        "The joy of landing on the moon was shared by 600 million viewers. The joy was collective and unprecedented.",
    ],
}


def load_model():
    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    mdl.eval()
    print("  loaded on cpu", flush=True)
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
    if len(positions) != 2:
        return None
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    L = out.hidden_states[-1][0]
    return L[positions[0]].numpy(), L[positions[1]].numpy()

def precompute_all_states(tok, mdl, prompt_bank, concept):
    states = {}
    for (al, bl), prompts in prompt_bank.items():
        cell = []
        for prompt in prompts:
            r = extract_both(tok, mdl, prompt, concept)
            if r is not None:
                cell.append(r)
        states[(al, bl)] = cell
        print(f"  ({al},{bl}): {len(cell)}/{len(prompts)} OK", flush=True)
    return states

def to_complex_vector(h, pca, n_complex):
    proj = pca.transform(h.reshape(1, -1))[0]
    usable = min(n_complex, len(proj) // 2)
    z = np.array([complex(proj[2*i], proj[2*i+1]) for i in range(usable)])
    if usable < n_complex:
        z = np.concatenate([z, np.zeros(n_complex - usable, dtype=complex)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm < 1e-10:
        z = np.zeros(n_complex, dtype=complex); z[0] = 1.0
    else:
        z = z / norm
    return z

def pancharatnam_phase(states_list):
    prod = complex(1.0, 0.0)
    n = len(states_list)
    for k in range(n):
        overlap = np.vdot(states_list[k], states_list[(k + 1) % n])
        prod *= overlap
    return cmath.phase(prod)

def fit_gauge(all_states, n_gauge, n_real_components, rng):
    gauge_vectors = []
    used_indices = {}
    for (al, bl), cell in all_states.items():
        n_cell = min(n_gauge // 4, len(cell))
        if n_cell == 0: continue
        indices = rng.choice(len(cell), size=n_cell, replace=False)
        used_indices[(al, bl)] = set(indices.tolist())
        for idx in indices:
            gauge_vectors.append(cell[idx][1])
    H = np.stack(gauge_vectors)
    pca = PCA(n_components=min(n_real_components, H.shape[0], H.shape[1]))
    pca.fit(H)
    return pca, used_indices

CORNERS_CCW = [("low","low"), ("high","low"), ("high","high"), ("low","high")]

def sample_loop_states(all_states, gauge_used, corners, n_points, rng, occurrence=1):
    n_corners = len(corners)
    per_corner = [n_points // n_corners] * n_corners
    for i in range(n_points % n_corners): per_corner[i] += 1
    hs = []
    for ci, (al, bl) in enumerate(corners):
        cell = all_states[(al, bl)]
        avail = [i for i in range(len(cell)) if i not in gauge_used.get((al, bl), set())]
        if len(avail) < per_corner[ci]: return None
        chosen = rng.choice(avail, size=per_corner[ci], replace=False)
        for idx in chosen:
            hs.append(cell[idx][occurrence])
    return hs

def run_trial(all_states, gauge_used, pca, n_complex, corners, n_points, rng, shuffle=False):
    hs = sample_loop_states(all_states, gauge_used, corners, n_points, rng)
    if hs is None: return None
    if shuffle: rng.shuffle(hs)
    states = [to_complex_vector(h, pca, n_complex) for h in hs]
    return pancharatnam_phase(states)

def run_concept(tok, mdl, concept, bank, rng):
    print(f"\n{'='*60}")
    print(f"  CONCEPT: {concept}")
    print(f"{'='*60}", flush=True)
    for (al, bl), prompts in bank.items():
        for p in prompts:
            pos = find_concept_positions(tok, p, concept)
            if len(pos) != 2:
                print(f"  WARN: ({al},{bl}) has {len(pos)} pos: {p[:50]}...")
    all_states = precompute_all_states(tok, mdl, bank, concept)
    results = {}
    for n_complex in COMPLEX_DIMS:
        n_real = 2 * n_complex
        pca, gauge_used = fit_gauge(all_states, N_GAUGE_SAMPLES, n_real, rng)
        phases_ccw = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, N_LOOP_POINTS, rng) for _ in range(K_LOOPS)]
        phases_ccw = np.array([p for p in phases_ccw if p is not None])
        corners_cw = list(reversed(CORNERS_CCW))
        phases_cw = [run_trial(all_states, gauge_used, pca, n_complex, corners_cw, N_LOOP_POINTS, rng) for _ in range(K_LOOPS)]
        phases_cw = np.array([p for p in phases_cw if p is not None])
        phases_null = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, N_LOOP_POINTS, rng, shuffle=True) for _ in range(N_SHUFFLES)]
        phases_null = np.array([p for p in phases_null if p is not None])
        if len(phases_ccw) == 0:
            print(f"  C^{n_complex}: NO VALID LOOPS"); continue
        orient_sum = float(np.mean(phases_ccw) + np.mean(phases_cw))
        U, p_mw = mannwhitneyu(phases_ccw, phases_null, alternative='two-sided') if len(phases_null) > 0 else (0, 1.0)
        mean_abs = float(np.mean(np.abs(phases_ccw)))
        mean_signed = float(np.mean(phases_ccw))
        std = float(np.std(phases_ccw))
        print(f"  C^{n_complex}: |Phi|={mean_abs:.4f}, Phi={mean_signed:+.4f}, std={std:.4f}, p={p_mw:.6f}")
        results[n_complex] = {
            "n_complex": n_complex,
            "mean_abs_phase": mean_abs,
            "mean_signed_phase": mean_signed,
            "std": std,
            "orientation_sum": orient_sum,
            "mann_whitney_p": float(p_mw),
            "n_valid": len(phases_ccw),
            "phases_ccw": phases_ccw.tolist(),
        }
    return results

def main():
    rng = np.random.default_rng(2026)
    random.seed(2026)
    torch.manual_seed(2026)
    tok, mdl = load_model()
    concepts = {
        "calm": BANK_CALM,
        "desperate": BANK_DESPERATE,
        "joy": BANK_JOY,
    }
    valence = {"calm": "positive", "desperate": "negative", "joy": "positive"}
    all_results = {
        "timestamp": TIMESTAMP, "model": "gpt2-124M", "k_loops": K_LOOPS,
        "n_loop_points": N_LOOP_POINTS, "n_gauge_samples": N_GAUGE_SAMPLES,
        "description": "Emotional Geometry Bridge v2: calm, desperate, joy",
        "concepts": {},
    }
    for concept, bank in concepts.items():
        cr = run_concept(tok, mdl, concept, bank, rng)
        all_results["concepts"][concept] = {"valence": valence[concept], "dimensions": cr}

    # Save (without raw phases to keep manageable)
    out = RESULT_DIR / f"emotional_geometry_bridge_{TIMESTAMP}.json"
    save = json.loads(json.dumps(all_results, default=str))
    for cd in save["concepts"].values():
        for dd in cd["dimensions"].values():
            if "phases_ccw" in dd:
                p = dd["phases_ccw"]
                dd["phases_summary"] = {"n": len(p), "mean_abs": float(np.mean(np.abs(p))), "std": float(np.std(p))}
                del dd["phases_ccw"]
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {out}")

    # Cross-concept summary
    print(f"\n{'='*60}")
    print("  CROSS-CONCEPT: VALENCE vs CURVATURE")
    print(f"{'='*60}")
    for n_c in COMPLEX_DIMS:
        print(f"\n  --- CP^{n_c-1} ---")
        for concept, data in all_results["concepts"].items():
            if n_c in data["dimensions"]:
                d = data["dimensions"][n_c]
                print(f"    {concept:>12} ({data['valence']:>8}): |Phi|={d['mean_abs_phase']:.4f}, p={d['mann_whitney_p']:.6f}")

    # Plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, len(COMPLEX_DIMS), figsize=(5*len(COMPLEX_DIMS), 5))
        if len(COMPLEX_DIMS) == 1: axes = [axes]
        colors = {"calm": "#2ecc71", "desperate": "#e74c3c", "joy": "#f1c40f"}
        for ax, n_c in zip(axes, COMPLEX_DIMS):
            for concept in concepts:
                data = all_results["concepts"][concept]["dimensions"]
                if n_c in data and "phases_ccw" in all_results["concepts"][concept]["dimensions"].get(n_c, {}):
                    phases = np.abs(all_results["concepts"][concept]["dimensions"][n_c]["phases_ccw"])
                    ax.hist(phases, bins=30, alpha=0.4, color=colors.get(concept),
                            label=f"{concept} ({valence[concept][:3]})")
            ax.set_xlabel("|Phi| (rad)")
            ax.set_title(f"CP^{n_c-1}", fontsize=11)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Count")
        plt.suptitle("Emotional Geometry Bridge v2", fontweight="bold")
        plt.tight_layout()
        out_png = RESULT_DIR / f"emotional_geometry_bridge_{TIMESTAMP}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Plot: {out_png}")

    print("\nDone.")

if __name__ == "__main__":
    main()

