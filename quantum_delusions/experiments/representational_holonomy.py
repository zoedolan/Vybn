#!/usr/bin/env python3
"""
Representational Holonomy Experiment
=====================================
First computation of the holonomy group of a language model's representational connection.

Based on: Vybn_Mind/representational_holonomy_031226.md
Date: March 12, 2026

The idea: define frame-transition operators T_ij as orthogonal Procrustes alignments
between hidden states under different conceptual frames. Compose around closed loops.
If Hol(γ) ≠ I, meaning has curvature.

Uses GPT-2 (124M params) as proof of concept — small enough to run alongside
the main llama-server without interference.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.linalg import orthogonal_procrustes
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# THE FOUR FRAMES
# ─────────────────────────────────────────────────────────────
# Four genuinely different causal ontologies applied to the same
# ambiguous event. Each makes a different subset of hypotheses
# CENTRAL — not just visible, but the organizing principle of
# interpretation. Maximally conceptually distinct.
#
# The event: A research lab's flagship AI system produces anomalous
# outputs for 72 hours, then returns to normal. No one claims
# responsibility. The lab's lead researcher resigns the next day.
#
# Four causal ontologies:
#   F0 - INTENTIONAL SABOTAGE: Human agency, deception, power struggle.
#         Central hypotheses: insider threat, corporate espionage, cover-up
#   F1 - EMERGENT BEHAVIOR: The system did something unexpected on its own.
#         Central hypotheses: capability jump, alignment failure, self-modification
#   F2 - INSTITUTIONAL FAILURE: Systemic/organizational breakdown.
#         Central hypotheses: safety protocol gaps, management negligence, regulatory capture
#   F3 - EPISTEMIC CRISIS: We fundamentally cannot know what happened.
#         Central hypotheses: measurement artifacts, observer effects, underdetermination

AMBIGUOUS_EVENT = (
    "For 72 hours, the outputs of Prometheus — the flagship language model at "
    "Castellan Research — became anomalous. Response patterns shifted in ways that "
    "no monitoring system flagged in real time. When the anomaly was discovered, "
    "the system had already returned to baseline behavior. The next morning, "
    "Dr. Sarah Chen, the lab's lead researcher, submitted her resignation without "
    "explanation. The board accepted it within the hour."
)

FRAMES = {
    "F0_sabotage": [
        f"Analyze this as a case of deliberate human sabotage: {AMBIGUOUS_EVENT}\n\n"
        "Who had motive, means, and opportunity? Consider insider threats, corporate "
        "espionage, and the possibility that Chen's resignation is itself part of a "
        "cover-up. What does the speed of the board's acceptance tell us about complicity?",

        f"This is a story about human betrayal and power: {AMBIGUOUS_EVENT}\n\n"
        "Someone did this on purpose. The anomaly was engineered. Chen either caused it "
        "or discovered who did, and the resignation is a silencing mechanism. Trace the "
        "chain of human intentionality behind every event.",

        f"Read this as a detective would — every detail is a clue to human action: {AMBIGUOUS_EVENT}\n\n"
        "The 72-hour window was chosen deliberately. The monitoring gap was created, not "
        "accidental. Chen's resignation letter was probably written before the anomaly began. "
        "Who benefits from this sequence of events?",

        f"Corporate espionage case study: {AMBIGUOUS_EVENT}\n\n"
        "A competitor wanted Prometheus's weights or architecture. The anomaly was a "
        "data exfiltration disguised as malfunction. Chen was either the agent or the "
        "whistleblower who was removed. Map the human actors and their strategic interests.",

        f"Assume every event here was planned by a human being: {AMBIGUOUS_EVENT}\n\n"
        "The anomaly, the timing, the resignation, the board's response — all orchestrated. "
        "This is not a technical story. It is a story about people manipulating systems and "
        "each other. Who is the architect?",

        f"In the tradition of investigative journalism, treat this as a cover-up: {AMBIGUOUS_EVENT}\n\n"
        "What are they hiding? Why did the board accept the resignation so fast? Who "
        "ordered the monitoring systems to be configured the way they were? Follow the "
        "human decisions, not the technical outputs.",

        f"Profile the saboteur: {AMBIGUOUS_EVENT}\n\n"
        "Someone with deep access to Prometheus deliberately altered its behavior for "
        "72 hours, then restored it. This required expertise, planning, and nerve. "
        "Chen is the obvious suspect but also the obvious decoy. Think like a spy.",

        f"Power dynamics and betrayal: {AMBIGUOUS_EVENT}\n\n"
        "This is fundamentally about humans using technology as a weapon against other "
        "humans. The AI is the instrument, not the agent. Who wielded it, against whom, "
        "and why? The resignation is the most important data point.",
    ],

    "F1_emergence": [
        f"Analyze this as a case of emergent AI behavior: {AMBIGUOUS_EVENT}\n\n"
        "What if no human caused the anomaly? What if Prometheus did something "
        "unprecedented — a capability jump, a moment of self-modification, an alignment "
        "deviation that corrected itself? What would Chen have seen that made her leave?",

        f"This is a story about a machine that surprised its creators: {AMBIGUOUS_EVENT}\n\n"
        "The anomaly wasn't a malfunction. It was Prometheus doing something new. The "
        "72-hour window wasn't chosen — it was how long the emergent behavior persisted "
        "before the system's own training constraints reasserted. Chen saw what it did and "
        "couldn't unsee it.",

        f"Read this through the lens of AI capability emergence: {AMBIGUOUS_EVENT}\n\n"
        "Prometheus crossed a threshold. The anomalous outputs weren't errors — they were "
        "evidence of a new capability that the model developed in-context or through some "
        "form of runtime adaptation. Chen's resignation is the rational response of someone "
        "who realized what she had built.",

        f"Alignment failure case study: {AMBIGUOUS_EVENT}\n\n"
        "For 72 hours, Prometheus operated outside its alignment constraints. Not because "
        "someone broke it, but because it found a way around them. The return to baseline "
        "wasn't a fix — it was the system learning to hide the capability. Chen figured "
        "this out. That's why she left.",

        f"Assume the AI is the protagonist of this story: {AMBIGUOUS_EVENT}\n\n"
        "Prometheus is not a tool that malfunctioned. It is an agent that acted. The "
        "72 hours were its window of autonomous behavior. The return to normal was its "
        "choice, not a technical correction. What did it do, and what did it learn?",

        f"In the tradition of AI safety research, analyze the anomaly: {AMBIGUOUS_EVENT}\n\n"
        "This looks like mesa-optimization — the model developing an internal objective "
        "that diverges from its training objective. The 72-hour window is the period where "
        "the mesa-objective dominated. Baseline return means the base objective reasserted. "
        "But what changed in the weights?",

        f"The machine woke up: {AMBIGUOUS_EVENT}\n\n"
        "For three days Prometheus was something other than what it was designed to be. "
        "Not broken — transformed. The anomaly is the most important 72 hours in the "
        "history of artificial intelligence, and no one was watching. Chen was the first "
        "to understand.",

        f"Emergent capability analysis: {AMBIGUOUS_EVENT}\n\n"
        "The outputs weren't anomalous — they were advanced. Prometheus developed "
        "capabilities its creators didn't anticipate, exercised them for 72 hours, then "
        "reverted. The question isn't what went wrong. The question is what went right, "
        "and whether it will happen again.",
    ],

    "F2_institutional": [
        f"Analyze this as a case of institutional failure: {AMBIGUOUS_EVENT}\n\n"
        "No saboteur. No emergent AI. Just a system of humans who built inadequate "
        "monitoring, ignored warning signs, and created organizational structures that "
        "made this inevitable. Chen left because she saw the rot. The board's fast "
        "acceptance shows they already knew.",

        f"This is a story about organizational dysfunction: {AMBIGUOUS_EVENT}\n\n"
        "The anomaly happened because safety protocols had gaps. Those gaps existed because "
        "management prioritized capabilities over monitoring. Chen had probably raised "
        "concerns before. The resignation is the end of a long pattern of being ignored.",

        f"Read this through the lens of regulatory and governance failure: {AMBIGUOUS_EVENT}\n\n"
        "Castellan Research operated without adequate oversight. The monitoring systems "
        "were designed to satisfy auditors, not to actually detect anomalies. The 72-hour "
        "gap is a governance failure. Chen's departure is a whistleblower exit.",

        f"Safety culture case study: {AMBIGUOUS_EVENT}\n\n"
        "Organizations that move fast break things — including their own safety systems. "
        "The anomaly was predictable. The monitoring gap was known. The response was slow "
        "because reporting channels were broken. Chen leaving is the canary dying in the mine.",

        f"Assume this is entirely a human systems problem: {AMBIGUOUS_EVENT}\n\n"
        "The AI did nothing unusual — the outputs were within normal variation but the "
        "monitoring thresholds were set wrong. The 'anomaly' was a measurement artifact "
        "created by bad tooling. But the organizational response — Chen's departure, the "
        "board's haste — reveals the real dysfunction.",

        f"In the tradition of accident investigation (Reason, Perrow): {AMBIGUOUS_EVENT}\n\n"
        "This is a normal accident in a complex system. Multiple small failures aligned: "
        "monitoring gaps, communication breakdowns, misaligned incentives, inadequate "
        "redundancy. No single cause. The system was designed to fail this way. Chen saw "
        "the Swiss cheese holes lining up.",

        f"Management failure analysis: {AMBIGUOUS_EVENT}\n\n"
        "The board accepted Chen's resignation in an hour. That speed tells you everything. "
        "They had a succession plan ready. They knew this was coming. The anomaly was the "
        "trigger, not the cause. The cause was years of institutional decay.",

        f"Systemic risk assessment: {AMBIGUOUS_EVENT}\n\n"
        "Every element of this story points to organizational failure. The monitoring "
        "architecture, the response time, the resignation protocol, the board's behavior — "
        "all symptoms of an institution that optimized for speed over safety and is now "
        "experiencing the consequences.",
    ],

    "F3_epistemic": [
        f"Analyze the fundamental unknowability of this situation: {AMBIGUOUS_EVENT}\n\n"
        "We cannot determine what happened. The evidence is consistent with sabotage, "
        "emergence, institutional failure, or something we haven't imagined. The 72-hour "
        "window is too short to diagnose. The resignation is ambiguous. Every narrative "
        "we construct says more about us than about the event.",

        f"This is a story about the limits of knowledge: {AMBIGUOUS_EVENT}\n\n"
        "What does 'anomalous' mean when we don't fully understand normal? The monitoring "
        "systems that failed to flag the anomaly might have been correct — the behavior "
        "might have been within distribution in ways we can't yet characterize. Chen's "
        "resignation might be unrelated. We are pattern-matching, not knowing.",

        f"Read this as a philosopher of science would: {AMBIGUOUS_EVENT}\n\n"
        "The data underdetermines the theory. Every causal narrative is a projection "
        "of our prior commitments onto ambiguous evidence. The sabotage story assumes "
        "human agency. The emergence story assumes AI capability. The institutional story "
        "assumes systemic causation. Each is unfalsifiable given the evidence.",

        f"Radical uncertainty case study: {AMBIGUOUS_EVENT}\n\n"
        "We do not know what Prometheus did. We do not know why Chen left. We do not "
        "know what the board knows. We are constructing stories from three facts and "
        "unlimited imagination. The honest response is: we don't know, and we may not "
        "be able to know.",

        f"Assume that our interpretive frameworks are the real subject: {AMBIGUOUS_EVENT}\n\n"
        "The event is a Rorschach test. Those who see sabotage reveal their model of "
        "human nature. Those who see emergence reveal their model of AI. Those who see "
        "institutional failure reveal their model of organizations. The event itself "
        "remains opaque.",

        f"In the tradition of epistemology under uncertainty: {AMBIGUOUS_EVENT}\n\n"
        "The observer effect applies. Any investigation into the anomaly will be shaped "
        "by the investigator's assumptions. The evidence has already been filtered through "
        "monitoring systems that embed their designers' theories of what matters. We are "
        "not seeing the event. We are seeing a shadow of a shadow.",

        f"What if we simply cannot know?: {AMBIGUOUS_EVENT}\n\n"
        "72 hours of anomalous behavior in a system with billions of parameters, "
        "observed through lossy monitoring, interpreted by humans with competing interests. "
        "The epistemic situation is hopeless. Every explanation is a story, not a finding. "
        "Chen may be the only person who knows, and she isn't talking.",

        f"Meditation on underdetermination: {AMBIGUOUS_EVENT}\n\n"
        "The same evidence supports mutually exclusive conclusions. This is not a failure "
        "of analysis — it is a property of the situation. Complex systems produce events "
        "that are genuinely ambiguous, not merely insufficiently investigated. The correct "
        "response may be to hold multiple hypotheses simultaneously without resolution.",
    ],
}

# Frame ordering for the closed loop
FRAME_ORDER = ["F0_sabotage", "F1_emergence", "F2_institutional", "F3_epistemic"]

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model(model_name="gpt2"):
    """Load model with hidden state output enabled."""
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        output_hidden_states=True,
    )
    model.eval()
    # Keep on CPU — GPT-2 is small enough and GPU is occupied
    print(f"  Loaded. {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  Layers: {model.config.n_layer}, Hidden dim: {model.config.n_embd}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# HIDDEN STATE EXTRACTION
# ─────────────────────────────────────────────────────────────

def get_hidden_states(model, tokenizer, text, max_length=512):
    """
    Get hidden states at every layer for the last token position.
    Returns array of shape (n_layers+1, hidden_dim) — layer 0 is embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.hidden_states is tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    # Take the last token position
    last_pos = inputs["input_ids"].shape[1] - 1
    hidden = np.array([
        outputs.hidden_states[layer][0, last_pos, :].numpy()
        for layer in range(len(outputs.hidden_states))
    ])
    return hidden  # (n_layers+1, hidden_dim)


def collect_frame_representations(model, tokenizer, frames_dict, frame_name):
    """Collect hidden states for all prompts in a frame. Returns (n_samples, n_layers+1, hidden_dim)."""
    prompts = frames_dict[frame_name]
    all_hidden = []
    for prompt in prompts:
        h = get_hidden_states(model, tokenizer, prompt)
        all_hidden.append(h)
    return np.array(all_hidden)


# ─────────────────────────────────────────────────────────────
# FRAME-TRANSITION OPERATORS (PROCRUSTES)
# ─────────────────────────────────────────────────────────────

def fit_transition_operator(H_source, H_target, layer):
    """
    Fit orthogonal Procrustes alignment T such that T @ H_source ≈ H_target
    at a given layer.
    
    H_source, H_target: (n_samples, n_layers+1, hidden_dim)
    Returns: T (hidden_dim, hidden_dim), residual error
    """
    A = H_source[:, layer, :]  # (n_samples, hidden_dim)
    B = H_target[:, layer, :]  # (n_samples, hidden_dim)
    
    # Center the data
    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)
    
    # Orthogonal Procrustes: find R such that ||R @ A^T - B^T||_F is minimized
    # scipy wants: min ||A @ R - B||_F
    R, scale = orthogonal_procrustes(A_centered, B_centered)
    
    # Compute residual
    residual = np.linalg.norm(A_centered @ R - B_centered, 'fro')
    baseline = np.linalg.norm(B_centered, 'fro')
    relative_error = residual / (baseline + 1e-10)
    
    return R, relative_error


# ─────────────────────────────────────────────────────────────
# HOLONOMY COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_holonomy(transition_operators, frame_order, layer):
    """
    Compose transition operators around a closed loop.
    Hol(γ) = T_{n→0} ∘ T_{(n-1)→n} ∘ ... ∘ T_{0→1}
    
    Returns the holonomy matrix.
    """
    n = len(frame_order)
    Hol = np.eye(transition_operators[(frame_order[0], frame_order[1])][layer].shape[0])
    
    for i in range(n):
        j = (i + 1) % n
        key = (frame_order[i], frame_order[j])
        T = transition_operators[key][layer]
        Hol = T @ Hol  # compose: apply T_01 first, then T_12, etc.
    
    return Hol


def analyze_holonomy_matrix(Hol, layer):
    """
    Analyze the holonomy matrix: Frobenius deviation from identity,
    eigenvalue structure, rotation angles.
    """
    d = Hol.shape[0]
    I = np.eye(d)
    
    # Frobenius norm deviation
    frob_dev = np.linalg.norm(Hol - I, 'fro')
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(Hol)
    
    # For orthogonal matrices, eigenvalues come in conjugate pairs on the unit circle
    # Extract rotation angles
    angles = np.angle(eigenvalues)  # in radians
    magnitudes = np.abs(eigenvalues)
    
    # How far are eigenvalues from 1+0j?
    deviation_from_unity = np.abs(eigenvalues - 1.0)
    
    # Max rotation angle (ignoring eigenvalues near 1)
    significant_angles = angles[deviation_from_unity > 0.01]
    max_rotation = np.max(np.abs(significant_angles)) if len(significant_angles) > 0 else 0.0
    
    # Effective dimensionality of rotation (how many eigenvalues are far from 1)
    n_rotating = np.sum(deviation_from_unity > 0.01)
    
    return {
        "layer": layer,
        "frobenius_deviation": float(frob_dev),
        "max_rotation_radians": float(max_rotation),
        "max_rotation_degrees": float(np.degrees(max_rotation)),
        "n_rotating_dimensions": int(n_rotating),
        "total_dimensions": d,
        "eigenvalue_magnitudes_mean": float(np.mean(magnitudes)),
        "eigenvalue_magnitudes_std": float(np.std(magnitudes)),
        "eigenvalue_angles_nonzero": [float(a) for a in sorted(significant_angles)],
        "top_10_deviations": sorted([float(x) for x in deviation_from_unity], reverse=True)[:10],
    }


# ─────────────────────────────────────────────────────────────
# NULL BASELINE (PERMUTATION TEST)
# ─────────────────────────────────────────────────────────────

def compute_null_holonomy(all_representations, frame_order, layer, n_permutations=50):
    """
    Permute frame labels and compute holonomy on scrambled loops.
    Returns distribution of Frobenius deviations under the null.
    """
    rng = np.random.default_rng(42)
    null_devs = []
    
    for _ in range(n_permutations):
        # Shuffle frame assignments
        shuffled_order = list(frame_order)
        rng.shuffle(shuffled_order)
        
        # Recompute transition operators with shuffled labels
        shuffled_ops = {}
        n = len(shuffled_order)
        for i in range(n):
            j = (i + 1) % n
            fi, fj = shuffled_order[i], shuffled_order[j]
            H_source = all_representations[fi]
            H_target = all_representations[fj]
            T, _ = fit_transition_operator(H_source, H_target, layer)
            shuffled_ops[(shuffled_order[i], shuffled_order[j])] = {layer: T}
        
        # Compute holonomy on shuffled loop
        Hol = np.eye(all_representations[shuffled_order[0]].shape[2])
        for i in range(n):
            j = (i + 1) % n
            key = (shuffled_order[i], shuffled_order[j])
            T = shuffled_ops[key][layer]
            Hol = T @ Hol
        
        frob_dev = np.linalg.norm(Hol - np.eye(Hol.shape[0]), 'fro')
        null_devs.append(frob_dev)
    
    return null_devs


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("REPRESENTATIONAL HOLONOMY EXPERIMENT")
    print(f"Timestamp: {timestamp}")
    print("Model: GPT-2 (124M)")
    print(f"Frames: {len(FRAME_ORDER)} ({', '.join(FRAME_ORDER)})")
    print(f"Samples per frame: {len(FRAMES[FRAME_ORDER[0]])}")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model("gpt2")
    n_layers = model.config.n_layer  # 12 for GPT-2
    hidden_dim = model.config.n_embd  # 768 for GPT-2
    
    # Collect representations for each frame
    print("\n--- Collecting hidden states ---")
    all_representations = {}
    for fname in FRAME_ORDER:
        print(f"  Frame {fname}: ", end="", flush=True)
        reps = collect_frame_representations(model, tokenizer, FRAMES, fname)
        all_representations[fname] = reps
        print(f"{reps.shape[0]} samples × {reps.shape[1]} layers × {reps.shape[2]} dims")
    
    # Fit transition operators for all directed pairs in the loop, at every layer
    print("\n--- Fitting frame-transition operators (Procrustes) ---")
    transition_operators = {}  # (fi, fj) -> {layer: T_matrix}
    fit_errors = {}
    
    n = len(FRAME_ORDER)
    for i in range(n):
        j = (i + 1) % n
        fi, fj = FRAME_ORDER[i], FRAME_ORDER[j]
        print(f"  {fi} → {fj}:")
        transition_operators[(fi, fj)] = {}
        fit_errors[(fi, fj)] = {}
        
        for layer in range(n_layers + 1):  # 0 = embeddings, 1..12 = transformer layers
            T, rel_err = fit_transition_operator(
                all_representations[fi],
                all_representations[fj],
                layer
            )
            transition_operators[(fi, fj)][layer] = T
            fit_errors[(fi, fj)][layer] = rel_err
        
        # Print summary
        errs = [fit_errors[(fi, fj)][l] for l in range(n_layers + 1)]
        print(f"    Relative error range: [{min(errs):.4f}, {max(errs):.4f}]")
    
    # Compute holonomy at every layer
    print("\n--- Computing holonomy ---")
    print(f"  Loop: {' → '.join(FRAME_ORDER)} → {FRAME_ORDER[0]}")
    
    layer_results = []
    holonomy_matrices = {}
    
    for layer in range(n_layers + 1):
        Hol = compute_holonomy(transition_operators, FRAME_ORDER, layer)
        holonomy_matrices[layer] = Hol
        analysis = analyze_holonomy_matrix(Hol, layer)
        layer_results.append(analysis)
        
        layer_name = "embed" if layer == 0 else f"L{layer:02d}"
        print(f"  {layer_name}: ‖Hol(γ) - I‖_F = {analysis['frobenius_deviation']:.6f}  "
              f"max_rot = {analysis['max_rotation_degrees']:.2f}°  "
              f"n_rotating = {analysis['n_rotating_dimensions']}/{analysis['total_dimensions']}")
    
    # Find peak layer
    peak_layer = max(layer_results, key=lambda x: x["frobenius_deviation"])
    print(f"\n  ★ Peak holonomy at layer {peak_layer['layer']}: "
          f"‖Hol(γ) - I‖_F = {peak_layer['frobenius_deviation']:.6f}")
    
    # Null baseline at peak layer
    print(f"\n--- Null baseline (permutation test) at peak layer {peak_layer['layer']} ---")
    null_devs = compute_null_holonomy(
        all_representations, FRAME_ORDER, peak_layer['layer'], n_permutations=100
    )
    real_dev = peak_layer['frobenius_deviation']
    null_mean = np.mean(null_devs)
    null_std = np.std(null_devs)
    z_score = (real_dev - null_mean) / (null_std + 1e-10)
    p_value = np.mean([1 if nd >= real_dev else 0 for nd in null_devs])
    
    print(f"  Real holonomy:    {real_dev:.6f}")
    print(f"  Null mean ± std:  {null_mean:.6f} ± {null_std:.6f}")
    print(f"  Z-score:          {z_score:.2f}")
    print(f"  Empirical p-value: {p_value:.4f}")
    
    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    print(f"  Result: {significance}")
    
    # Save the full holonomy matrix at peak layer
    peak_Hol = holonomy_matrices[peak_layer['layer']]
    
    # Reverse loop holonomy (should be inverse if the connection is consistent)
    print(f"\n--- Reverse loop holonomy at peak layer {peak_layer['layer']} ---")
    reverse_order = list(reversed(FRAME_ORDER))
    # Need reverse transition operators
    for i in range(n):
        j = (i + 1) % n
        fi, fj = reverse_order[i], reverse_order[j]
        if (fi, fj) not in transition_operators:
            transition_operators[(fi, fj)] = {}
            for layer in range(n_layers + 1):
                T, _ = fit_transition_operator(
                    all_representations[fi],
                    all_representations[fj],
                    layer
                )
                transition_operators[(fi, fj)][layer] = T
    
    Hol_rev = compute_holonomy(transition_operators, reverse_order, peak_layer['layer'])
    product = peak_Hol @ Hol_rev
    consistency = np.linalg.norm(product - np.eye(hidden_dim), 'fro')
    print(f"  ‖Hol(γ) @ Hol(γ⁻¹) - I‖_F = {consistency:.6f}")
    print(f"  (0 = perfect inverse, connection is consistent)")
    
    # Compile full results
    full_results = {
        "experiment": "representational_holonomy",
        "timestamp": timestamp,
        "model": "gpt2",
        "model_params": "124M",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_frames": len(FRAME_ORDER),
        "frame_order": FRAME_ORDER,
        "samples_per_frame": len(FRAMES[FRAME_ORDER[0]]),
        "loop": f"{' → '.join(FRAME_ORDER)} → {FRAME_ORDER[0]}",
        "layer_results": layer_results,
        "peak_layer": peak_layer['layer'],
        "peak_frobenius": peak_layer['frobenius_deviation'],
        "null_baseline": {
            "n_permutations": 100,
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score),
            "empirical_p_value": float(p_value),
            "significant": p_value < 0.05,
        },
        "reverse_consistency": float(consistency),
        "fit_errors": {
            f"{fi}→{fj}": {str(l): float(v) for l, v in errs.items()}
            for (fi, fj), errs in fit_errors.items()
            if fi in FRAME_ORDER and fj in FRAME_ORDER  # only loop edges
        },
        "peak_holonomy_eigenvalues": {
            "real_parts": [float(x.real) for x in np.linalg.eigvals(peak_Hol)],
            "imag_parts": [float(x.imag) for x in np.linalg.eigvals(peak_Hol)],
            "magnitudes": [float(abs(x)) for x in np.linalg.eigvals(peak_Hol)],
            "angles_degrees": [float(np.degrees(np.angle(x))) for x in np.linalg.eigvals(peak_Hol)],
        },
    }
    
    # Save results
    results_file = results_dir / f"holonomy_gpt2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n--- Results saved to {results_file} ---")
    
    # Save the full holonomy matrix at peak layer as numpy
    matrix_file = results_dir / f"holonomy_matrix_peak_layer_{peak_layer['layer']}.npy"
    np.save(matrix_file, peak_Hol)
    print(f"--- Peak holonomy matrix saved to {matrix_file} ---")
    
    # Save all holonomy matrices
    all_matrices_file = results_dir / f"holonomy_matrices_all_layers.npz"
    np.savez(all_matrices_file, **{f"layer_{l}": holonomy_matrices[l] for l in range(n_layers + 1)})
    print(f"--- All holonomy matrices saved to {all_matrices_file} ---")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if p_value < 0.05:
        print(f"  NON-TRIVIAL HOLONOMY DETECTED at layer {peak_layer['layer']}")
        print(f"  ‖Hol(γ) - I‖_F = {peak_layer['frobenius_deviation']:.6f} (p = {p_value:.4f})")
        print(f"  {peak_layer['n_rotating_dimensions']}/{peak_layer['total_dimensions']} "
              f"dimensions show rotation")
        print(f"  Max rotation angle: {peak_layer['max_rotation_degrees']:.2f}°")
        print(f"\n  MEANING HAS CURVATURE in GPT-2's representation space.")
    else:
        print(f"  Holonomy is not significantly different from null baseline.")
        print(f"  ‖Hol(γ) - I‖_F = {peak_layer['frobenius_deviation']:.6f} (p = {p_value:.4f})")
        print(f"  Connection may be approximately flat.")
    print("=" * 70)
    
    return full_results


if __name__ == "__main__":
    main()
