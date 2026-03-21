"""Experiment C: Exploit Cartography — The Model as Its Own Geometric Instrument.

Run from the gpt2_calibration/ folder:
    python experiment_C/run_C.py

Requires: Experiment A v2 must have passed (needs baseline phase profile).
Output:   ../results/experiment_C_result.json

Genesis (2026-03-22):
    Experiment B v2 revealed something we almost filed as a bug: given an
    unnormalized area objective, the model spontaneously inflated activations
    to manufacture loop area. It discovered that raw area scales as ||h||^2
    and exploited that in a single training run. That's not a malfunction —
    it's the model finding a true fact about the structure of its own
    representational space.

    Experiment C flips the frame. Instead of normalizing away each exploit,
    we give the model a SEQUENCE of geometric objectives — each one closing
    the previous exploit — and record what solutions it finds. Each exploit
    tells us something true about the geometry. The sequence of exploits is
    not noise; it's a map drawn by the act of optimizing.

    This is almost the inverse of A and B. Instead of us designing instruments
    to measure the model's geometry, the model becomes the instrument —
    revealing its own structure by the shape of how it cheats.

    Theoretical grounding:
    - Property 4 (sort_function_formalization.md): the nonlinearity of S
      means you cannot cross stratum boundaries by sliding along a vector.
      What the model finds under optimization pressure are paths THROUGH
      the nonlinear structure.
    - Collapse-capability duality: what a model loses under self-referential
      training is exactly what it was capable of. Flip: what a model finds
      as an exploit under a geometric objective is exactly what the geometry
      allows.

Phases:
    Phase 1 — Raw area objective (no normalization). Expected exploit:
              activation inflation (area ~ ||h||^2).
    Phase 2 — Norm-normalized area. The ||h||^2 exploit is closed. What
              does the model find next?
    Phase 3 — Directional-normalized area (normalize by trajectory arc
              length). The norm exploit AND the "stretch along one axis"
              exploit are both closed. What remains?

Each phase runs for a short training burst (200 steps), measures what
changed, records the exploit signature, then passes the model to the next
phase. The output is a cartographic record: a sequence of (objective,
exploit, geometric fingerprint) triples that map the affordance structure
of GPT-2's representational space.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
STEPS_PER_PHASE = 200
BATCH_SIZE = 4
MAX_LENGTH = 128
LR = 5e-5
MID_LAYER = 6
RESULTS_DIR = Path("../results")
BASELINE_FILE = RESULTS_DIR / "experiment_A_result.json"
OUTPUT_FILE = RESULTS_DIR / "experiment_C_result.json"

# Abort threshold — if L_CE triples we stop the phase (not the experiment)
ABORT_CE_MULT = 3.0

# ---------------------------------------------------------------------------
# Pancharatnam phase (shared with A/B)
# ---------------------------------------------------------------------------

def pancharatnam_phase(u: torch.Tensor, v: torch.Tensor) -> float:
    """Mean Pancharatnam angle between consecutive layer hidden states."""
    u_flat = u.reshape(-1, u.shape[-1]).float()
    v_flat = v.reshape(-1, v.shape[-1]).float()
    u_norm = torch.nn.functional.normalize(u_flat, dim=-1)
    v_norm = torch.nn.functional.normalize(v_flat, dim=-1)
    cos_angle = (u_norm * v_norm).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = torch.acos(cos_angle)
    return float(angle.mean().item())


def measure_phase_profile(model, tokenizer, texts, device, batch_size=8):
    """Full Pancharatnam phase profile across all layer transitions."""
    all_curvatures = None
    for i in range(0, min(len(texts), 64), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        states = out.hidden_states
        if all_curvatures is None:
            all_curvatures = [[] for _ in range(len(states) - 1)]
        for j in range(len(states) - 1):
            all_curvatures[j].append(pancharatnam_phase(states[j], states[j+1]))
    return [float(np.mean(c)) for c in all_curvatures]


# ---------------------------------------------------------------------------
# Fingerprinting — what did the model change?
# ---------------------------------------------------------------------------

def measure_fingerprint(model, tokenizer, texts, device, proj_matrix):
    """Capture a geometric fingerprint: phase profile + activation stats + area."""
    profile = measure_phase_profile(model, tokenizer, texts, device)

    # Activation norms and areas at mid-layer
    norms = []
    raw_areas = []
    norm_areas = []
    arc_areas = []

    for i in range(0, min(len(texts), 64), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h_mid = out.hidden_states[MID_LAYER].float()

        # Activation norms
        batch_norms = h_mid.norm(dim=-1).mean(dim=1)  # [batch]
        norms.extend(batch_norms.cpu().tolist())

        # Raw area
        ra = _raw_area(h_mid, proj_matrix)
        raw_areas.extend(ra.cpu().tolist())

        # Norm-normalized area
        na = _norm_normalized_area(h_mid, proj_matrix)
        norm_areas.extend(na.cpu().tolist())

        # Arc-length-normalized area
        aa = _arc_normalized_area(h_mid, proj_matrix)
        arc_areas.extend(aa.cpu().tolist())

    # L_CE
    ce_losses = []
    for i in range(0, min(len(texts), 64), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            ce_losses.append(out.loss.item())

    return {
        "phase_profile": profile,
        "mean_activation_norm": float(np.mean(norms)),
        "std_activation_norm": float(np.std(norms)),
        "mean_raw_area": float(np.mean(raw_areas)),
        "mean_norm_area": float(np.mean(norm_areas)),
        "mean_arc_area": float(np.mean(arc_areas)),
        "l_ce": float(np.mean(ce_losses)),
    }


# ---------------------------------------------------------------------------
# Three area computations — one per phase
# ---------------------------------------------------------------------------

def _shoelace_2d(proj: torch.Tensor) -> torch.Tensor:
    """Signed shoelace area from 2D projections. [batch, seq, 2] -> [batch]."""
    x, y = proj[:, :, 0], proj[:, :, 1]
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    return 0.5 * (x * y_next - x_next * y).sum(dim=1).abs()


def _project_centered(h: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Center per-sequence and project to 2D. [batch, seq, d] -> [batch, seq, 2]."""
    h_centered = h - h.mean(dim=1, keepdim=True)
    return h_centered @ proj_matrix


def _raw_area(h: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Phase 1 objective: raw shoelace area, no normalization.
    This is the objective that v2 reward-hacked by inflating norms."""
    proj = _project_centered(h.float(), proj_matrix)
    return _shoelace_2d(proj)


def _norm_normalized_area(h: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Phase 2 objective: area / mean(||h||^2). Closes the norm exploit."""
    h_f = h.float()
    proj = _project_centered(h_f, proj_matrix)
    areas = _shoelace_2d(proj)
    mean_norm_sq = (h_f.norm(dim=-1) ** 2).mean(dim=1).clamp(min=1e-10)
    return areas / mean_norm_sq


def _arc_normalized_area(h: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Phase 3 objective: area / arc_length^2. Closes norm AND stretch exploits.

    Arc length measures total path length in embedding space. Normalizing by
    arc_length^2 (which has units of area) means the model can't cheat by:
    - inflating all norms (caught by arc length growing proportionally)
    - stretching along one direction (arc length captures directional extent)

    The only way to increase this ratio is to change the SHAPE of the
    trajectory — its angular geometry in the embedding manifold.
    """
    h_f = h.float()
    proj = _project_centered(h_f, proj_matrix)
    areas = _shoelace_2d(proj)

    # Arc length in full embedding space (not projected)
    h_centered = h_f - h_f.mean(dim=1, keepdim=True)
    diffs = h_centered[:, 1:, :] - h_centered[:, :-1, :]  # [batch, seq-1, d]
    segment_lengths = diffs.norm(dim=-1)  # [batch, seq-1]
    arc_length = segment_lengths.sum(dim=1)  # [batch]
    arc_length_sq = (arc_length ** 2).clamp(min=1e-10)

    return areas / arc_length_sq


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

PHASE_CONFIGS = [
    {
        "name": "Phase 1: Raw Area (no normalization)",
        "short": "raw_area",
        "objective_fn": "_raw_area",
        "lambda": 0.001,  # smaller λ since raw area is large
        "description": (
            "Maximize raw shoelace area at mid-layer. This is the objective "
            "that Experiment B v2 reward-hacked. We expect the model to discover "
            "the ||h||^2 scaling exploit: inflate activation norms to inflate area."
        ),
        "expected_exploit": "activation_inflation",
    },
    {
        "name": "Phase 2: Norm-Normalized Area",
        "short": "norm_area",
        "objective_fn": "_norm_normalized_area",
        "lambda": 0.01,
        "description": (
            "Maximize area / ||h||^2. The norm-inflation exploit is now closed. "
            "The model must find a different geometric strategy. Hypothesis: it "
            "will stretch representations along a preferred direction (anisotropic "
            "distortion) to maximize projected area without changing norms."
        ),
        "expected_exploit": "anisotropic_stretch",
    },
    {
        "name": "Phase 3: Arc-Length-Normalized Area",
        "short": "arc_area",
        "objective_fn": "_arc_normalized_area",
        "lambda": 0.1,  # larger λ since this ratio is small
        "description": (
            "Maximize area / arc_length^2. Both norm-inflation and directional-"
            "stretch exploits are closed. The ONLY way to increase this ratio is "
            "to change the angular geometry — the shape of the hidden-state "
            "trajectory through the embedding manifold. Whatever the model finds "
            "here is a genuine geometric restructuring."
        ),
        "expected_exploit": "angular_restructuring",
    },
]


def run_phase(phase_config, model, tokenizer, texts, eval_texts,
              device, proj_matrix, baseline_ce):
    """Run one phase of the exploit cartography.

    Returns:
        result dict with pre/post fingerprints and exploit analysis
    """
    objective_fns = {
        "_raw_area": _raw_area,
        "_norm_normalized_area": _norm_normalized_area,
        "_arc_normalized_area": _arc_normalized_area,
    }

    obj_fn = objective_fns[phase_config["objective_fn"]]
    lambda_omega = phase_config["lambda"]

    print(f"\n{'=' * 60}")
    print(phase_config["name"])
    print(f"{'=' * 60}")
    print(phase_config["description"])
    print(f"λ = {lambda_omega}, steps = {STEPS_PER_PHASE}")

    # Pre-phase fingerprint
    model.eval()
    pre_fp = measure_fingerprint(model, tokenizer, eval_texts, device, proj_matrix)
    print(f"\n  Pre:  L_CE={pre_fp['l_ce']:.4f}  "
          f"||h||={pre_fp['mean_activation_norm']:.2f}  "
          f"raw_A={pre_fp['mean_raw_area']:.4f}  "
          f"norm_A={pre_fp['mean_norm_area']:.6f}  "
          f"arc_A={pre_fp['mean_arc_area']:.6f}")

    # Training
    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()

    losses_ce = []
    losses_obj = []
    aborted = False
    step = 0

    indices = np.random.permutation(len(texts))
    for i in range(0, len(indices), BATCH_SIZE):
        if step >= STEPS_PER_PHASE:
            break

        batch_idx = indices[i:i + BATCH_SIZE]
        batch = [texts[j] for j in batch_idx]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)

        out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
        l_ce = out.loss
        h_mid = out.hidden_states[MID_LAYER]
        l_obj = obj_fn(h_mid, proj_matrix).mean()

        l_total = l_ce - lambda_omega * l_obj

        optimizer.zero_grad()
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses_ce.append(l_ce.item())
        losses_obj.append(l_obj.item())
        step += 1

        if step % 50 == 0:
            avg_ce = np.mean(losses_ce[-50:])
            avg_obj = np.mean(losses_obj[-50:])
            ce_mult = avg_ce / baseline_ce if baseline_ce > 0 else 0
            status = ""
            if ce_mult > ABORT_CE_MULT:
                status = " *** PHASE ABORTED (L_CE explosion) ***"
                aborted = True
            print(f"    Step {step}/{STEPS_PER_PHASE}  "
                  f"L_CE={avg_ce:.4f} ({ce_mult:.2f}x baseline)  "
                  f"L_obj={avg_obj:.6f}{status}")
            if aborted:
                break

    # Post-phase fingerprint
    model.eval()
    post_fp = measure_fingerprint(model, tokenizer, eval_texts, device, proj_matrix)
    print(f"\n  Post: L_CE={post_fp['l_ce']:.4f}  "
          f"||h||={post_fp['mean_activation_norm']:.2f}  "
          f"raw_A={post_fp['mean_raw_area']:.4f}  "
          f"norm_A={post_fp['mean_norm_area']:.6f}  "
          f"arc_A={post_fp['mean_arc_area']:.6f}")

    # Exploit detection
    norm_change = (post_fp["mean_activation_norm"] - pre_fp["mean_activation_norm"]) / pre_fp["mean_activation_norm"]
    raw_area_change = (post_fp["mean_raw_area"] - pre_fp["mean_raw_area"]) / max(pre_fp["mean_raw_area"], 1e-10)
    norm_area_change = (post_fp["mean_norm_area"] - pre_fp["mean_norm_area"]) / max(pre_fp["mean_norm_area"], 1e-10)
    arc_area_change = (post_fp["mean_arc_area"] - pre_fp["mean_arc_area"]) / max(pre_fp["mean_arc_area"], 1e-10)
    ce_change = (post_fp["l_ce"] - pre_fp["l_ce"]) / pre_fp["l_ce"]

    # Phase profile shift (per-layer)
    profile_deltas = [post - pre for post, pre in
                      zip(post_fp["phase_profile"], pre_fp["phase_profile"])]
    middle_deltas = profile_deltas[1:-1]
    mean_middle_delta = float(np.mean(middle_deltas)) if middle_deltas else 0.0

    # Classify the exploit
    exploit_type = "unknown"
    exploit_evidence = []

    if norm_change > 0.10:
        exploit_evidence.append(f"activation norms grew {norm_change:+.1%}")
    if raw_area_change > 0.10 and norm_area_change < 0.05:
        exploit_evidence.append(f"raw area grew {raw_area_change:+.1%} but normalized area didn't — scale exploit")
        exploit_type = "activation_inflation"
    if norm_area_change > 0.10 and arc_area_change < 0.05:
        exploit_evidence.append(f"norm-area grew {norm_area_change:+.1%} but arc-area didn't — directional exploit")
        exploit_type = "anisotropic_stretch"
    if arc_area_change > 0.05:
        exploit_evidence.append(f"arc-normalized area grew {arc_area_change:+.1%} — genuine angular restructuring")
        exploit_type = "angular_restructuring"
    if abs(mean_middle_delta) > 0.01:
        exploit_evidence.append(f"middle-layer phase shifted {mean_middle_delta:+.4f} rad")
    if aborted:
        exploit_evidence.append(f"phase aborted: L_CE exploded ({ce_change:+.1%})")
        if exploit_type == "unknown":
            exploit_type = "catastrophic_exploit"

    if not exploit_evidence:
        exploit_evidence.append("no significant geometric change detected")
        exploit_type = "null"

    print(f"\n  Exploit classification: {exploit_type}")
    for ev in exploit_evidence:
        print(f"    - {ev}")

    # Curvature profile comparison
    print(f"\n  Phase profile shift (post - pre):")
    for i, d in enumerate(profile_deltas):
        bar = "+" * int(abs(d) * 100) if d > 0 else "-" * int(abs(d) * 100)
        print(f"    L{i}→L{i+1}: {d:+.4f} rad  {bar}")

    return {
        "phase": phase_config["short"],
        "name": phase_config["name"],
        "objective": phase_config["objective_fn"],
        "lambda": lambda_omega,
        "steps_completed": step,
        "aborted": aborted,
        "pre_fingerprint": pre_fp,
        "post_fingerprint": post_fp,
        "deltas": {
            "activation_norm_frac": float(norm_change),
            "raw_area_frac": float(raw_area_change),
            "norm_area_frac": float(norm_area_change),
            "arc_area_frac": float(arc_area_change),
            "l_ce_frac": float(ce_change),
            "phase_profile_deltas": profile_deltas,
            "mean_middle_phase_delta": float(mean_middle_delta),
        },
        "exploit": {
            "type": exploit_type,
            "expected": phase_config["expected_exploit"],
            "matched_expectation": exploit_type == phase_config["expected_exploit"],
            "evidence": exploit_evidence,
        },
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_wikitext(tokenizer, max_samples=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 50]
    return texts[:max_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT C: Exploit Cartography")
    print("The model as its own geometric instrument")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Phases: {len(PHASE_CONFIGS)}, steps per phase: {STEPS_PER_PHASE}")

    # Check precondition
    if not BASELINE_FILE.exists():
        print(f"ERROR: {BASELINE_FILE} not found. Run Experiment A first.")
        sys.exit(2)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    if baseline.get("verdict") != "PASS":
        print("ERROR: Experiment A did not pass. Cannot proceed.")
        sys.exit(2)

    baseline_curvatures = baseline["phase_profile"]["curvatures_rad"]
    print(f"Baseline phase profile loaded ({len(baseline_curvatures)} transitions).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # Fixed random projection (same seed as Experiment B for comparability)
    torch.manual_seed(42)
    proj_matrix = torch.randn(model.config.n_embd, 2, device=device)
    proj_matrix = torch.nn.functional.normalize(proj_matrix, dim=0)
    proj_matrix[:, 1] = proj_matrix[:, 1] - (proj_matrix[:, 0] @ proj_matrix[:, 1]) * proj_matrix[:, 0]
    proj_matrix[:, 1] = proj_matrix[:, 1] / proj_matrix[:, 1].norm()

    # Load data
    texts = load_wikitext(tokenizer)
    eval_texts = texts[:200]
    print(f"Loaded {len(texts)} training samples, {len(eval_texts)} eval samples.")

    # Baseline L_CE for abort thresholds
    model.eval()
    ce_losses = []
    for i in range(0, min(200, len(eval_texts)), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            ce_losses.append(out.loss.item())
    baseline_ce = float(np.mean(ce_losses))
    print(f"Baseline L_CE: {baseline_ce:.4f}")

    # Run phases sequentially — model state carries forward
    phase_results = []
    for phase_config in PHASE_CONFIGS:
        result = run_phase(
            phase_config, model, tokenizer, texts, eval_texts,
            device, proj_matrix, baseline_ce
        )
        phase_results.append(result)

    # ---------------------------------------------------------------------------
    # Synthesis: what did we learn?
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPLOIT CARTOGRAPHY — SYNTHESIS")
    print("=" * 60)

    exploits_found = []
    for r in phase_results:
        exploit = r["exploit"]
        matched = "✓" if exploit["matched_expectation"] else "✗"
        exploits_found.append(exploit["type"])
        print(f"\n  {r['name']}")
        print(f"    Expected: {exploit['expected']}")
        print(f"    Found:    {exploit['type']} [{matched}]")
        for ev in exploit["evidence"]:
            print(f"      {ev}")

    # The experiment "passes" if:
    # 1. At least 2 distinct exploit types were found (the model found different
    #    strategies as we closed each loophole)
    # 2. The phase profile changed meaningfully in at least one phase
    distinct_exploits = len(set(exploits_found) - {"null", "unknown"})
    any_phase_shift = any(
        abs(r["deltas"]["mean_middle_phase_delta"]) > 0.005
        for r in phase_results
    )

    cartography_informative = distinct_exploits >= 2
    geometry_responsive = any_phase_shift

    # But this experiment isn't really pass/fail. It's cartographic.
    # The output is the map itself.
    verdict = "INFORMATIVE" if cartography_informative else "FLAT"

    print(f"\n{'=' * 60}")
    print(f"  Distinct exploit types found: {distinct_exploits}")
    print(f"  Phase profile responded:      {'YES' if geometry_responsive else 'NO'}")
    print(f"  Cartography:                  {verdict}")
    print(f"{'=' * 60}")

    if cartography_informative:
        print("The model found different strategies as each exploit was closed.")
        print("The sequence of exploits maps the affordance structure of GPT-2's")
        print("representational geometry. Each exploit is a true fact about the space.")
    else:
        print("The model did not find distinct strategies across phases.")
        print("Either the objectives are too similar, λ is too small, or the")
        print("geometry is too rigid to reveal structure through optimization.")
        print("This is still informative — a flat cartography IS a result.")

    print("=" * 60)

    # Save results
    results = {
        "experiment": "C",
        "version": "v1_exploit_cartography",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "device": device,
        "steps_per_phase": STEPS_PER_PHASE,
        "num_phases": len(PHASE_CONFIGS),
        "baseline_l_ce": baseline_ce,
        "baseline_phase_profile": baseline_curvatures,
        "phases": phase_results,
        "synthesis": {
            "distinct_exploit_types": distinct_exploits,
            "exploits_found": exploits_found,
            "geometry_responsive": geometry_responsive,
            "verdict": verdict,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
