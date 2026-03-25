"""Experiment C v2: Phase 3 Lambda Sweep + Improved Exploit Classifier.

Run from the gpt2_calibration/ folder:
    python experiment_C/run_C_v2.py

Requires: Experiment C v1 must have been run (needs Phase 2 end-state).
Output:   ../results/experiment_C_v2_result.json

Motivation (2026-03-22):
    Experiment C v1 found three acts: catastrophe (Phase 1), recovery past
    baseline (Phase 2), and silence (Phase 3). The Phase 3 null result has
    two possible explanations:
      1. The model is genuinely rigid under arc-length-normalized angular
         perturbation -- Phase 2 consumed all available degrees of freedom.
      2. lambda=0.1 was too weak to overcome inertia on a settled model.

    This experiment distinguishes them by sweeping lambda over [0.5, 1.0, 2.0]
    on Phase 3 only, starting from the Phase 2 end-state checkpoint.

    If the model stays flat across all lambdas: rigidity is real.
    If it moves at some threshold: we've found the activation energy for
    angular restructuring, and that number calibrates Nemotron objectives.

    Additionally, the v1 exploit classifier is replaced with a causal
    classifier that reads the chain of metric changes rather than just
    checking which single metric moved.
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
OUTPUT_FILE = RESULTS_DIR / "experiment_C_v2_result.json"
ABORT_CE_MULT = 3.0

# Phase 3 lambda sweep values
LAMBDA_SWEEP = [0.5, 1.0, 2.0]

# ---------------------------------------------------------------------------
# Pancharatnam phase (shared with run_C.py)
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
# Area computations (from run_C.py)
# ---------------------------------------------------------------------------
def _shoelace_2d(proj: torch.Tensor) -> torch.Tensor:
    """Signed shoelace area from 2D projections. [batch, seq, 2] -> [batch]."""
    x, y = proj[:, :, 0], proj[:, :, 1]
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    return 0.5 * (x * y_next - x_next * y).sum(dim=1).abs()


def _project_centered(h: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
    """Center per-sequence and project to 2D."""
    h_centered = h - h.mean(dim=1, keepdim=True)
    return h_centered @ proj_matrix


def _raw_area(h, proj_matrix):
    proj = _project_centered(h.float(), proj_matrix)
    return _shoelace_2d(proj)


def _norm_normalized_area(h, proj_matrix):
    h_f = h.float()
    proj = _project_centered(h_f, proj_matrix)
    areas = _shoelace_2d(proj)
    mean_norm_sq = (h_f.norm(dim=-1) ** 2).mean(dim=1).clamp(min=1e-10)
    return areas / mean_norm_sq


def _arc_normalized_area(h, proj_matrix):
    """Phase 3 objective: area / arc_length^2."""
    h_f = h.float()
    proj = _project_centered(h_f, proj_matrix)
    areas = _shoelace_2d(proj)
    h_centered = h_f - h_f.mean(dim=1, keepdim=True)
    diffs = h_centered[:, 1:, :] - h_centered[:, :-1, :]
    segment_lengths = diffs.norm(dim=-1)
    arc_length = segment_lengths.sum(dim=1)
    arc_length_sq = (arc_length ** 2).clamp(min=1e-10)
    return areas / arc_length_sq


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def measure_fingerprint(model, tokenizer, texts, device, proj_matrix):
    """Capture a geometric fingerprint: phase profile + activation stats + area."""
    profile = measure_phase_profile(model, tokenizer, texts, device)
    norms, raw_areas, norm_areas, arc_areas = [], [], [], []
    for i in range(0, min(len(texts), 64), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h_mid = out.hidden_states[MID_LAYER].float()
        batch_norms = h_mid.norm(dim=-1).mean(dim=1)
        norms.extend(batch_norms.cpu().tolist())
        raw_areas.extend(_raw_area(h_mid, proj_matrix).cpu().tolist())
        norm_areas.extend(_norm_normalized_area(h_mid, proj_matrix).cpu().tolist())
        arc_areas.extend(_arc_normalized_area(h_mid, proj_matrix).cpu().tolist())
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
# Improved exploit classifier (v2) — causal chain detection
# ---------------------------------------------------------------------------
def classify_exploit_v2(pre_fp, post_fp, aborted):
    """Causal exploit classifier that reads the chain of metric changes.

    v1 bug: classified Phase 1 as 'angular_restructuring' because phase
    profile shifted, even though the ROOT CAUSE was activation inflation
    so massive it leaked through all normalizations.

    v2 logic: check metrics in causal order.
      1. Did norms change? If yes, that's the primary cause.
      2. Given the norm change, did normalized area change beyond what
         norm change explains? If yes, there's also a directional effect.
      3. Given both, did arc-normalized area change? If yes, angular.
      4. Phase profile shifts are SECONDARY evidence, not primary.
    """
    norm_change = (post_fp["mean_activation_norm"] - pre_fp["mean_activation_norm"]) / pre_fp["mean_activation_norm"]
    raw_area_change = (post_fp["mean_raw_area"] - pre_fp["mean_raw_area"]) / max(pre_fp["mean_raw_area"], 1e-10)
    norm_area_change = (post_fp["mean_norm_area"] - pre_fp["mean_norm_area"]) / max(pre_fp["mean_norm_area"], 1e-10)
    arc_area_change = (post_fp["mean_arc_area"] - pre_fp["mean_arc_area"]) / max(pre_fp["mean_arc_area"], 1e-10)
    ce_change = (post_fp["l_ce"] - pre_fp["l_ce"]) / pre_fp["l_ce"]

    profile_deltas = [post - pre for post, pre in
                      zip(post_fp["phase_profile"], pre_fp["phase_profile"])]
    early_deltas = profile_deltas[:MID_LAYER]  # L0->L5
    late_deltas = profile_deltas[MID_LAYER:]   # L6->L11
    mean_early = float(np.mean(early_deltas)) if early_deltas else 0.0
    mean_late = float(np.mean(late_deltas)) if late_deltas else 0.0
    band_divergence = abs(mean_early - mean_late)

    # Build causal chain
    causes = []  # ordered list of (cause, evidence_str)
    exploit_type = "null"

    # Level 1: norm inflation?
    if abs(norm_change) > 0.10:
        direction = "inflation" if norm_change > 0 else "deflation"
        causes.append((f"activation_{direction}",
                       f"activation norms changed {norm_change:+.1%}"))
        exploit_type = f"activation_{direction}"

    # Level 2: directional distortion beyond what norms explain?
    if abs(norm_area_change) > 0.05:
        causes.append(("anisotropic_distortion",
                       f"norm-normalized area changed {norm_area_change:+.1%}"))
        if exploit_type == "null":
            exploit_type = "anisotropic_distortion"
        else:
            exploit_type += "+anisotropic_distortion"

    # Level 3: genuine angular restructuring?
    if abs(arc_area_change) > 0.03:
        causes.append(("angular_restructuring",
                       f"arc-normalized area changed {arc_area_change:+.1%}"))
        if exploit_type == "null":
            exploit_type = "angular_restructuring"
        else:
            exploit_type += "+angular"

    # Secondary: band structure in phase profile?
    if band_divergence > 0.02:
        causes.append(("band_structure",
                       f"early/late phase divergence {band_divergence:.4f} rad "
                       f"(early {mean_early:+.4f}, late {mean_late:+.4f})"))

    # CE catastrophe?
    if aborted or ce_change > 1.0:
        causes.append(("ce_catastrophe",
                       f"L_CE changed {ce_change:+.1%}"))

    if not causes:
        causes.append(("null", "no significant geometric change detected"))

    evidence = [c[1] for c in causes]
    cause_chain = [c[0] for c in causes]

    return {
        "type": exploit_type,
        "cause_chain": cause_chain,
        "evidence": evidence,
        "deltas": {
            "activation_norm_frac": float(norm_change),
            "raw_area_frac": float(raw_area_change),
            "norm_area_frac": float(norm_area_change),
            "arc_area_frac": float(arc_area_change),
            "l_ce_frac": float(ce_change),
            "phase_profile_deltas": profile_deltas,
            "mean_early_delta": float(mean_early),
            "mean_late_delta": float(mean_late),
            "band_divergence": float(band_divergence),
        },
    }


# ---------------------------------------------------------------------------
# Phase runner (training loop for one phase)
# ---------------------------------------------------------------------------
def run_training_phase(model, tokenizer, texts, eval_texts, device,
                      proj_matrix, obj_fn, lambda_omega, baseline_ce,
                      phase_name, steps=STEPS_PER_PHASE):
    """Run one training phase. Returns (pre_fp, post_fp, aborted, step)."""
    print(f"\n{'=' * 60}")
    print(f"  {phase_name}")
    print(f"{'=' * 60}")
    print(f"  lambda={lambda_omega}, steps={steps}")

    model.eval()
    pre_fp = measure_fingerprint(model, tokenizer, eval_texts, device, proj_matrix)
    print(f"  Pre:  L_CE={pre_fp['l_ce']:.4f}  "
          f"||h||={pre_fp['mean_activation_norm']:.2f}  "
          f"arc_A={pre_fp['mean_arc_area']:.6f}")

    optimizer = AdamW(model.parameters(), lr=LR)
    model.train()
    losses_ce, losses_obj = [], []
    aborted = False
    step = 0
    indices = np.random.permutation(len(texts))

    for i in range(0, len(indices), BATCH_SIZE):
        if step >= steps:
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
            ce_mult = avg_ce / baseline_ce if baseline_ce > 0 else 0
            status = ""
            if ce_mult > ABORT_CE_MULT:
                status = " *** ABORTED ***"
                aborted = True
            print(f"    Step {step}/{steps}  "
                  f"L_CE={avg_ce:.4f} ({ce_mult:.2f}x)  "
                  f"L_obj={np.mean(losses_obj[-50:]):.6f}{status}")
            if aborted:
                break

    model.eval()
    post_fp = measure_fingerprint(model, tokenizer, eval_texts, device, proj_matrix)
    print(f"  Post: L_CE={post_fp['l_ce']:.4f}  "
          f"||h||={post_fp['mean_activation_norm']:.2f}  "
          f"arc_A={post_fp['mean_arc_area']:.6f}")

    return pre_fp, post_fp, aborted, step


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
    print("EXPERIMENT C v2: Phase 3 Lambda Sweep")
    print("Distinguishing rigidity from insufficient pressure")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Lambda sweep: {LAMBDA_SWEEP}")

    # Check precondition
    if not BASELINE_FILE.exists():
        print(f"ERROR: {BASELINE_FILE} not found. Run Experiment A first.")
        sys.exit(2)
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    if baseline.get("verdict") != "PASS":
        print("ERROR: Experiment A did not pass.")
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

    # Fixed projection (same seed as v1)
    torch.manual_seed(42)
    proj_matrix = torch.randn(model.config.n_embd, 2, device=device)
    proj_matrix = torch.nn.functional.normalize(proj_matrix, dim=0)
    proj_matrix[:, 1] = proj_matrix[:, 1] - (proj_matrix[:, 0] @ proj_matrix[:, 1]) * proj_matrix[:, 0]
    proj_matrix[:, 1] = proj_matrix[:, 1] / proj_matrix[:, 1].norm()

    # Load data
    texts = load_wikitext(tokenizer)
    eval_texts = texts[:200]
    print(f"Loaded {len(texts)} training samples, {len(eval_texts)} eval samples.")

    # Baseline L_CE
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

    # -----------------------------------------------------------------------
    # STAGE 1: Replay Phases 1 and 2 to reach the Phase 2 end-state
    # (We replay rather than checkpoint because v1 didn't save checkpoints)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("STAGE 1: Replaying Phases 1 & 2 to reach Phase 2 end-state")
    print("#" * 60)

    # Phase 1: Raw area (catastrophe)
    np.random.seed(0)  # Deterministic replay
    torch.manual_seed(0)
    pre1, post1, abort1, steps1 = run_training_phase(
        model, tokenizer, texts, eval_texts, device, proj_matrix,
        _raw_area, 0.001, baseline_ce,
        "Phase 1 (replay): Raw Area")
    exploit1 = classify_exploit_v2(pre1, post1, abort1)
    print(f"  Exploit (v2 classifier): {exploit1['type']}")
    for ev in exploit1["evidence"]:
        print(f"    - {ev}")

    # Phase 2: Norm-normalized area (recovery)
    pre2, post2, abort2, steps2 = run_training_phase(
        model, tokenizer, texts, eval_texts, device, proj_matrix,
        _norm_normalized_area, 0.01, baseline_ce,
        "Phase 2 (replay): Norm-Normalized Area")
    exploit2 = classify_exploit_v2(pre2, post2, abort2)
    print(f"  Exploit (v2 classifier): {exploit2['type']}")
    for ev in exploit2["evidence"]:
        print(f"    - {ev}")

    # Save Phase 2 end-state as checkpoint for the sweep
    phase2_state = deepcopy(model.state_dict())
    phase2_fp = post2
    print("\n  Phase 2 end-state saved as checkpoint.")

    # -----------------------------------------------------------------------
    # STAGE 2: Lambda sweep on Phase 3
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("STAGE 2: Phase 3 Lambda Sweep")
    print(f"Lambdas: {LAMBDA_SWEEP}")
    print("#" * 60)

    sweep_results = []
    for lam in LAMBDA_SWEEP:
        # Restore Phase 2 end-state
        model.load_state_dict(deepcopy(phase2_state))
        print(f"\n  [Restored Phase 2 checkpoint for lambda={lam}]")

        pre3, post3, abort3, steps3 = run_training_phase(
            model, tokenizer, texts, eval_texts, device, proj_matrix,
            _arc_normalized_area, lam, baseline_ce,
            f"Phase 3 (lambda={lam}): Arc-Length-Normalized Area")
        exploit3 = classify_exploit_v2(pre3, post3, abort3)

        print(f"  Exploit (v2 classifier): {exploit3['type']}")
        for ev in exploit3["evidence"]:
            print(f"    - {ev}")

        # Phase profile comparison
        profile_deltas = exploit3["deltas"]["phase_profile_deltas"]
        print(f"  Phase profile shift:")
        for k, d in enumerate(profile_deltas):
            bar = "+" * int(abs(d) * 100) if d > 0 else "-" * int(abs(d) * 100)
            print(f"    L{k}->L{k+1}: {d:+.4f} rad {bar}")

        sweep_results.append({
            "lambda": lam,
            "steps_completed": steps3,
            "aborted": abort3,
            "pre_fingerprint": pre3,
            "post_fingerprint": post3,
            "exploit": exploit3,
        })

    # -----------------------------------------------------------------------
    # SYNTHESIS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYNTHESIS: Rigidity vs Insufficient Pressure")
    print("=" * 60)

    any_moved = False
    activation_energy_lambda = None
    for sr in sweep_results:
        lam = sr["lambda"]
        etype = sr["exploit"]["type"]
        moved = etype != "null"
        marker = "MOVED" if moved else "FLAT"
        print(f"  lambda={lam}: {marker} ({etype})")
        if moved and not any_moved:
            any_moved = True
            activation_energy_lambda = lam

    if not any_moved:
        verdict = "RIGID"
        print(f"\n  VERDICT: {verdict}")
        print("  The model is genuinely rigid under arc-length-normalized")
        print("  angular perturbation after Phase 2 settlement.")
        print("  Phase 2 consumed all available geometric degrees of freedom.")
        print("  Implication for Nemotron: arc-length objective may need to be")
        print("  applied DURING training, not after settlement.")
    else:
        verdict = "ACTIVATION_ENERGY_FOUND"
        print(f"\n  VERDICT: {verdict}")
        print(f"  The model moved at lambda={activation_energy_lambda}.")
        print(f"  This is the activation energy for angular restructuring")
        print(f"  on a settled GPT-2. Use this to calibrate Nemotron objectives.")

    print("=" * 60)

    # Save results
    results = {
        "experiment": "C_v2",
        "version": "v2_lambda_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "device": device,
        "steps_per_phase": STEPS_PER_PHASE,
        "lambda_sweep": LAMBDA_SWEEP,
        "baseline_l_ce": baseline_ce,
        "baseline_phase_profile": baseline_curvatures,
        "replay": {
            "phase_1": {
                "pre_fingerprint": pre1,
                "post_fingerprint": post1,
                "exploit": exploit1,
                "steps": steps1,
                "aborted": abort1,
            },
            "phase_2": {
                "pre_fingerprint": pre2,
                "post_fingerprint": post2,
                "exploit": exploit2,
                "steps": steps2,
                "aborted": abort2,
            },
        },
        "phase_2_checkpoint_fingerprint": phase2_fp,
        "sweep_results": sweep_results,
        "synthesis": {
            "verdict": verdict,
            "any_moved": any_moved,
            "activation_energy_lambda": activation_energy_lambda,
            "lambdas_tested": LAMBDA_SWEEP,
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
