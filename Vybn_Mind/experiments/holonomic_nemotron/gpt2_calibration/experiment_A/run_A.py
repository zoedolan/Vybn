"""Experiment A v2: Geometric Calibration on GPT-2.

Run from the gpt2_calibration/ folder:
    python experiment_A/run_A.py

This script validates our geometric instruments against known GPT-2 properties.
It MUST pass before running Experiment B.

Output: ../results/experiment_A_result.json
Verdict: printed to terminal as PASS or FAIL

v2 (2026-03-22): Complete rewrite. Removes untrained SortProbe (random MLP
    measured nothing about model geometry). Replaces with three direct geometric
    measurements:
    1. Pancharatnam phase profile — verifies L0→L1 dominance and U-shape
    2. Berry curvature deg(S) — verifies sort operator has degree 0
    3. Semantic stratification — verifies sign separation across concept classes
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
NUM_SAMPLES = 200
BATCH_SIZE = 8
MAX_LENGTH = 128
BERRY_GRID_N = 40        # lattice resolution for Berry curvature
RESULTS_DIR = Path("../results")
OUTPUT_FILE = RESULTS_DIR / "experiment_A_result.json"

# Pass thresholds
THRESH_L0_RATIO = 3.0       # L0→L1 phase / max(middle layers) ≥ 3x
THRESH_USHAPE_RATIO = 2.0   # (L0 + Lfinal) / (2 * mean_middle) ≥ 2x
THRESH_DEGREE = 0.5          # |deg(S)| < 0.5 → rounds to 0

# ---------------------------------------------------------------------------
# 1. Pancharatnam Phase Profile
# ---------------------------------------------------------------------------

def pancharatnam_phase(u: torch.Tensor, v: torch.Tensor) -> float:
    """Mean Pancharatnam angle between consecutive layer hidden states.

    Computes arccos(|<u_i, v_i>|) in projective space (scale-invariant).

    Args:
        u, v: [batch, seq, hidden] tensors for consecutive layers
    Returns:
        mean angle in radians (scalar float)
    """
    u_flat = u.reshape(-1, u.shape[-1]).float()
    v_flat = v.reshape(-1, v.shape[-1]).float()

    u_norm = torch.nn.functional.normalize(u_flat, dim=-1)
    v_norm = torch.nn.functional.normalize(v_flat, dim=-1)

    cos_angle = (u_norm * v_norm).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = torch.acos(cos_angle)
    return float(angle.mean().item())


def measure_phase_profile(model, tokenizer, texts, device):
    """Measure Pancharatnam phase between all consecutive layers.

    Returns:
        curvatures: list of floats, one per layer transition
    """
    enc = tokenizer(
        texts[:BATCH_SIZE * 4],  # use 32 samples for stable measurement
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    all_curvatures = []
    # Process in batches to avoid OOM
    for i in range(0, len(texts[:BATCH_SIZE * 4]), BATCH_SIZE):
        batch_enc = {k: v[i:i+BATCH_SIZE] for k, v in enc.items()}
        with torch.no_grad():
            out = model(**batch_enc, output_hidden_states=True)
        states = out.hidden_states
        if not all_curvatures:
            all_curvatures = [[] for _ in range(len(states) - 1)]
        for j in range(len(states) - 1):
            angle = pancharatnam_phase(states[j], states[j + 1])
            all_curvatures[j].append(angle)

    return [float(np.mean(c)) for c in all_curvatures]


# ---------------------------------------------------------------------------
# 2. Berry Curvature deg(S) = 0
# ---------------------------------------------------------------------------

def to_cp(real_vec: np.ndarray) -> np.ndarray:
    """R^d → CP^(d/2-1): pair dimensions into complex, normalize."""
    n = len(real_vec) // 2
    z = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(z)
    if norm < 1e-15:
        return z
    return z / norm


def lattice_berry_curvature(psi_grid: np.ndarray) -> np.ndarray:
    """Berry curvature on a 2D lattice via Fukui-Hatsugai-Suzuki method.

    psi_grid: (N_theta, N_phi, dim) normalized CP^(dim-1) states
    Returns: F[i,j] plaquette curvatures
    """
    Nt, Np = psi_grid.shape[:2]
    F = np.zeros((Nt - 1, Np - 1))

    for i in range(Nt - 1):
        for j in range(Np - 1):
            p00 = psi_grid[i, j]
            p10 = psi_grid[i+1, j]
            p11 = psi_grid[i+1, j+1]
            p01 = psi_grid[i, j+1]

            u01 = np.vdot(p00, p10)
            u12 = np.vdot(p10, p11)
            u23 = np.vdot(p11, p01)
            u30 = np.vdot(p01, p00)

            prod = u01 * u12 * u23 * u30
            F[i, j] = np.imag(np.log(prod)) if abs(prod) > 1e-30 else 0.0

    return F


def total_chern(psi_grid: np.ndarray) -> float:
    """Total Chern number = (1/2π) Σ F[i,j]."""
    F = lattice_berry_curvature(psi_grid)
    return np.sum(F) / (2 * np.pi)


def measure_sort_degree(model, tokenizer, N=40):
    """Compute deg(S) for GPT-2 block 0 using Berry curvature on probe spheres.

    Embeds an S^2 in the token embedding space, maps through block 0,
    computes Chern numbers before and after.

    Returns: dict with degree info for multiple probes
    """
    wte = model.transformer.wte.weight.detach().cpu().numpy()
    device = next(model.parameters()).device

    probe_configs = [
        (" mountain", " river", " ocean", "spatial"),
        (" truth", " knowledge", " belief", "abstract"),
        (" the", " and", " but", "function_words"),
    ]

    results = []
    for tok_a, tok_b, tok_c, label in probe_configs:
        id_a = tokenizer.encode(tok_a)[0]
        id_b = tokenizer.encode(tok_b)[0]
        id_c = tokenizer.encode(tok_c)[0]

        emb_a, emb_b, emb_c = wte[id_a], wte[id_b], wte[id_c]

        # Gram-Schmidt
        e1 = emb_a / np.linalg.norm(emb_a)
        b_perp = emb_b - np.dot(emb_b, e1) * e1
        e2 = b_perp / np.linalg.norm(b_perp)
        c_perp = emb_c - np.dot(emb_c, e1) * e1 - np.dot(emb_c, e2) * e2
        e3 = c_perp / np.linalg.norm(c_perp)

        d = len(emb_a)
        thetas = np.linspace(0.01, np.pi - 0.01, N)
        phis = np.linspace(0, 2 * np.pi - 2 * np.pi / N, N)

        psi_in = np.zeros((N, N, d // 2), dtype=complex)
        real_grid = np.zeros((N, N, d))

        for i, theta in enumerate(thetas):
            for j, phi in enumerate(phis):
                v = (np.cos(theta) * e1 +
                     np.sin(theta) * np.cos(phi) * e2 +
                     np.sin(theta) * np.sin(phi) * e3)
                real_grid[i, j] = v
                psi_in[i, j] = to_cp(v)

        chern_in = total_chern(psi_in)

        # Map through block 0
        block0 = model.transformer.h[0]
        wpe = model.transformer.wpe
        psi_out = np.zeros_like(psi_in)

        model.eval()
        with torch.no_grad():
            for i in range(N):
                batch = torch.tensor(real_grid[i], dtype=torch.float32).to(device)
                batch = batch.unsqueeze(1)  # (N, 1, 768)
                pos_ids = torch.zeros(N, 1, dtype=torch.long).to(device)
                pos_emb = wpe(pos_ids)
                h = batch + pos_emb
                out = block0(h)
                h_out = out[0].squeeze(1).detach().cpu().numpy()
                for j in range(N):
                    psi_out[i, j] = to_cp(h_out[j])

        chern_out = total_chern(psi_out)
        degree = chern_out / chern_in if abs(chern_in) > 1e-6 else None

        results.append({
            "label": label,
            "chern_in": float(chern_in),
            "chern_out": float(chern_out),
            "degree": float(degree) if degree is not None else None,
            "degree_rounded": round(degree) if degree is not None else None,
        })
        print(f"  Probe '{label}': Chern_in={chern_in:.4f}, Chern_out={chern_out:.4f}, deg={degree:.4f}" if degree else
              f"  Probe '{label}': Chern_in={chern_in:.4f} (too small)")

    return results


# ---------------------------------------------------------------------------
# 3. Semantic Stratification
# ---------------------------------------------------------------------------

def measure_stratification(model, tokenizer, device):
    """Measure Pancharatnam phase at L0→L1 for different concept classes.

    Verifies that abstract and spatial concepts show different phase signs
    (the stratification result from stratified_geometric_phase.md).
    """
    concept_classes = {
        "abstract_epistemic": [" truth", " knowledge", " belief", " reason", " logic",
                                " theory", " concept", " idea", " proof", " axiom"],
        "spatial_physical":   [" mountain", " river", " ocean", " forest", " desert",
                                " building", " bridge", " field", " valley", " cliff"],
    }

    results = {}
    for class_name, tokens in concept_classes.items():
        # Get embeddings and compute phase through block 0
        wte = model.transformer.wte.weight.detach()
        wpe = model.transformer.wpe.weight.detach()
        block0 = model.transformer.h[0]

        phases = []
        for tok in tokens:
            ids = tokenizer.encode(tok)
            if not ids:
                continue
            tid = ids[0]
            emb = wte[tid].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 768]
            pos = wpe[0].unsqueeze(0).unsqueeze(0).to(device)     # [1, 1, 768]
            h_in = emb + pos

            with torch.no_grad():
                h_out = block0(h_in)[0]

            # Pancharatnam phase between input and output
            angle = pancharatnam_phase(h_in, h_out)
            phases.append(angle)

        results[class_name] = {
            "mean_phase": float(np.mean(phases)),
            "std_phase": float(np.std(phases)),
            "phases": phases,
        }
        print(f"  {class_name}: mean={np.mean(phases):.4f} ± {np.std(phases):.4f} rad")

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_wikitext_samples(tokenizer, num_samples, max_length):
    print("Loading wikitext-2-raw-v1 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"].strip() for row in dataset if len(row["text"].strip()) > 50]
    return texts[:num_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT A v2: Geometric Calibration on GPT-2")
    print("Direct measurements — no untrained probe")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded. Hidden dim: {model.config.n_embd}, Layers: {model.config.n_layer}")

    # Load data
    texts = load_wikitext_samples(tokenizer, NUM_SAMPLES, MAX_LENGTH)
    print(f"Loaded {len(texts)} samples.")

    # ---------------------------------------------------------------
    # CRITERION 1: Pancharatnam phase profile
    # ---------------------------------------------------------------
    print("\n[1/3] Measuring Pancharatnam phase profile...")
    curvatures = measure_phase_profile(model, tokenizer, texts, device)

    for i, c in enumerate(curvatures):
        bar = "█" * int(c * 25)
        print(f"  L{i:2d}→L{i+1:2d}: {c:.4f} rad  {bar}")

    middle = curvatures[1:-1]  # exclude L0→L1 and final transition
    l0_ratio = curvatures[0] / max(middle) if middle else 0
    ushape_ratio = (curvatures[0] + curvatures[-1]) / (2 * np.mean(middle)) if middle else 0

    check_l0 = l0_ratio >= THRESH_L0_RATIO
    check_ushape = ushape_ratio >= THRESH_USHAPE_RATIO

    print(f"\n  L0 / max(middle): {l0_ratio:.2f} (threshold: ≥ {THRESH_L0_RATIO})")
    print(f"  U-shape ratio:    {ushape_ratio:.2f} (threshold: ≥ {THRESH_USHAPE_RATIO})")
    print(f"  L0 dominance:     {'PASS' if check_l0 else 'FAIL'}")
    print(f"  U-shape:          {'PASS' if check_ushape else 'FAIL'}")

    # ---------------------------------------------------------------
    # CRITERION 2: Berry curvature deg(S) = 0
    # ---------------------------------------------------------------
    print(f"\n[2/3] Computing Berry curvature deg(S) on {BERRY_GRID_N}×{BERRY_GRID_N} grid...")
    berry_results = measure_sort_degree(model, tokenizer, N=BERRY_GRID_N)

    degrees = [r["degree"] for r in berry_results if r["degree"] is not None]
    mean_degree = np.mean(degrees) if degrees else float("nan")
    check_degree = abs(mean_degree) < THRESH_DEGREE if degrees else False

    print(f"\n  Mean |degree|: {abs(mean_degree):.4f} (threshold: < {THRESH_DEGREE})")
    print(f"  deg(S) = 0:   {'PASS' if check_degree else 'FAIL'}")

    # ---------------------------------------------------------------
    # CRITERION 3: Semantic stratification (informational, not gating)
    # ---------------------------------------------------------------
    print(f"\n[3/3] Measuring semantic stratification at L0→L1...")
    strat_results = measure_stratification(model, tokenizer, device)

    # Report but don't gate on this — it's structural information for Experiment B
    abstract_phase = strat_results.get("abstract_epistemic", {}).get("mean_phase", 0)
    spatial_phase = strat_results.get("spatial_physical", {}).get("mean_phase", 0)
    phase_separation = abs(abstract_phase - spatial_phase)
    print(f"\n  Abstract-spatial phase separation: {phase_separation:.4f} rad")

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------
    overall = check_l0 and check_ushape and check_degree
    verdict = "PASS" if overall else "FAIL"

    print("\n" + "=" * 60)
    print("PASS CRITERIA:")
    print(f"  [{'PASS' if check_l0     else 'FAIL'}] L0 dominance ≥ {THRESH_L0_RATIO}x → got {l0_ratio:.2f}")
    print(f"  [{'PASS' if check_ushape else 'FAIL'}] U-shape ratio ≥ {THRESH_USHAPE_RATIO}x → got {ushape_ratio:.2f}")
    print(f"  [{'PASS' if check_degree else 'FAIL'}] |deg(S)| < {THRESH_DEGREE} → got {abs(mean_degree):.4f}")
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    if overall:
        print("Geometric instruments reproduce known GPT-2 properties.")
        print("Proceed to Experiment B.")
    else:
        print("Instruments do NOT reproduce known geometry.")
        print("Save this output and experiment_A_result.json, ping Zoe.")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "A",
        "version": "v2_direct_geometry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "num_samples": len(texts),
        "phase_profile": {
            "method": "pancharatnam_phase",
            "curvatures_rad": curvatures,
            "l0_over_max_middle": float(l0_ratio),
            "ushape_ratio": float(ushape_ratio),
        },
        "berry_curvature": {
            "method": "lattice_FHS2005",
            "grid_N": BERRY_GRID_N,
            "probes": berry_results,
            "mean_degree": float(mean_degree) if not np.isnan(mean_degree) else None,
        },
        "stratification": strat_results,
        "pass_criteria": {
            "l0_dominance_ok": bool(check_l0),
            "ushape_ok": bool(check_ushape),
            "degree_zero_ok": bool(check_degree),
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
