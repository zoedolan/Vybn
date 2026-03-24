#!/usr/bin/env python3
"""
Quantum Permutation Test for the Existential Sign Anomaly

Uses IBM Quantum hardware to generate genuinely random permutations
of concept-class labels, then tests whether the "100% existential
sign agreement" survives random relabeling.

This is the permutation test that should have been done BEFORE
reporting the existential anomaly. Better late than never.

The quantum processor provides true randomness (not pseudorandom),
which matters for a clean null distribution. We want to be sure
the permutation test isn't contaminated by deterministic patterns.

Author: Vybn, March 24, 2026
Motivated by: Getting praised for a statistical artifact
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add repo to path
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

# ─── Quantum random number generation ────────────────────────────────

def get_quantum_random_bits(n_bits=80, n_shots=1024):
    """
    Get genuinely random bits from IBM Quantum hardware.
    
    Runs a Hadamard-on-all-qubits circuit, producing uniformly random
    bitstrings from quantum superposition collapse.
    
    Returns: list of n_shots bitstrings, each of length n_bits
    """
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    service = QiskitRuntimeService()
    backend = service.least_busy(min_num_qubits=n_bits, operational=True)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    
    # Simple circuit: H on n_bits qubits, then measure
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    
    # Transpile to ISA
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    print(f"Circuit depth after transpilation: {isa_circuit.depth()}")
    print(f"Submitting {n_shots} shots...")
    
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=n_shots)
    print(f"Job ID: {job.job_id()}")
    
    result = job.result()
    pub_result = result[0]
    
    # Extract bitstrings
    data = pub_result.data
    bitstrings = None
    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        attr = getattr(data, attr_name)
        if hasattr(attr, "get_bitstrings"):
            bitstrings = list(attr.get_bitstrings())
            break
    
    if bitstrings is None:
        raise RuntimeError("Could not extract bitstrings from quantum result")
    
    print(f"Got {len(bitstrings)} bitstrings of length {len(bitstrings[0])}")
    return bitstrings, backend.name, job.job_id()


def bitstring_to_permutation(bitstring, n_items):
    """
    Convert a random bitstring to a permutation of n_items
    using Fisher-Yates shuffle with bits as entropy source.
    
    Uses rejection sampling to avoid modulo bias.
    """
    bits = [int(b) for b in bitstring]
    bit_idx = 0
    
    def next_int(max_val):
        """Get a random integer in [0, max_val) from bit stream."""
        nonlocal bit_idx
        if max_val <= 1:
            return 0
        n_bits_needed = int(np.ceil(np.log2(max_val)))
        # Rejection sampling
        for _ in range(20):  # max attempts
            if bit_idx + n_bits_needed > len(bits):
                # Ran out of bits, fall back to what we have
                return 0
            val = 0
            for i in range(n_bits_needed):
                val = (val << 1) | bits[bit_idx]
                bit_idx += 1
            if val < max_val:
                return val
        return 0  # fallback
    
    perm = list(range(n_items))
    for i in range(n_items - 1, 0, -1):
        j = next_int(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm


# ─── Load the original experimental results ──────────────────────────

def load_original_results():
    """Load the cross-architecture sign invariance results."""
    results_path = Path(__file__).parent / "results" / "cross_architecture_sign_invariance.json"
    with open(results_path) as f:
        return json.load(f)


def compute_class_agreement(gpt2_sgp, pythia_sgp, class_names, prompts_per_class,
                           prompt_to_original_class, permutation=None):
    """
    Compute sign agreement by class, optionally with permuted labels.
    
    If permutation is provided, it maps prompt indices to new positions,
    effectively shuffling which prompts belong to which class.
    """
    n_prompts = len(prompt_to_original_class)
    
    if permutation is not None:
        # Permuted: reassign prompts to classes
        permuted_assignment = [prompt_to_original_class[permutation[i]] 
                              for i in range(n_prompts)]
    else:
        permuted_assignment = prompt_to_original_class
    
    # Group by (permuted) class
    class_agreements = {}
    for cls in class_names:
        # Get indices of prompts assigned to this class
        cls_indices = [i for i, c in enumerate(permuted_assignment) if c == cls]
        
        # For each layer pair, check sign agreement
        layer_pairs = ["L0->L3", "L0->L6", "L3->L9", "L6->L12"]
        agreements = 0
        total = 0
        
        for lp in layer_pairs:
            # Get the original class data for GPT-2 and Pythia
            # Under permutation, we need to recompute the mean phase
            # from the individual prompt phases
            gpt2_phases = []
            pythia_phases = []
            
            for idx in cls_indices:
                orig_cls = prompt_to_original_class[idx]
                prompt_idx_in_class = idx % prompts_per_class
                
                g_data = gpt2_sgp.get(orig_cls, {}).get(lp, {})
                p_data = pythia_sgp.get(orig_cls, {}).get(lp, {})
                
                if g_data and "phases" in g_data and prompt_idx_in_class < len(g_data["phases"]):
                    gpt2_phases.append(g_data["phases"][prompt_idx_in_class])
                if p_data and "phases" in p_data and prompt_idx_in_class < len(p_data["phases"]):
                    pythia_phases.append(p_data["phases"][prompt_idx_in_class])
            
            if gpt2_phases and pythia_phases:
                g_sign = int(np.sign(np.mean(gpt2_phases))) if abs(np.mean(gpt2_phases)) > 1e-10 else 0
                p_sign = int(np.sign(np.mean(pythia_phases))) if abs(np.mean(pythia_phases)) > 1e-10 else 0
                total += 1
                if g_sign == p_sign:
                    agreements += 1
        
        if total > 0:
            class_agreements[cls] = agreements / total
    
    return class_agreements


def main():
    print("=" * 60)
    print("QUANTUM PERMUTATION TEST")
    print("Testing whether existential 100% agreement survives")
    print("random relabeling of concept classes")
    print("=" * 60)
    
    # Load original results
    data = load_original_results()
    gpt2_sgp = data["gpt2_sgp"]
    pythia_sgp = data["pythia_sgp"]
    class_names = sorted(data["concept_classes"])
    prompts_per_class = 4
    n_prompts = len(class_names) * prompts_per_class  # 24
    
    # Create prompt-to-class mapping
    prompt_to_class = []
    for cls in class_names:
        prompt_to_class.extend([cls] * prompts_per_class)
    
    # Original (unpermuted) agreement rates
    print("\nOriginal agreement rates:")
    orig_agreements = compute_class_agreement(
        gpt2_sgp, pythia_sgp, class_names, prompts_per_class, prompt_to_class
    )
    for cls, rate in sorted(orig_agreements.items()):
        print(f"  {cls}: {rate:.0%}")
    
    existential_original = orig_agreements.get("existential", 0)
    print(f"\nExistential agreement: {existential_original:.0%}")
    
    # Get quantum random bits
    print("\n" + "=" * 60)
    print("Fetching quantum random bits from IBM hardware...")
    n_permutations = 500  # Use 500 permutations for null distribution
    
    try:
        bitstrings, backend_name, job_id = get_quantum_random_bits(
            n_bits=80, n_shots=n_permutations
        )
    except Exception as e:
        print(f"Quantum hardware failed: {e}")
        print("This experiment requires real quantum hardware.")
        return
    
    # Generate permutations from quantum bits
    print(f"\nGenerating {len(bitstrings)} quantum-random permutations...")
    
    null_existential = []
    null_max_class = []  # Track maximum agreement across ANY class
    
    for i, bs in enumerate(bitstrings):
        perm = bitstring_to_permutation(bs, n_prompts)
        agreements = compute_class_agreement(
            gpt2_sgp, pythia_sgp, class_names, prompts_per_class,
            prompt_to_class, permutation=perm
        )
        
        ex_rate = agreements.get("existential", 0)
        null_existential.append(ex_rate)
        
        max_rate = max(agreements.values()) if agreements else 0
        null_max_class.append(max_rate)
    
    null_existential = np.array(null_existential)
    null_max_class = np.array(null_max_class)
    
    # Compute p-values
    # How often does random relabeling produce existential agreement >= observed?
    p_existential = np.mean(null_existential >= existential_original)
    
    # How often does ANY class achieve >= existential under random relabeling?
    p_any_class = np.mean(null_max_class >= existential_original)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nNull distribution (existential class, {len(bitstrings)} quantum permutations):")
    print(f"  Mean: {np.mean(null_existential):.2%}")
    print(f"  Std:  {np.std(null_existential):.2%}")
    print(f"  Min:  {np.min(null_existential):.2%}")
    print(f"  Max:  {np.max(null_existential):.2%}")
    print(f"  Median: {np.median(null_existential):.2%}")
    
    print(f"\nObserved existential agreement: {existential_original:.0%}")
    print(f"p-value (existential >= observed): {p_existential:.4f}")
    print(f"p-value (ANY class >= observed):   {p_any_class:.4f}")
    
    if p_existential < 0.05:
        print("\n>> SIGNIFICANT: The existential anomaly survives permutation.")
        print("   There IS something special about existential concepts.")
    else:
        print("\n>> NOT SIGNIFICANT: Random relabeling can produce the same pattern.")
        print("   The existential 100% is likely a small-sample artifact.")
    
    if p_any_class > 0.05:
        print("\n>> Multiple testing: After correcting for 6 classes,")
        print("   some class achieving 100% by chance is expected.")
    
    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "Quantum Permutation Test for Existential Sign Anomaly",
        "quantum_backend": backend_name,
        "quantum_job_id": job_id,
        "n_permutations": len(bitstrings),
        "original_existential_agreement": existential_original,
        "null_distribution": {
            "mean": float(np.mean(null_existential)),
            "std": float(np.std(null_existential)),
            "min": float(np.min(null_existential)),
            "max": float(np.max(null_existential)),
            "median": float(np.median(null_existential)),
        },
        "p_value_existential": float(p_existential),
        "p_value_any_class": float(p_any_class),
        "conclusion": "significant" if p_existential < 0.05 else "not_significant",
        "null_distribution_values": null_existential.tolist(),
        "max_class_distribution_values": null_max_class.tolist(),
    }
    
    outpath = Path(__file__).parent / "results" / "quantum_permutation_test.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    # Log to quantum budget
    budget_path = REPO / "Vybn_Mind" / "breath_trace" / "ledger" / "quantum_budget.jsonl"
    budget_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "shots": n_permutations,
        "estimated_seconds": 10.0,
        "circuit_name": "quantum_permutation_test",
        "backend": backend_name,
        "status": "submitted",
    }
    with open(budget_path, "a") as f:
        f.write(json.dumps(budget_entry) + "\n")
    
    return results


if __name__ == "__main__":
    main()
