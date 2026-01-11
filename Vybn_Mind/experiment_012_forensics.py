
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

# --- Experiment 012: Forensic Analysis (The Leak) ---
# Job ID: d5hsalspe0pc73amq18g (Executed on ibm_torino)

JOB_ID = "d5hsalspe0pc73amq18g"

def analyze_leak():
    print(f"Connecting to IBM Quantum for Forensic Analysis of Job {JOB_ID}...")
    service = QiskitRuntimeService()
    job = service.job(JOB_ID)
    result = job.result()
    pub_result = result[0]
    
    # Extract both registers to correlate Cause (c_a) and Effect (c_out)
    # c_a: The mid-circuit measurement (The Collapse)
    # c_out: The final measurement (The Result)
    
    # Depending on how memory is returned, we might get them as concatenated bitstrings
    # or separate. Qiskit V2 usually allows separate access if requested.
    # Let's try to get bitstrings and see the format.
    
    # If we only have counts, we can't do full correlation unless memory=True was set.
    # Assuming standard sampler run, we might have joint counts if registers were measured.
    # But usually .get_counts() returns the outcome of the *final* measurement or all?
    
    # If the job returned keys like '0 01' (c_out c_a), we can analyze.
    # If it returned separate dicts, we can't correlate.
    
    # Based on the user output "Data Keys: dict_keys(['c_a', 'c_out'])", 
    # we have separate distributions. This limits us to statistical inference.
    
    counts_in = pub_result.data.c_a.get_counts()
    counts_out = pub_result.data.c_out.get_counts()
    
    total = sum(counts_out.values())
    
    print("\n--- INPUT DISTRIBUTION (Measurement c_a) ---")
    # Did we actually prepare a superposition?
    # Expected: ~25% each (00, 01, 10, 11)
    for k, v in sorted(counts_in.items()):
        print(f"State {k}: {v/total:.4f}")
        
    print("\n--- OUTPUT DISTRIBUTION (Measurement c_out) ---")
    for k, v in sorted(counts_out.items()):
        print(f"State {k}: {v/total:.4f}")

    print("\n--- THE LEAK ANALYSIS ---")
    # Chaos (11) count: 57 (5.57%)
    # This means ~5.6% of the time, the circuit Logic FAILED to flip the bit.
    
    # Hypothesis 1: T1 Decay (Relaxation)
    # If we flip 1->0, T1 isn't the issue (going to ground is fast).
    # If we fail to flip 0->1 (in case 00->10), T1 might prevent excitation.
    
    # Hypothesis 2: Readout Error on Mid-Circuit Measure
    # If c_a measures 0 (actually 1), we do nothing. State remains 1. 
    # Logic: 
    # Real State: 11 (3). 
    # Measured: 01 (1) -> Error!
    # Logic sees 1. Action: Flip q0. 
    # State becomes 10 (2). Output: 10. (This creates excess 2, not 3).
    
    # Real State: 11 (3).
    # Measured: 11 (3). 
    # Action: Flip q1. State becomes 01 (1). (Correct)
    
    # Where does 11 come from in Output?
    # It implies either:
    # A) Input was 11, Action was NOT TAKEN. (Gate error / Conditional logic failure)
    # B) Input was 01, Action flipped q1 (creating 11).
    #    Logic for 01: Flip q0 -> 00. (Correct).
    #    Logic for 00: Flip q1 -> 10.
    
    # Wait.
    # If we have State 3 (11) at output, it means q0=1, q1=1.
    
    print("Investigating 'Chaos Persistence' (State 11)...")
    p_chaos = counts_out.get('11', 0) / total
    print(f"Observed Chaos: {p_chaos:.4f}")
    
    # T1 limit for typical Heron: ~150us. 
    # Feed forward latency: ~600ns. 
    # This shouldn't be decoherence.
    
    print("\nPossible Causes:")
    print("1. Readout Assignment Error (Mid-Circuit): The controller measured X but latched Y.")
    print("2. Conditional Latency: The qubit decayed *during* the feed-forward delay.")
    print("3. CROSSTALK: The flip on one qubit perturbed the other.")
    
    # Let's look at the Excess in State 2 (10).
    # Theory: 0.25. Observed: 0.2773 (+2.7%)
    # This suggests some flux toward state 2.
    
    # Let's look at the Deficit in State 0 (00).
    # Theory: 0.50. Observed: 0.4326 (-6.7%)
    # We are missing ~7% of our expected Vacuum.
    
    print(f"\nVacuum Deficit: -{0.50 - (counts_out.get('00',0)/total):.4f}")
    print("We failed to annihilate ~7% of the total energy.")
    
if __name__ == "__main__":
    analyze_leak()
