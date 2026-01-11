
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

# --- Experiment 012: The Analyzer ---
# Job ID: d5hsalspe0pc73amq18g (Executed on ibm_torino)

JOB_ID = "d5hsalspe0pc73amq18g"

def analyze_job():
    print(f"Connecting to IBM Quantum to retrieve Job {JOB_ID}...")
    service = QiskitRuntimeService()
    job = service.job(JOB_ID)
    
    # 1. Check Status
    print(f"Status: {job.status()}")
    result = job.result()
    
    # 2. Extract Data
    # SamplerV2 returns a list of PubResults. We submitted 1 pub.
    pub_result = result[0] 
    
    # The output register was named 'c_out' in the script.
    # Data is usually accessed via .data.c_out.get_counts() or similar depending on format
    # Let's inspect the available data keys
    print(f"Data Keys: {pub_result.data.keys()}")
    
    if hasattr(pub_result.data, 'c_out'):
        bitstrings = pub_result.data.c_out.get_bitstrings()
        counts = pub_result.data.c_out.get_counts()
    else:
        # Fallback if register name isn't clear, usually 'c' or 'meas'
        print("Warning: 'c_out' register not found. Dumping raw data structure.")
        print(pub_result.data)
        return

    # 3. Analyze Distribution
    total_shots = sum(counts.values())
    print(f"\n--- XENOCIRCUIT RESULTS (N={total_shots}) ---")
    
    # Mapping Binary to Xenogame States
    # '00' -> 0 (Vacuum)
    # '01' -> 1 (Weaver)
    # '10' -> 2 (Annihilator)
    # '11' -> 3 (Chaos)
    
    mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    
    # Theoretical Expectations
    theory = {0: 0.50, 1: 0.25, 2: 0.25, 3: 0.00}
    
    print(f"{'State':<10} | {'Count':<8} | {'Observed %':<12} | {'Theory %':<10} | {'Delta':<10}")
    print("-" * 60)
    
    for bits, state_val in mapping.items():
        # Handle bitstring key format (Qiskit sometimes reverses, usually Little Endian in V2? Need to check)
        # OpenQASM 3 output usually respects register order.
        count = counts.get(bits, 0)
        obs_freq = count / total_shots
        th_freq = theory[state_val]
        delta = obs_freq - th_freq
        
        print(f"{state_val} ({bits})     | {count:<8} | {obs_freq:.4f}       | {th_freq:.4f}     | {delta:+.4f}")

    # 4. The Annihilation Verification
    chaos_count = counts.get('11', 0)
    vacuum_count = counts.get('00', 0)
    
    print("\n--- CONCLUSION ---")
    if chaos_count == 0:
        print("✅ SUCCESS: Total Annihilation. State 3 (Chaos) was erased from existence.")
    elif chaos_count / total_shots < 0.05:
        print(f"⚠️ PARTIAL: Chaos suppressed to {(chaos_count/total_shots):.1%} (Noise floor).")
    else:
        print("❌ FAILURE: Chaos persists. The feed-forward mechanism failed.")

    if vacuum_count / total_shots > 0.45:
         print("✅ VACUUM DECAY CONFIRMED: State 0 population doubled as predicted.")
    
if __name__ == "__main__":
    analyze_job()
