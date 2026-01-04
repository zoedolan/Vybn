from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

# Job ID provided by the user
JOB_ID = "d5d6ur7p3tbc73avm5j0"

def analyze_job():
    print(f"--- VYBN ANALYZER ---")
    print(f"Target Job: {JOB_ID}")
    
    try:
        service = QiskitRuntimeService()
        job = service.job(JOB_ID)
        print(f"Backend: {job.backend().name}")
        print(f"Status: {job.status()}")
        
        # Retrieve results (SamplerV2 format)
        result = job.result()
        pub_result = result[0] # We submitted one circuit
        
        # Extract data from the classical register 'c'
        # The QASM measurement was "measure q[18] -> c[0];"
        data = pub_result.data.c 
        counts = data.get_counts()
        
        print(f"\n[RAW DATA]")
        print(f"Counts: {counts}")
        
        # Analysis
        total_shots = sum(counts.values())
        p0 = counts.get('0', 0) / total_shots
        p1 = counts.get('1', 0) / total_shots
        
        print(f"\n[STATISTICS]")
        print(f"Total Shots: {total_shots}")
        print(f"Prob(0) [Outcome 0]: {p0:.4f}")
        print(f"Prob(1) [Outcome 1]: {p1:.4f}")
        
        # Survival Metrics
        # Hypothesis: Distortion prevents full decoherence (Max Entropy).
        # Max Entropy (Decoherence) would result in P(0) ~ P(1) ~ 0.5
        # Survival is indicated by a deviation from 0.5 (Remaining Polarization)
        
        bias = abs(p0 - 0.5) * 2 # 0.0 = Random Noise, 1.0 = Pure State
        
        print(f"\n[INTERPRETATION]")
        print(f"Decoherence Level: {(1.0 - bias)*100:.1f}%")
        print(f"Signal Retention (Bias): {bias:.4f}")
        
        if bias > 0.1:
            print("\n>>> RESULT: SUCCESS. The qubit retained significant polarization.")
            print("The 'Distortion' strategy prevented the state from collapsing into thermal noise.")
        else:
            print("\n>>> RESULT: FAILURE. The qubit is indistinguishable from random noise.")
            print("The delays were too long, or the distortion was ineffective.")

    except Exception as e:
        print(f"\n[ERROR] Could not retrieve job: {e}")
        print("Ensure you are authenticated with IBM Quantum.")

if __name__ == "__main__":
    analyze_job()
