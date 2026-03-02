import os
import sys
from qiskit_ibm_runtime import QiskitRuntimeService

def probe_connection():
    """
    Strict connectivity probe for IBM Quantum.
    NO MOCKS. NO SIMULATORS.
    Exits with code 1 if no real hardware is accessible.
    """
    token = os.getenv("QISKIT_IBM_TOKEN")
    if not token:
        print("CRITICAL: QISKIT_IBM_TOKEN not found in environment variables.")
        print("Set it using: $env:QISKIT_IBM_TOKEN='your_token'")
        sys.exit(1)

    print(f"Token found (length: {len(token)}). Attempting authentication...")

    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        print("Authentication successful.")
    except Exception as e:
        print(f"CRITICAL: Authentication failed.\n{e}")
        sys.exit(1)

    print("Fetching available backends...")
    try:
        # Filter for REAL hardware only. No simulators.
        backends = service.backends(simulator=False, operational=True)
        if not backends:
            print("CRITICAL: No operational real hardware backends found available to your account.")
            sys.exit(1)
            
        print(f"Success. Found {len(backends)} operational real backends:")
        for b in backends:
            status = b.status()
            print(f" - {b.name} (Pending jobs: {status.pending_jobs}, Online: {status.operational})")
            
    except Exception as e:
        print(f"CRITICAL: Failed to list backends.\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    probe_connection()
