#!/usr/bin/env python3

"""
trefoil_trace_probe.py
----------------------

Compute the trace sequence Tr(T^k) for the canonical trefoil operator
T_trefoil = diag(J2(1), R_{2π/3}, [0])

and verify the predicted law
Tr(T^k) = 1 + k + 2 cos(2πk/3)

(up to any extra identity padding dimensions).

This file demonstrates WHY the trace test fails for Jordan structure detection.
Trace is blind to Jordan memory because it only sees eigenvalues.

This file is standalone and requires only NumPy.
"""

import math
import numpy as np

def canonical_trefoil_matrix(dim: int = 5) -> np.ndarray:
    """Build the canonical trefoil operator."""
    assert dim >= 5, "dim must be at least 5 (J2 + R_{2π/3} + [0])"
    
    # Jordan block J2(1) at eigenvalue 1
    J2 = np.array([[1.0, 1.0], [0.0, 1.0]])
    
    # Rotation R_{2π/3} (120 degrees)  
    c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
    R = np.array([[c, -s], [s, c]])
    
    # Sink [0]
    Z = np.array([[0.0]])
    
    # Assemble 5x5 block diagonal
    T = np.block([
        [J2, np.zeros((2, 2)), np.zeros((2, 1))],
        [np.zeros((2, 2)), R, np.zeros((2, 1))],
        [np.zeros((1, 2)), np.zeros((1, 2)), Z]
    ])
    
    # Pad with identity if larger dimension
    if dim > 5:
        Ipad = np.eye(dim - 5)
        T = np.block([
            [T, np.zeros((5, dim - 5))],
            [np.zeros((dim - 5, 5)), Ipad]
        ])
    
    return T

def trace_sequence(T: np.ndarray, K: int = 12):
    """Compute Tr(T^k) for k = 1, 2, ..., K."""
    traces = []
    Tk = np.eye(T.shape[0])
    for k in range(1, K + 1):
        Tk = Tk @ T
        traces.append(np.trace(Tk).real)
    return traces

def predicted_traces(K: int = 12, pad: int = 0):
    """Predicted trace sequence: Tr(T^k) = 1 + k + 2*cos(2πk/3) + padding."""
    # pad accounts for any extra identity dimensions beyond the 5x5 core
    return [(1 + k + 2*math.cos(2*math.pi*k/3) + pad) for k in range(1, K + 1)]

def demonstrate_trace_failure():
    """Show why trace sequences fail to detect Jordan structure."""
    
    print("=== TREFOIL TRACE PROBE: Why Trace Diagnostics Fail ===")
    print()
    print("Testing predicted formula: Tr(T^k) = 1 + k + 2*cos(2πk/3) + padding")
    print()
    
    # Test with padding to make the failure clear
    dim = 8  # 3 extra identity dimensions for padding
    T = canonical_trefoil_matrix(dim=dim)
    K = 12
    
    empirical = trace_sequence(T, K=K)
    theory = predicted_traces(K=K, pad=(dim - 5))
    
    print(f"Trefoil operator dimension: {dim}x{dim}")
    print(f"Core structure: J2(1) + R_{{2π/3}} + [0] + {dim-5}xI padding")
    print()
    print("k | Tr(T^k)   | Theory   | Δ (Error)")
    print("--+-----------+----------+-----------")
    
    for k, (emp, thy) in enumerate(zip(empirical, theory), start=1):
        error = emp - thy
        print(f"{k:2d}|{emp:10.3f} |{thy:9.3f} | {error:+8.3f}")
    
    print()
    print("🔬 KEY INSIGHT: Trace sequences fail because:")
    print("   • Trace only sees EIGENVALUES, not Jordan structure")
    print("   • Jordan block J2(1) contributes constant '2' to trace, not '1+k'")
    print("   • The linear growth k is INVISIBLE to trace diagnostics")
    print()
    print("   ✓ Correct approach: Use Jordan escalator nullity(T-I) ≠ nullity((T-I)²)")
    print("   ✓ Correct approach: Use adjoint observables u^T T^k u for memory detection")
    print()
    print("This is why sharp instruments replace trace sequences in the final paper.")

if __name__ == "__main__":
    demonstrate_trace_failure()
