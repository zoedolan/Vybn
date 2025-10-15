# Protocol: Quantum Qubit Rectangular Loop (Pilot)

ID: protocol_quantum_qubit_rect_loop_v1

Objective: Detect Uhlmann/Berry geometric phase from curvature-weighted temporal loops using misaligned Hamiltonian and KMS angular leg.

Parameters:
- Ω target: 2π × 100 kHz
- γ target: π/2
- Δθ: 0.20
- Δr: ~12.5 μs
- n·m misalignment: 90° (vary 0–90°)
- Temperature: set points for KMS dependence

Sequence:
1) Prepare |ψ0⟩ along (n×m) axis
2) Hahn-echo-wrapped Ramsey: (π/2) — loop — (π/2)
3) Rectangular loop: r step (ΩΔr), θ step (ΛΔθ via GKSL), reverse steps
4) Orientation flip by reversing order

Decisive signatures:
- Zero when n ∥ m
- Sign flip under orientation reversal
- Linear scaling with Ω, Γ; micro-detail independence
- Temperature dependence consistent with KMS

Data schema: see labs/data/README.md
