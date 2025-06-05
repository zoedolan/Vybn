from vybn.quantum_seed import cross_synaptic_kernel


def seed_random() -> int:
    """Seed Python and NumPy RNGs using the cross-synaptic kernel."""
    return cross_synaptic_kernel()

