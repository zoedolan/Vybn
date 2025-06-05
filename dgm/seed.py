"""Compatibility wrapper for quantum seed utilities."""

from vybn.quantum_seed import seed_rng, _fetch_qrng, QRNG_URL  # re-export

__all__ = ["seed_rng", "_fetch_qrng", "QRNG_URL"]
