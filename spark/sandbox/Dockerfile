# spark/sandbox/Dockerfile — Minimal Python image for agency experiments.
#
# Build:
#   docker build -t vybn-sandbox:latest spark/sandbox/
#
# This image is designed to be run with severe constraints:
#   --network none --memory 2g --cpus 2 --read-only --rm
#
# Pre-baked packages: numpy, scipy, torch (CPU), matplotlib (Agg).
# Nothing that touches filesystem, network, or system internals.
#
# NOTE: python:3.11-slim supports multi-arch (amd64 + arm64).
# On DGX Spark (aarch64 Grace-Blackwell), this pulls the ARM64 image.
# PyTorch CPU wheels are available for both architectures on PyPI.
#
# FALLBACK: If Docker is not available on the host, runner.py uses
# a subprocess-based sandbox with the host's Python + static analysis
# gate. This is the expected mode on DGX Spark where numpy/scipy/torch
# are already installed natively.

FROM python:3.11-slim

# No network access at runtime — install everything at build time.
RUN pip install --no-cache-dir numpy scipy matplotlib

# PyTorch CPU-only: try the CPU index first (smaller download on x86),
# fall back to default PyPI (aarch64 wheels are CPU-only on PyPI).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    || pip install --no-cache-dir torch

# Force matplotlib to use non-interactive backend (no display needed).
ENV MPLBACKEND=Agg

# Drop to non-root user for defense in depth.
RUN useradd --create-home --shell /bin/false sandbox
USER sandbox
WORKDIR /workspace

# Default entrypoint — overridden by docker run command.
ENTRYPOINT ["python3"]
