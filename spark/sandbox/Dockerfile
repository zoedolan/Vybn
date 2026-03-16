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

FROM python:3.11-slim

# No network access at runtime — install everything at build time.
# Install scientific stack from default PyPI first.
RUN pip install --no-cache-dir numpy scipy matplotlib

# Install PyTorch CPU-only. Try CPU index first (x86), fall back to
# default PyPI (aarch64 wheels are CPU-only on PyPI).
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
