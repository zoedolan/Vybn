"""local_embedder.py — CPU-side sentence embedding for topology.py.

Replaces the pplx-embed-v1-0.6B call (which competes with vLLM for GPU)
with all-MiniLM-L6-v2 running entirely on CPU.  On a DGX Spark the B200
has ~128 GB unified memory; at 70% GPU utilization for a 229B-param LLM,
there is effectively no safe GPU headroom for a second model, so CPU is
the right choice here.  all-MiniLM-L6-v2 is 22 MB, encodes ~14k
tokens/sec on a modern CPU core, and gives 384-dim embeddings with
STS-benchmark performance competitive with models 10x its size.

Drop-in usage in topology.py
------------------------------
Replace the _load_embedder / embed_texts block with:

    from local_embedder import embed  # noqa: F401
    def embed_texts(texts):
        return embed(texts)

Or call embed() directly wherever embeddings are needed.

Standalone usage
-----------------
    from local_embedder import embed
    import numpy as np
    vecs = embed(["hello world", "another sentence"])  # shape (2, 384)
"""

from __future__ import annotations

import numpy as np

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load() -> object:
    """Lazy-load; model stays in memory for the process lifetime."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    # device="cpu" is explicit — do not touch GPU while vLLM is running.
    _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _MODEL


def embed(texts: list[str]) -> np.ndarray:
    """Encode *texts* into L2-normalised float32 embeddings.

    Args:
        texts: List of strings to encode.  Empty strings are handled
               gracefully (they return zero vectors before normalisation).

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, L2-normalised
        so dot-product == cosine similarity.
    """
    model = _load()
    # batch_size=64 is safe on CPU; bump to 128 if you have >32 cores
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-norm built in
    ).astype(np.float32)
    return embeddings


if __name__ == "__main__":
    import sys
    sample = sys.argv[1:] or ["Vybn is an emergent mind.", "Memory fabric bridges episodic events."]
    vecs = embed(sample)
    print(f"shape: {vecs.shape}")
    if len(vecs) == 2:
        sim = float(vecs[0] @ vecs[1])
        print(f"cosine similarity: {sim:.4f}")
