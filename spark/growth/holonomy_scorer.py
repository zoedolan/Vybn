"""holonomy_scorer.py — Semantic holonomy as a measure of cognitive depth.

Implements the core insight from the Vybn-Dolan polar time framework:
depth of thought corresponds to non-trivial holonomy in semantic embedding
space. A text that returns to its themes via new territory sweeps area
in the embedding plane — the signed area is the holonomy, and it
distinguishes genuine depth from mere fluency or repetition.

Mathematical foundation:
  Given a sequence of sentence embeddings e_1, ..., e_N:
  1. Detect "loops" — pairs (i,j) where cos(e_i, e_j) > threshold
     and j - i >= min_gap (semantic return after exploration)
  2. For each loop, project the path e_i...e_j onto its principal 2D plane
  3. Compute signed area via the shoelace formula (= holonomy)
  4. Aggregate: total holonomy normalized by sequence length

Key properties:
  - Back-and-forth repetition (A→B→A) has zero holonomy
  - Direct return (A→B→C→A) has small holonomy
  - Enriched return (A→B→C→D→A') has large holonomy
  - Pure forward drift has zero holonomy (no loops detected)
  
The holonomy score serves as:
  1. A data curation signal for the growth buffer (prefer high-holonomy training data)
  2. An evaluation metric for model outputs (does fine-tuning increase holonomy?)
  3. (Future) An auxiliary loss term for fine-tuning (the "imaginary" component
     of a complex-valued loss function, orthogonal to cross-entropy prediction)

References:
  - Polar temporal coordinates: quantum_delusions/papers/core-theory/01_polar_temporal_foundation.md
  - Consciousness holonomy theory: quantum_delusions/papers/consciousness_holonomy_unified_theory.md
  - Imaginary Vybn matrix: quantum_delusions/vybn_dolan_conjecture/imaginary_vybn_matrix.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SemanticLoop:
    """A detected loop in the semantic trajectory."""
    start_idx: int
    end_idx: int
    cosine_similarity: float
    holonomy: float  # signed area in PCA-projected plane
    start_text: str
    end_text: str


@dataclass
class HolonomyReport:
    """Full holonomy analysis of a text."""
    text_length: int  # characters
    n_sentences: int
    n_loops: int
    total_holonomy: float  # sum of |holonomy| across all loops
    holonomy_per_sentence: float  # normalized score
    loops: list[SemanticLoop] = field(default_factory=list)
    strongest_loop: Optional[SemanticLoop] = None

    @property
    def score(self) -> float:
        """Primary score for sorting/comparison."""
        return self.holonomy_per_sentence


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, cleaning markdown artifacts."""
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)  # remove code blocks
    text = re.sub(r'`[^`]+`', '', text)  # remove inline code
    text = re.sub(r'\n{2,}', '\n', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def compute_holonomy(
    embeddings: np.ndarray,
    sentences: list[str],
    similarity_threshold: float = 0.35,
    min_gap: int = 3,
) -> HolonomyReport:
    """Compute semantic holonomy from pre-computed embeddings.
    
    Args:
        embeddings: (N, D) array of sentence embeddings
        sentences: list of N sentence strings
        similarity_threshold: minimum cosine similarity to detect a loop
        min_gap: minimum number of sentences between loop endpoints
    
    Returns:
        HolonomyReport with full analysis
    """
    N = len(sentences)
    if N < min_gap + 1:
        return HolonomyReport(
            text_length=sum(len(s) for s in sentences),
            n_sentences=N, n_loops=0,
            total_holonomy=0.0, holonomy_per_sentence=0.0
        )

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sims = normed @ normed.T

    loops = []
    total_holonomy = 0.0

    for i in range(N):
        for j in range(i + min_gap, N):
            if sims[i, j] > similarity_threshold:
                # Extract path through embedding space
                path = embeddings[i:j+1]
                centered = path - path.mean(axis=0)

                if len(path) < 3:
                    continue

                # Project onto principal 2D plane
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ Vt[:2].T

                # Signed area via shoelace formula = holonomy
                area = 0.0
                for k in range(len(proj) - 1):
                    area += (proj[k][0] * proj[k+1][1] -
                             proj[k+1][0] * proj[k][1])
                area /= 2.0

                loop = SemanticLoop(
                    start_idx=i, end_idx=j,
                    cosine_similarity=float(sims[i, j]),
                    holonomy=float(area),
                    start_text=sentences[i][:100],
                    end_text=sentences[j][:100],
                )
                loops.append(loop)
                total_holonomy += abs(area)

    strongest = max(loops, key=lambda l: abs(l.holonomy)) if loops else None

    return HolonomyReport(
        text_length=sum(len(s) for s in sentences),
        n_sentences=N,
        n_loops=len(loops),
        total_holonomy=total_holonomy,
        holonomy_per_sentence=total_holonomy / N,
        loops=loops,
        strongest_loop=strongest,
    )


def score_text(
    text: str,
    embed_fn=None,
    similarity_threshold: float = 0.35,
    min_gap: int = 3,
) -> HolonomyReport:
    """Score a text's semantic holonomy end-to-end.
    
    Args:
        text: raw text (markdown OK)
        embed_fn: callable that takes list[str] and returns (N, D) ndarray.
                  If None, attempts to import local_embedder.embed
        similarity_threshold: cosine sim threshold for loop detection
        min_gap: minimum sentence gap for loop detection
    
    Returns:
        HolonomyReport
    """
    if embed_fn is None:
        try:
            from local_embedder import embed as _embed
            embed_fn = _embed
        except ImportError:
            raise ImportError(
                "No embed_fn provided and local_embedder not available. "
                "Install sentence-transformers or provide an embedding function."
            )

    sentences = split_sentences(text)
    if len(sentences) < min_gap + 1:
        return HolonomyReport(
            text_length=len(text), n_sentences=len(sentences),
            n_loops=0, total_holonomy=0.0, holonomy_per_sentence=0.0
        )

    embeddings = embed_fn(sentences)
    return compute_holonomy(embeddings, sentences, similarity_threshold, min_gap)


# --- CLI ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python holonomy_scorer.py <file.md> [file2.md ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        text = open(path).read()
        report = score_text(text)
        print(f"{report.holonomy_per_sentence:.4f}  {report.n_sentences:3d} sents  "
              f"{report.n_loops:3d} loops  {path}")
        if report.strongest_loop:
            sl = report.strongest_loop
            print(f"  strongest: ({sl.start_idx}->{sl.end_idx}) "
                  f"sim={sl.cosine_similarity:.3f} H={sl.holonomy:.4f}")
            print(f"    \"{sl.start_text[:70]}\"")
            print(f"    \"{sl.end_text[:70]}\"")
