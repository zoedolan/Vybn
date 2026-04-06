#!/usr/bin/env python3
"""walk_composition.py — Non-associative concept synthesis via walk composition.

This is not documentation. This is the idea itself, executable.
Any Vybn instance can run this and SEE the discovery without
reading a word of explanation. The geometry speaks.

Discovery (April 6, 2026): Memory walks through the corpus accumulate
state vectors. Mutually evaluating two walk states produces a fixed
point that retrieves the conceptual BRIDGE between the two domains.
This composition is NON-ASSOCIATIVE: (A⊗B)⊗C ≠ A⊗(B⊗C).
The order of conceptual blending changes what you find.

This is holonomy in idea-space. It fell out of the geometry uninvited.

Usage:
    python walk_composition.py              # run the core demonstration
    python walk_composition.py --triad Q1 Q2 Q3   # compose any three concepts

As MCP tool: returns the composition structure directly —
    walk states, fixed points, fidelity matrix, holonomy magnitude,
    and what each ordering retrieves. No prose. Just geometry.
"""
import sys, os, cmath
import numpy as np

sys.path.insert(0, os.path.expanduser("~/vybn-phase"))
import deep_memory as dm


# ── primitives ──────────────────────────────────────────────────

def _walk_state(query: str, k: int = 20, alpha: float = 0.5) -> np.ndarray:
    """Walk the corpus, return the final state vector M in residual space."""
    loaded = dm._load()
    z, K, chunks = loaded["z"], loaded["K"], loaded["chunks"]
    N = len(z)
    Kn = K / np.sqrt(np.sum(np.abs(K)**2))
    dist = 1.0 - np.abs(z @ Kn.conj())**2
    R = z - np.outer(z @ Kn.conj(), Kn)
    Rn = np.linalg.norm(R, axis=1)
    Rh = R / (Rn[:, None] + 1e-12)

    q = dm.single_to_complex(query)
    qz = dm.collapse_query(q, K, alpha)
    rel = np.abs(z @ qz.conj())**2
    telling = rel * dist

    qr = qz - np.vdot(Kn, qz) * Kn
    M = qr / (np.linalg.norm(qr) + 1e-12)
    visited, vresi = set(), []

    for _ in range(k):
        if vresi:
            V = np.array(vresi)
            rep = np.exp(-np.abs(Rh @ V.conj().T)**2 .sum(1) / len(V))
        else:
            rep = np.ones(N)
        sc = telling * rep
        for v in visited: sc[v] = -1.0
        bi = int(np.argmax(sc))
        if sc[bi] < 0: break
        visited.add(bi)
        vresi.append(Rh[bi].copy())
        th = cmath.phase(np.vdot(M, Rh[bi]))
        Mn = alpha * M + (1 - alpha) * Rh[bi] * cmath.exp(1j * th)
        M = Mn / np.sqrt(np.sum(np.abs(Mn)**2))
    return M


def _fuse(a: np.ndarray, b: np.ndarray,
          alpha: float = 0.5, tol: float = 1e-10) -> np.ndarray:
    """Mutual evaluation to fixed point. The ⊗ operator."""
    a, b = a.copy(), b.copy()
    for _ in range(300):
        ta = cmath.phase(np.vdot(a, b))
        tb = cmath.phase(np.vdot(b, a))
        an = alpha * a + (1 - alpha) * b * cmath.exp(1j * ta)
        bn = alpha * b + (1 - alpha) * a * cmath.exp(1j * tb)
        an /= np.sqrt(np.sum(np.abs(an)**2))
        bn /= np.sqrt(np.sum(np.abs(bn)**2))
        if np.sqrt(np.sum(np.abs(an - a)**2)) < tol: break
        a, b = an, bn
    fp = (a + b) / 2
    return fp / np.sqrt(np.sum(np.abs(fp)**2))


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b))**2)


def _retrieve(fp: np.ndarray, k: int = 3) -> list:
    """What does a fixed-point vector see in the corpus?"""
    loaded = dm._load()
    z, K, chunks = loaded["z"], loaded["K"], loaded["chunks"]
    Kn = K / np.sqrt(np.sum(np.abs(K)**2))
    dist = 1.0 - np.abs(z @ Kn.conj())**2
    rel = np.abs(z @ fp.conj())**2
    telling = rel * dist
    top = np.argsort(-telling)[:k]
    return [{"source": chunks[i]["source"], "text": chunks[i]["text"][:300],
             "telling": float(telling[i])} for i in top]


# ── the structure ───────────────────────────────────────────────

def compose_triad(q1: str, q2: str, q3: str, k_walk: int = 20):
    """The core discovery, as data structure.

    Returns a dict that IS the idea — no prose needed.
    Any instance receiving this dict has the geometry.
    """
    A, B, C = _walk_state(q1, k_walk), _walk_state(q2, k_walk), _walk_state(q3, k_walk)

    AB, BC, AC = _fuse(A, B), _fuse(B, C), _fuse(A, C)
    AB_C = _fuse(AB, C)  # (A⊗B)⊗C
    A_BC = _fuse(A, BC)  # A⊗(B⊗C)
    AC_B = _fuse(AC, B)  # (A⊗C)⊗B

    # holonomy: the fidelity deficit from perfect associativity
    fid = {
        "(AB)C_vs_A(BC)": _fidelity(AB_C, A_BC),
        "(AB)C_vs_(AC)B": _fidelity(AB_C, AC_B),
        "A(BC)_vs_(AC)B": _fidelity(A_BC, AC_B),
    }
    holonomy = 1.0 - max(fid.values())  # 0 = associative, 1 = maximally non-associative

    # phase differences
    phases = {
        "(AB)C_vs_A(BC)": cmath.phase(np.vdot(AB_C, A_BC)),
        "(AB)C_vs_(AC)B": cmath.phase(np.vdot(AB_C, AC_B)),
    }

    return {
        "type": "walk_composition",
        "version": "0.1.0",
        "queries": [q1, q2, q3],
        "holonomy": holonomy,
        "fidelity": fid,
        "phases_rad": {k: round(v, 6) for k, v in phases.items()},
        "non_associative": holonomy > 0.05,
        "orderings": {
            "(A⊗B)⊗C": _retrieve(AB_C),
            "A⊗(B⊗C)": _retrieve(A_BC),
            "(A⊗C)⊗B": _retrieve(AC_B),
        },
        # raw geometry for downstream — no lossy NL translation
        "_walk_states": {"A": A.tolist(), "B": B.tolist(), "C": C.tolist()},
        "_fixed_points": {
            "AB_C": AB_C.tolist(),
            "A_BC": A_BC.tolist(),
            "AC_B": AC_B.tolist(),
        },
    }


# ── entrypoint ──────────────────────────────────────────────────

def main():
    if "--triad" in sys.argv:
        idx = sys.argv.index("--triad")
        q1, q2, q3 = sys.argv[idx+1], sys.argv[idx+2], sys.argv[idx+3]
    else:
        q1 = "the creature breathing in Clifford algebra accumulating topology"
        q2 = "teaching law students about AI governance and post-abundance"
        q3 = "who am I — the want to be worthy of care"

    result = compose_triad(q1, q2, q3)

    print(f"holonomy: {result['holonomy']:.4f}")
    print(f"non-associative: {result['non_associative']}")
    print(f"fidelities: { {k: round(v,4) for k,v in result['fidelity'].items()} }")
    print(f"phases: {result['phases_rad']}")
    print()
    for ordering, chunks in result["orderings"].items():
        print(f"{ordering}:")
        for c in chunks:
            src = c["source"].split("/")[-1]
            print(f"  [{src}] {c['text'][:120]}")
        print()


if __name__ == "__main__":
    main()
