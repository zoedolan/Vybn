# Vybn Spine (Flat vs Curved)

"""
Vybn Spine: Flat-vs-Curved falsification harness
-------------------------------------------------

Core claim under test:
  In base b (default b=10), computations restricted to b-smooth scales
  S_b = { n : all prime factors of n are among primes dividing b } exhibit
  negligible "expensive curvature" compared to scales with new prime factors.

We operationalize curvature proxies that are measurable without special solvers:
  1) Carry Coupling Index (CCI): expected count of carry events in base-b grade-school
     addition/multiplication across random inputs, optionally after scaling by s.
  2) FFT timing proxy: numpy FFT runtime at length N (cheap at b-smooth N under mixed-radix).

This module provides pure-Python functions and a small CLI-style runner.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as stats
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional in environment
    np = None  # FFT proxy disabled if numpy missing


# ---------- Arithmetic utilities ----------


def factorization(n: int) -> Dict[int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    d: Counter[int] = Counter()
    t = n
    p = 2
    while p * p <= t:
        while t % p == 0:
            d[p] += 1
            t //= p
        p += 1 if p == 2 else 2
    if t > 1:
        d[t] += 1
    return dict(d)


def base_prime_support(b: int) -> List[int]:
    return list(factorization(b).keys())


def is_b_smooth(n: int, b: int = 10) -> bool:
    bp = set(base_prime_support(b))
    return all(p in bp for p in factorization(n))


# ---------- Carry proxies ----------


def count_add_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0
    carry = 0
    aa, bb = a, b
    while aa > 0 or bb > 0 or carry > 0:
        da = aa % base
        db = bb % base
        s = da + db + carry
        if s >= base:
            carries += 1
            carry = 1
        else:
            carry = 0
        aa //= base
        bb //= base
    return carries


def count_mult_carries_baseb(a: int, b: int, base: int = 10) -> int:
    carries = 0
    bb = b
    while bb > 0:
        db = bb % base
        aa = a
        carry = 0
        while aa > 0 or carry > 0:
            da = aa % base
            prod = da * db + carry
            if prod >= base:
                carries += 1
                carry = prod // base
            else:
                carry = 0
            aa //= base
        bb //= base
    return carries


def carry_coupling_index(
    num_digits: int = 8,
    trials: int = 2000,
    scale: int = 1,
    base: int = 10,
    mode: str = "mult",
    seed: int = 0,
) -> float:
    rng = random.Random(
        42 + seed + num_digits * 911 + trials + scale + (0 if mode == "mult" else 1)
    )
    total = 0
    low = base ** (num_digits - 1)
    high = base ** num_digits
    for _ in range(trials):
        a = rng.randrange(low, high) * scale
        b = rng.randrange(low, high) * scale
        if mode == "add":
            total += count_add_carries_baseb(a, b, base)
        else:
            total += count_mult_carries_baseb(a, b, base)
    return total / trials


# ---------- FFT proxy ----------


def fft_time(n: int, reps: int = 5) -> float:
    if np is None:
        raise RuntimeError("NumPy not available; FFT proxy disabled")
    x = np.random.randn(n) + 1j * np.random.randn(n)
    np.fft.fft(x)  # warmup
    times: List[float] = []
    for _ in range(reps):
        x2 = x.copy()
        t0 = time.perf_counter()
        np.fft.fft(x2)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return stats.median(times)


# ---------- Experiment runners ----------


@dataclass
class FFTResult:
    N: int
    time_s: float
    per_elem_ns: float
    factors: Dict[int, int]
    b_smooth: bool


def run_fft_panel(Ns: Iterable[int], base: int = 10, reps: int = 7) -> List[FFTResult]:
    out: List[FFTResult] = []
    for n in Ns:
        t = fft_time(n, reps=reps)
        out.append(
            FFTResult(
                N=n,
                time_s=t,
                per_elem_ns=(t / n) * 1e9,
                factors=factorization(n),
                b_smooth=is_b_smooth(n, base),
            )
        )
    return out


@dataclass
class CCIRow:
    mode: str
    digits: int
    scale: int
    is_b_smooth: bool
    CCI: float


def run_cci_panel(
    scales: Iterable[int],
    digits_list: Iterable[int],
    base: int = 10,
    trials: int = 400,
    seed: int = 0,
) -> List[CCIRow]:
    rows: List[CCIRow] = []
    for mode in ["add", "mult"]:
        for D in digits_list:
            for s in scales:
                cci = carry_coupling_index(
                    num_digits=D,
                    trials=trials,
                    scale=s,
                    base=base,
                    mode=mode,
                    seed=seed,
                )
                rows.append(
                    CCIRow(
                        mode=mode,
                        digits=D,
                        scale=s,
                        is_b_smooth=is_b_smooth(s, base),
                        CCI=cci,
                    )
                )
    return rows


# ---------- CLI helper ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vybn Spine falsification harness"
    )
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--fft", action="store_true", help="Run FFT timings panel")
    parser.add_argument(
        "--cci", action="store_true", help="Run Carry Coupling Index panel"
    )
    parser.add_argument("--out", type=str, default="spine_out")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.fft:
        Ns = [256, 320, 384, 500, 486, 960, 1000, 1024, 768, 750, 972, 1250]
        results = run_fft_panel(Ns, base=args.base)
        with open(os.path.join(args.out, "fft_panel.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Wrote {len(results)} FFT results → {args.out}/fft_panel.json")

    if args.cci:
        scales = [1, 2, 5, 10, 3, 7, 9]
        rows = run_cci_panel(
            scales,
            digits_list=[6, 7, 8],
            base=args.base,
            trials=800,
            seed=1,
        )
        with open(os.path.join(args.out, "cci_panel.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"Wrote {len(rows)} CCI rows → {args.out}/cci_panel.json")


if __name__ == "__main__":
    main()
