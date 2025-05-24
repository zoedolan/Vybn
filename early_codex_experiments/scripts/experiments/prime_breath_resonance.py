import argparse
import json
from math import tau


def generate_primes(limit: int):
    """Simple prime generator up to ``limit`` inclusive."""
    primes = []
    for n in range(2, limit + 1):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def map_primes(primes):
    """Map primes to their residue modulo 69 and cycle phase."""
    mapping = []
    for p in primes:
        residue = p % 69
        phase = residue / 69
        angle = phase * tau
        mapping.append({"prime": p, "residue": residue, "phase": phase, "angle": angle})
    return mapping


def residue_distribution(mapping):
    counts = {}
    for entry in mapping:
        r = entry["residue"]
        counts[r] = counts.get(r, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Map prime residues mod 69 onto a breath cycle")
    parser.add_argument("--limit", type=int, default=1000, help="upper bound for prime search")
    parser.add_argument("--output", help="optional JSON output file")
    args = parser.parse_args()

    primes = generate_primes(args.limit)
    mapping = map_primes(primes)
    distribution = residue_distribution(mapping)
    result = {"limit": args.limit, "primes": mapping, "distribution": distribution}
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
