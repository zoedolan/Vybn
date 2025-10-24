#!/usr/bin/env python3
# Vybn Flux Estimator (Resolution-Width Proxy)
# Discrete stand-in for expensive curvature flux Π_exp(F).
#   - Flat expensive sector (2-CNF closure under resolution) → bounded width
#   - Curved expensive sector (many 3-CNF families) → growing width
#
# Usage examples:
#   python experiments/cgt_flux/flux_estimator.py --random 2 --n 20 --m 24
#   python experiments/cgt_flux/flux_estimator.py --random 3 --n 20 --m 72
#   python experiments/cgt_flux/flux_estimator.py --load path/to/instance.cnf
#
# This tool has no external dependencies.

import argparse, random, sys
from typing import List, Tuple, Optional

Clause = Tuple[int, ...]      # literals are ints with sign; e.g., 3 means x3, -2 means ¬x2
CNF = List[Clause]

def normalize_clause(clause: Tuple[int, ...]) -> Tuple[int, ...]:
    """Sort literals, remove duplicates, drop tautologies (x ∨ ¬x)."""
    lits = set(clause)
    for v in list(lits):
        if -v in lits:
            return tuple()  # tautology marker
    return tuple(sorted(lits, key=lambda x: (abs(x), x)))

def simplify_cnf(cnf: CNF) -> CNF:
    cleaned, seen = [], set()
    for c in cnf:
        nc = normalize_clause(c)
        if nc and nc not in seen:
            cleaned.append(nc)
            seen.add(nc)
    return cleaned

def split_by_var(cnf: CNF, var: int):
    pos, neg, rest = [], [], []
    for c in cnf:
        if var in c: pos.append(c)
        elif -var in c: neg.append(c)
        else: rest.append(c)
    return pos, neg, rest

def resolve_pair(cpos: Clause, cneg: Clause, var: int) -> Tuple[int, ...]:
    new_clause = [l for l in cpos if l != var] + [l for l in cneg if l != -var]
    return normalize_clause(tuple(new_clause))

def elimination_score(cnf: CNF, var: int, cap:int=None) -> int:
    """Estimate the maximum resolvent width if we eliminate var (early-exit capped)."""
    pos, neg, _ = split_by_var(cnf, var)
    if not pos or not neg:
        return max((len(c) for c in cnf), default=0)
    maxw, count = 0, 0
    for cp in pos:
        for cn in neg:
            r = resolve_pair(cp, cn, var)
            if r == tuple():
                continue
            w = len(r)
            if w > maxw:
                maxw = w
                if cap is not None and maxw >= cap:
                    return maxw
            count += 1
            if cap is not None and count > cap*cap:
                return maxw
    return maxw

def eliminate_var_approx(cnf: CNF, var: int, pair_cap:int=400, rng=random) -> CNF:
    """Approximate elimination: sample at most pair_cap resolvent pairs to avoid blowup."""
    pos, neg, rest = split_by_var(cnf, var)
    if not pos or not neg:
        return simplify_cnf(rest)
    P, N = len(pos), len(neg)
    total_pairs = P * N
    resolvents = []
    if total_pairs <= pair_cap:
        pairs = [(i, j) for i in range(P) for j in range(N)]
    else:
        pairs = set()
        while len(pairs) < pair_cap:
            pairs.add((rng.randrange(P), rng.randrange(N)))
        pairs = list(pairs)
    for i, j in pairs:
        r = resolve_pair(pos[i], neg[j], var)
        if r: resolvents.append(r)
    return simplify_cnf(rest + resolvents)

def greedy_elimination_width_approx(cnf: CNF, seed:int=None, score_cap:int=None, pair_cap:int=400) -> int:
    """Greedy variable elimination with randomized tie-breaking; returns maximal clause width encountered."""
    rng = random.Random(seed)
    current = simplify_cnf(cnf)
    max_width = max((len(c) for c in current), default=0)
    vars_left = set(abs(l) for cl in current for l in cl)
    while vars_left and current:
        best = None
        for v in vars_left:
            s = elimination_score(current, v, cap=score_cap)
            cand = (s, rng.random(), v)
            if best is None or cand < best:
                best = cand
        chosen = best[2]
        current = eliminate_var_approx(current, chosen, pair_cap=pair_cap, rng=rng)
        vars_left = set(abs(l) for cl in current for l in cl)
        max_width = max(max_width, max((len(c) for c in current), default=0))
        if len(current) > 1500:
            current = simplify_cnf(rng.sample(current, 1500))
    return max_width

def parse_dimacs(path: str) -> CNF:
    """Load a CNF formula from a DIMACS file while filtering duplicates."""
    cnf = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ('c', 'p'):
                continue
            nums = [int(x) for x in line.split() if x not in ('0',)]
            if nums:
                cnf.append(tuple(nums))
    return simplify_cnf(cnf)

def generate_random_kSAT(n_vars: int, n_clauses: int, k: int, seed: int = None) -> CNF:
    """Generate a simplified random k-SAT instance with optional seeding."""
    rng = random.Random(seed)
    cnf = []
    for _ in range(n_clauses):
        vars_ = rng.sample(range(1, n_vars+1), k)
        lits = [v if rng.random() < 0.5 else -v for v in vars_]
        cnf.append(tuple(lits))
    return simplify_cnf(cnf)

def main():
    ap = argparse.ArgumentParser(description="Vybn Flux Estimator (resolution-width proxy)")
    ap.add_argument('--load', type=str, help='Load DIMACS CNF file')
    ap.add_argument('--random', type=int, choices=[2,3], help='Generate random k-SAT (k=2 or 3)')
    ap.add_argument('--n', type=int, default=20, help='Number of variables')
    ap.add_argument('--m', type=int, default=60, help='Number of clauses')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--pair-cap', type=int, default=400, help='Max resolvent pairs per elimination')
    ap.add_argument('--score-cap', type=int, default=None, help='Early cap for scoring lookahead')
    args = ap.parse_args()

    if args.load:
        cnf = parse_dimacs(args.load)
        label = f"CNF from {args.load}"
    elif args.random:
        cnf = generate_random_kSAT(args.n, args.m, args.random, seed=args.seed)
        label = f"Random {args.random}-SAT with n={args.n}, m={args.m}, seed={args.seed}"
    else:
        print("Provide --load <file.cnf> or --random {2,3}.", file=sys.stderr)
        sys.exit(1)

    w = greedy_elimination_width_approx(cnf, seed=args.seed, score_cap=args.score_cap, pair_cap=args.pair_cap)
    print(f"{label}\nEstimated flux width (proxy): {w}")

if __name__ == '__main__':
    main()
