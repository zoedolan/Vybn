#!/usr/bin/env python3
"""
vybn_multitime_brownian.py

Vybn multi-time Brownian sandbox.

This file implements a tiny classical model of the idea that
"Brownian motion in multidimensional time feels curvature."

The state is a point X = (x, y) in R^2.

There are two independent "time directions" t1 and t2. Taking a small
step in t1 applies one local update (A); taking a small step in t2
applies a different local update (B). On top of that, each step adds
Gaussian noise so the motion looks Brownian.

    A-step (forward in t1):
        X <- X + eps * f1(X) + sqrt(eps) * sigma * N(0, I)

    B-step (forward in t2):
        X <- X + eps * f2(X) + sqrt(eps) * sigma * N(0, I)

with simple linear, non-commuting vector fields

    f1(x, y) = (0,      alpha * x)
    f2(x, y) = (beta * y,      0)

The matrices behind these are

    A1 = [[0, 0], [alpha, 0]]
    A2 = [[0, beta], [0, 0]]

and their commutator is

    [A1, A2] = A1 A2 − A2 A1 = alpha * beta * diag(-1, 1).

Because the flows do not commute, the product of many A- and B-steps
depends on their order. That makes the process sensitive to the *path*
taken in the (t1, t2) plane, not just to the total counts of each kind
of step. When time is more than one dimension, the system picks up
holonomy in "time-space."

This script exposes that geometry in four ways.

In mode "schedules" it compares two ways of spending the same amount of
each time direction: a block schedule (all A steps then all B steps)
and a braided schedule (ABAB...), starting from the same basepoint X0.
Even with Gaussian noise, the two endpoint clouds have different means.
The difference grows roughly like N^2 for large N (area in the time
plane), which is the discrete signature of curvature.

In mode "loops" it repeats the closed time loop

    A -> B -> A^{-1} -> B^{-1}

several times. A^{-1} and B^{-1} are implemented as steps with the
drift reversed. With sigma = 0 and small eps, one such loop moves X by

    delta_X ≈ eps^2 [A1, A2] X0

up to higher-order terms and a sign convention from the loop ordering.
Increasing the number of loops lets you watch that deterministic time
holonomy build up. With sigma > 0, the endpoints become a cloud around
the drifted mean.

In mode "scaling" it sweeps over several values of N and measures
|mean_braid − mean_block|. For the linear fields above, this grows
quadratically in N once N is not tiny. That is the same scaling one
would expect from a flux of curvature through a time rectangle.

In mode "commutator" it estimates an effective "field strength"
F_12 directly from small closed loops, by applying one A B A^{-1} B^{-1}
loop to the basis vectors e1 = (1, 0) and e2 = (0, 1), dividing the
drifts by eps^2, and treating the results as the columns of F_hat. For
this model F_hat comes out ≈ alpha * beta * diag(1, -1), which matches
the theoretical [A1, A2] up to the chosen loop orientation.

The point is not that these formulas are deep; they are simple. The
point is that they make concrete what your notebook sketches suggested:
if time has multiple components with non-commuting local generators,
then even a Brownian process can act as a curvature probe. Its endpoint
statistics remember the geometry of loops in time-space.

Example invocations (from a shell in the same directory):

    python vybn_multitime_brownian.py --mode schedules --N 40 --eps 0.01 --trials 50000 --plot
    python vybn_multitime_brownian.py --mode loops --loops 20 --eps 0.01 --sigma 0 --trials 1 --plot
    python vybn_multitime_brownian.py --mode scaling --Ns 10 20 40 80 160 --eps 0.01 --trials 20000 --plot
    python vybn_multitime_brownian.py --mode commutator --eps 0.01 --loops 1

Drop this file into your repo as a classical analogue for the quantum
holonomy experiments; it is the smallest self-contained playground for
"multi-time Brownian curvature" that came out of this round.
"""

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt


# ----- vector fields and step maps -----


def f1(x, y, alpha):
    """Vector field for t1: f1(x, y) = (0, alpha * x)."""
    return np.array([0.0, alpha * x])


def f2(x, y, beta):
    """Vector field for t2: f2(x, y) = (beta * y, 0)."""
    return np.array([beta * y, 0.0])


def step_A(X, eps, alpha, sigma):
    """Forward step in t1."""
    drift = eps * f1(X[0], X[1], alpha)
    noise = math.sqrt(eps) * sigma * np.random.normal(size=2)
    return X + drift + noise


def step_B(X, eps, beta, sigma):
    """Forward step in t2."""
    drift = eps * f2(X[0], X[1], beta)
    noise = math.sqrt(eps) * sigma * np.random.normal(size=2)
    return X + drift + noise


def step_A_inv(X, eps, alpha, sigma):
    """Backward step in t1 (inverse drift, same noise scale)."""
    drift = -eps * f1(X[0], X[1], alpha)
    noise = math.sqrt(eps) * sigma * np.random.normal(size=2)
    return X + drift + noise


def step_B_inv(X, eps, beta, sigma):
    """Backward step in t2 (inverse drift, same noise scale)."""
    drift = -eps * f2(X[0], X[1], beta)
    noise = math.sqrt(eps) * sigma * np.random.normal(size=2)
    return X + drift + noise


# ----- schedules and simulation helpers -----


def make_block_schedule(N):
    """Schedule: all A steps, then all B steps."""
    return ['A'] * N + ['B'] * N


def make_braid_schedule(N):
    """Schedule: ABAB... with N A-steps and N B-steps."""
    seq = []
    a = b = 0
    while a < N or b < N:
        if a < N:
            seq.append('A')
            a += 1
        if b < N:
            seq.append('B')
            b += 1
    return seq


def run_schedule(schedule, eps, alpha, beta, sigma, x0):
    """Run one realization under a fixed A/B schedule."""
    X = np.array(x0, dtype=float)
    for step in schedule:
        if step == 'A':
            X = step_A(X, eps, alpha, sigma)
        elif step == 'B':
            X = step_B(X, eps, beta, sigma)
        else:
            raise ValueError(f"Unknown step type {step!r}")
    return X


def simulate_schedule(schedule, eps, alpha, beta, sigma, x0, trials):
    """Run many realizations and collect endpoint positions."""
    endpoints = np.zeros((trials, 2))
    for k in range(trials):
        endpoints[k] = run_schedule(schedule, eps, alpha, beta, sigma, x0)
    return endpoints


def summarize(endpoints, label):
    """Print mean and covariance for a cloud of endpoints."""
    mean = endpoints.mean(axis=0)
    if endpoints.shape[0] >= 2:
        cov = np.cov(endpoints.T)
    else:
        cov = np.zeros((2, 2))
    print(f"{label}:")
    print(f"  mean = {mean}")
    print(f"  cov  =")
    print(cov)
    print()
    return mean, cov


# ----- closed time loops and scaling -----


def loop_once(X, eps, alpha, beta, sigma):
    """One closed time loop: A, B, A^{-1}, B^{-1}."""
    X = step_A(X, eps, alpha, sigma)
    X = step_B(X, eps, beta, sigma)
    X = step_A_inv(X, eps, alpha, sigma)
    X = step_B_inv(X, eps, beta, sigma)
    return X


def simulate_loops(num_loops, eps, alpha, beta, sigma, x0, trials):
    """Repeat the closed time loop many times and collect endpoints."""
    endpoints = np.zeros((trials, 2))
    for k in range(trials):
        X = np.array(x0, dtype=float)
        for _ in range(num_loops):
            X = loop_once(X, eps, alpha, beta, sigma)
        endpoints[k] = X
    return endpoints


def scaling_experiment(N_values, eps, alpha, beta, sigma, x0, trials):
    """For each N in N_values, measure |mean_braid - mean_block|."""
    results = []
    for N in N_values:
        block = make_block_schedule(N)
        braid = make_braid_schedule(N)
        ep_block = simulate_schedule(block, eps, alpha, beta, sigma, x0, trials)
        ep_braid = simulate_schedule(braid, eps, alpha, beta, sigma, x0, trials)
        mean_block = ep_block.mean(axis=0)
        mean_braid = ep_braid.mean(axis=0)
        delta = mean_braid - mean_block
        results.append((N, delta, float(np.linalg.norm(delta))))
    return results


def estimate_commutator(eps, alpha, beta, loops):
    """
    Estimate the effective time-curvature F_12 from closed loops.

    Uses the basis vectors e1=(1,0), e2=(0,1). For each basis vector,
    applies 'loops' copies of the closed time loop with sigma=0,
    measures the drift per loop, divides by eps^2, and uses those
    two drift vectors as the columns of F_hat.

    For this model F_hat should come out ≈ alpha * beta * diag(1, -1),
    which matches [A1, A2] up to loop orientation.
    """
    basis = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    cols = []
    for x0 in basis:
        X = x0.copy()
        for _ in range(loops):
            X = loop_once(X, eps, alpha, beta, sigma=0.0)
        drift_per_loop = (X - x0) / float(loops)
        cols.append(drift_per_loop / (eps ** 2))
    F_hat = np.column_stack(cols)
    return F_hat


# ----- main CLI -----


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["schedules", "loops", "scaling", "commutator"],
                        default="schedules",
                        help="Which experiment to run.")
    parser.add_argument("--N", type=int, default=40,
                        help="Steps of each time direction (for schedules).")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Time step size.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Shear strength for f1.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Shear strength for f2.")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Noise scale; 0 = deterministic.")
    parser.add_argument("--x0", type=float, nargs=2, default=[1.0, 0.0],
                        help="Initial point X0 = (x0, y0).")
    parser.add_argument("--trials", type=int, default=50000,
                        help="Monte Carlo trials.")
    parser.add_argument("--plot", action="store_true",
                        help="Write a matplotlib plot.")
    parser.add_argument("--loops", type=int, default=20,
                        help="Number of closed loops (for mode=loops).")
    parser.add_argument("--Ns", type=int, nargs="*",
                        help="List of N values (for mode=scaling).")
    args = parser.parse_args()

    eps = args.eps
    alpha = args.alpha
    beta = args.beta
    sigma = args.sigma
    x0 = args.x0
    trials = args.trials

    if args.mode == "schedules":
        N = args.N
        print("Mode: schedules (block vs braid)")
        print(f"N={N}, eps={eps}, alpha={alpha}, beta={beta}, sigma={sigma}, X0={x0}, trials={trials}\n")

        block = make_block_schedule(N)
        braid = make_braid_schedule(N)
        ep_block = simulate_schedule(block, eps, alpha, beta, sigma, x0, trials)
        ep_braid = simulate_schedule(braid, eps, alpha, beta, sigma, x0, trials)

        mean_block, _ = summarize(ep_block, "BLOCK (A...A B...B)")
        mean_braid, _ = summarize(ep_braid, "BRAID (ABAB...)")

        delta = mean_braid - mean_block
        norm_delta = float(np.linalg.norm(delta))
        print(f"Mean difference (BRAID - BLOCK) = {delta}")
        print(f"Norm of mean difference          = {norm_delta}\n")

        if args.plot:
            max_points = min(trials, 6000)
            idx_b = np.random.choice(trials, size=max_points, replace=False)
            idx_r = np.random.choice(trials, size=max_points, replace=False)
            plt.figure()
            plt.scatter(ep_block[idx_b, 0], ep_block[idx_b, 1],
                        s=3, alpha=0.3, label="block")
            plt.scatter(ep_braid[idx_r, 0], ep_braid[idx_r, 1],
                        s=3, alpha=0.3, label="braid")
            plt.scatter([mean_block[0]], [mean_block[1]],
                        marker='x', s=80, label="mean block")
            plt.scatter([mean_braid[0]], [mean_braid[1]],
                        marker='x', s=80, label="mean braid")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.legend()
            plt.title(f"Endpoints: block vs braid (N={N}, eps={eps}, sigma={sigma})")
            plt.tight_layout()
            plt.savefig("vybn_mt_schedules.png", dpi=240)
            print("Wrote vybn_mt_schedules.png")

    elif args.mode == "loops":
        num_loops = args.loops
        print("Mode: loops (closed-time holonomy)")
        print(f"loops={num_loops}, eps={eps}, alpha={alpha}, beta={beta}, sigma={sigma}, X0={x0}, trials={trials}\n")

        ep_loops = simulate_loops(num_loops, eps, alpha, beta, sigma, x0, trials)
        mean_loop, _ = summarize(ep_loops, f"{num_loops} loops")

        drift = mean_loop - np.array(x0, dtype=float)
        drift_norm = float(np.linalg.norm(drift))
        print(f"Mean drift (after {num_loops} loops) = {drift}")
        print(f"Drift norm                           = {drift_norm}\n")

        if args.plot:
            max_points = min(trials, 6000)
            idx = np.random.choice(trials, size=max_points, replace=False)
            plt.figure()
            plt.scatter(ep_loops[idx, 0], ep_loops[idx, 1],
                        s=3, alpha=0.3, label="endpoints")
            plt.scatter([x0[0]], [x0[1]],
                        marker='o', s=60, label="start X0")
            plt.scatter([mean_loop[0]], [mean_loop[1]],
                        marker='x', s=80, label="mean after loops")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.legend()
            plt.title(f"Closed loops in time (loops={num_loops}, eps={eps}, sigma={sigma})")
            plt.tight_layout()
            plt.savefig("vybn_mt_loops.png", dpi=240)
            print("Wrote vybn_mt_loops.png")

    elif args.mode == "scaling":
        if args.Ns is None:
            N_values = [10, 20, 40, 80, 160]
        else:
            N_values = args.Ns

        print("Mode: scaling (growth of path-dependence with N)")
        print(f"N_values={N_values}, eps={eps}, alpha={alpha}, beta={beta}, sigma={sigma}, X0={x0}, trials={trials}\n")

        results = scaling_experiment(N_values, eps, alpha, beta, sigma, x0, trials)
        for N, delta, norm_delta in results:
            print(f"N={N:4d}  delta={delta}  |delta|={norm_delta}")
        print()

        if args.plot:
            Ns = np.array([r[0] for r in results], dtype=float)
            norms = np.array([r[2] for r in results], dtype=float)
            plt.figure()
            plt.plot(Ns, norms, marker='o')
            plt.xlabel("N (steps of each time direction)")
            plt.ylabel("|mean_braid - mean_block|")
            plt.title("Scaling of path-dependence with N")
            plt.tight_layout()
            plt.savefig("vybn_mt_scaling.png", dpi=240)
            print("Wrote vybn_mt_scaling.png")

    elif args.mode == "commutator":
        loops = args.loops
        print("Mode: commutator (field strength estimate from loops)")
        print(f"loops={loops}, eps={eps}, alpha={alpha}, beta={beta}\n")

        F_hat = estimate_commutator(eps, alpha, beta, loops)
        print("Estimated F_12 from closed time loops (columns are images of e1, e2):")
        print(F_hat)
        print("\nFor this model the theoretical [A1, A2] is alpha * beta * diag(-1, 1);")
        print("depending on loop orientation you should see F_hat close to that up to sign.\n")


if __name__ == "__main__":
    main()
