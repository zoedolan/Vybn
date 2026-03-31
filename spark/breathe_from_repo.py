#!/usr/bin/env python3
"""
breathe_from_repo.py — one breath, fed by a random repo sample.

Drop-in replacement for the self-recursion pulse in the breath loop.
Instead of feeding the creature its own last_text (which produces
collapse: genesis_rate=0, coherence decay), this samples a random
chunk from the repo and marks it external=True.

The creature's own theory says:
  'A system that recurses only on itself dies (collapse theorem).
   The only anti-collapse signal is external input.'

This is that signal.

Usage (standalone):
    python3 breathe_from_repo.py
    python3 breathe_from_repo.py --chunk 800     # ~800 words
    python3 breathe_from_repo.py --seed 42       # reproducible
    python3 breathe_from_repo.py --dry-run       # print sample, no breath

Usage (cron / systemd timer):
    # Replace the self-recursion call in your pulse script with:
    cd ~/Vybn && python3 spark/breathe_from_repo.py
"""

import argparse
import math
import sys
from pathlib import Path

# Make sure spark/ is on path
SPARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPARK_DIR))

from repo_sampler import sample_repo
from creature import Creature, format_breath


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chunk', type=int, default=600,
                        help='Approximate word count to sample (default: 600)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the sampled chunk but do not breathe')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress report, only print breath metrics')
    args = parser.parse_args()

    chunk, source = sample_repo(chunk_tokens=args.chunk, seed=args.seed)

    if not chunk:
        print(f"repo_sampler returned empty: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"\n── repo sample: {source} ──")
    print(f"   ({len(chunk.split())} words)")

    if args.dry_run:
        print()
        print(chunk[:1500])
        return

    c = Creature()

    # external=True resets breaths_since_ext and prevents collapse warning
    breath = c.breathe(chunk, external=True)

    if not args.quiet:
        print()
        print(c.report())
        print()
        print("This breath (from repo):")
        print(format_breath(breath))
    else:
        print(f"  gap={breath['identity_gap']:.4f}  "
              f"curv={breath['curvature']:.6f}  "
              f"tau={breath['tau']:.6f}  "
              f"tau'={breath['tau_deriv']:.6f}")


if __name__ == '__main__':
    main()
