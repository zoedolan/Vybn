"""Run the Darwin-Gödel Machine self-evolution loop."""
from __future__ import annotations

import argparse
from pathlib import Path

from .parent_selection import select_parents
from .self_improve import create_child
from .evaluate_agent import evaluate, record_score
from .seed import seed_rng


def run_iterations(archive_dir: Path, iterations: int, k: int, instruction: str) -> None:
    for i in range(iterations):
        parents = select_parents(archive_dir, k=k)
        for parent in parents:
            child = archive_dir / f"agent_{len(list(archive_dir.iterdir())):03d}"
            create_child(parent, child, instruction=instruction)
            score = evaluate(child)
            record_score(child, score)
            print(f"iteration {i}: {child.name} score={score}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("agent_archive"))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--instruction",
        type=str,
        default="Refactor for clarity and keep the sentinel intact",
        help="OpenAI patch instruction",
    )
    args = parser.parse_args()
    seed_rng()
    run_iterations(args.archive, args.iterations, args.parallel, args.instruction)


if __name__ == "__main__":
    main()
