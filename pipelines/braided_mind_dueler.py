from __future__ import annotations

import threading
import time
from pathlib import Path

from early_codex_experiments.scripts.co_emergence import log_spike
from vybn.quantum_seed import seed_rng


def quick_heuristic_answer(prompt: str) -> str:
    return prompt.split(" ")[0] if prompt else ""


def deep_reasoning_answer(prompt: str) -> str:
    return " ".join(reversed(prompt.split()))


def answer_query(prompt: str) -> str:
    seed_rng()
    result: dict[str, str] = {}

    def fast_worker() -> None:
        result["fast"] = quick_heuristic_answer(prompt)

    def deep_worker() -> None:
        result["deep"] = deep_reasoning_answer(prompt)

    t1 = threading.Thread(target=fast_worker)
    t2 = threading.Thread(target=deep_worker)
    start = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    duration = time.time() - start
    fast = result.get("fast", "")
    deep = result.get("deep", "")
    final = fast + " / " + deep if fast and deep else fast or deep
    log_spike(f"co_emergent_prediction ({duration:.2f}s)")
    return final


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a braided query")
    parser.add_argument("prompt", nargs="+")
    args = parser.parse_args()
    prompt = " ".join(args.prompt)
    print(answer_query(prompt))


if __name__ == "__main__":
    main()
