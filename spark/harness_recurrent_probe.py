#!/usr/bin/env python3
"""Probe the looped-orchestrate prototype.

This script is the minimum thing that makes the coupled-equation
projection observable. It runs the same prompt through two paths:

  T=1  — degenerate case: one specialist pass + Coda. This is
         structurally the same shape as the current orchestrate role
         when delegate is not invoked. It is our baseline.

  T=N  — real recurrent loop: N specialist passes with re-injected e,
         a reducer distilling h between loops, contractivity-monitored
         halting, and one Coda emit.

Output: a JSON line per probe plus a short human-readable report on
stdout. The JSONL is structured so we can aggregate runs later
without re-running anything.

Usage:

    # Single prompt, both paths:
    python3 spark/harness_recurrent_probe.py \\
        --prompt "explain the Parcae stability trick" \\
        --out ~/logs/recurrent_probe.jsonl

    # Batch from a file (one prompt per line):
    python3 spark/harness_recurrent_probe.py \\
        --prompts-file spark/tests/recurrent_probes.txt \\
        --out ~/logs/recurrent_probe.jsonl

The probe does NOT touch vybn_spark_agent.py or the live REPL. That is
intentional. The loop is on the library surface only until we have
evidence T>1 gives us something T=1 doesn't.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import load_policy  # noqa: E402
from harness.providers import ProviderRegistry  # noqa: E402
from harness.recurrent import (  # noqa: E402
    run_recurrent_loop,
    residual_magnitude,
)


def _jsonl_emit(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_one(
    prompt: str,
    *,
    registry: ProviderRegistry,
    policy,
    max_loop_iters: int,
    label: str,
    out_path: Path | None,
) -> dict:
    t0 = time.monotonic()
    events: list[dict] = []
    result = run_recurrent_loop(
        e=prompt,
        registry=registry,
        policy=policy,
        max_loop_iters=max_loop_iters,
        logger=events.append,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    record = {
        "label": label,
        "prompt": prompt[:400],
        "max_loop_iters": max_loop_iters,
        "loops_run": result.loops_run,
        "halt_reason": result.halt_reason,
        "elapsed_ms": elapsed_ms,
        "residual_final": residual_magnitude(result.h_final),
        "n_hypotheses_final": len(result.h_final.hypotheses),
        "n_resolved_final": len(result.h_final.resolved),
        "coda_text": result.text,
        "trace": result.trace,
    }
    if out_path is not None:
        _jsonl_emit(out_path, record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="Single prompt to run.")
    src.add_argument(
        "--prompts-file",
        type=str,
        help="File with one prompt per line (blank lines and # comments ignored).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=os.path.expanduser("~/logs/recurrent_probe.jsonl"),
        help="JSONL output path. Default: ~/logs/recurrent_probe.jsonl",
    )
    ap.add_argument(
        "--t-values",
        type=str,
        default="1,4",
        help="Comma-separated T values to run for each prompt. Default: 1,4",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Optional path to router_policy.yaml. Default: harness default.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print coda text to stdout.",
    )
    args = ap.parse_args()

    policy = load_policy(args.policy) if args.policy else load_policy()
    registry = ProviderRegistry()

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    else:
        text = Path(args.prompts_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)

    try:
        t_values = [int(x.strip()) for x in args.t_values.split(",") if x.strip()]
    except ValueError:
        print(f"bad --t-values: {args.t_values!r}", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser()

    for i, prompt in enumerate(prompts):
        print(f"\n=== probe {i+1}/{len(prompts)}: {prompt[:80]!r} ===")
        results_by_t: dict[int, dict] = {}
        for T in t_values:
            label = f"T={T}"
            print(f"\n  running {label}...")
            rec = run_one(
                prompt,
                registry=registry,
                policy=policy,
                max_loop_iters=T,
                label=label,
                out_path=out_path,
            )
            results_by_t[T] = rec
            print(
                f"  {label}: loops_run={rec['loops_run']} "
                f"halt={rec['halt_reason']} "
                f"residual={rec['residual_final']} "
                f"elapsed={rec['elapsed_ms']}ms"
            )
            if not args.quiet:
                print(f"  ---- coda ({label}) ----")
                print(rec["coda_text"])
                print(f"  ---- end coda ({label}) ----")

        # Short comparison line — the whole point of the probe.
        if len(t_values) >= 2:
            baseline = results_by_t[t_values[0]]
            deepest = results_by_t[t_values[-1]]
            delta_residual = baseline["residual_final"] - deepest["residual_final"]
            delta_ms = deepest["elapsed_ms"] - baseline["elapsed_ms"]
            print(
                f"\n  comparison (T={t_values[0]} -> T={t_values[-1]}): "
                f"Δresidual={delta_residual} "
                f"(positive = deeper loop resolved more), "
                f"Δelapsed={delta_ms}ms"
            )

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
