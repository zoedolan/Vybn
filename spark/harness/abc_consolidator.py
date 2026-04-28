"""ABC consolidator: thin executable metabolism over refactor perception.

Local-first runner. It does not invent doctrine; it calls the existing organs:
file-body visualization -> adaptive self-healing plan -> residual packet.

Default is dry-run because candidate mutation must be proposed from contact.
The point is to make ABC executable, not merely speakable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .refactor_perception import (
    adaptive_consolidation_plan_for,
    packet_for,
    render_repo_file_body_visualization,
    visualize_repo_file_bodies,
)


def _tracked(root: Path) -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [line for line in out.splitlines() if line.strip()]


def choose_candidate(root: Path, *, top_n: int = 80) -> str | None:
    viz = visualize_repo_file_bodies(root, tracked_paths=_tracked(root), top_n=top_n)
    packets = viz.pressures
    for pkt in packets:
        action = " ".join(getattr(pkt, "pressure", ()) or ())
        layer = "appendage" if getattr(pkt, "role", "") in {"generated exhaust", "runtime log", "archive/provenance candidate"} else "organ"
        path = getattr(pkt, "path", "")
        if layer == "appendage" and any(token in action for token in ("archive", "manifest", "ignore", "shell")):
            return path
    for pkt in packets:
        path = getattr(pkt, "path", "")
        role = getattr(pkt, "role", "")
        if path and "provenance" not in role and "personal-history" not in role:
            return path
    return None


def build_tick(root: Path, *, candidate: str | None = None, top_n: int = 80) -> dict:
    candidate = candidate or choose_candidate(root, top_n=top_n)
    visualization = render_repo_file_body_visualization(root, top_n=top_n)
    if not candidate:
        return {
            "status": "no_candidate",
            "root": str(root),
            "visualization": visualization,
            "next": "no safe candidate surfaced; refine classifier or widen contact",
        }

    proposed_change = "smallest consequential consolidation tick with restore path"
    pkt = packet_for(candidate, proposed_change=proposed_change)
    plan = adaptive_consolidation_plan_for(candidate, proposed_change)
    return {
        "status": "candidate",
        "root": str(root),
        "candidate": candidate,
        "proposed_change": proposed_change,
        "packet": pkt,
        "adaptive_plan": asdict(plan),
        "visualization": visualization,
        "next": "route candidate through references/imports/tests/restore-path checks before mutation",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one local-first ABC consolidation tick.")
    ap.add_argument("root", nargs="?", default=".", help="repo root")
    ap.add_argument("--candidate", help="explicit candidate path")
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)

    root = Path(ns.root).resolve()
    tick = build_tick(root, candidate=ns.candidate, top_n=ns.top_n)
    if ns.json:
        print(json.dumps(tick, indent=2, sort_keys=True))
    else:
        print(f"ABC tick: {tick['status']}")
        if tick.get("candidate"):
            print(f"candidate: {tick['candidate']}")
            print(f"next: {tick['next']}")
        print("\n--- visualization ---")
        print(tick.get("visualization", ""))
    return 0 if tick["status"] in {"candidate", "no_candidate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
