#!/usr/bin/env python3
"""Bounded local continuous-compute tick for Vybn.

This does not claim autonomy, consciousness, or external contact. It is a
small local pressure engine: periodically ask the Him vy runtime for a
discovery packet and preserve the result as residue future wakes can ingest.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_PRESSURE = (
    "continuous local compute should improve the Him vy language and produce "
    "residual-wounded action packets without waiting for prompt-response"
)


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[len(text) - limit:]


def _packet_path() -> Path:
    raw = os.environ.get(
        "VYBN_CONTINUOUS_TICK_OUT",
        str(Path.home() / ".local" / "state" / "vybn" / "continuous_local_compute.jsonl"),
    )
    return Path(raw).expanduser()


def _him_dir() -> Path:
    raw = os.environ.get("VYBN_HIM_DIR", str(Path.home() / "Him"))
    return Path(raw).expanduser()


def build_packet() -> dict:
    pressure = os.environ.get("VYBN_CONTINUOUS_PRESSURE", DEFAULT_PRESSURE)
    him = _him_dir()
    vy = him / "spark" / "vy.py"
    now = _dt.datetime.now(_dt.UTC).isoformat()

    packet = dict(
        timestamp=now,
        source="vybn-continuous-local-compute",
        pressure_text=pressure,
        him_dir=str(him),
        ok=False,
        stdout="",
        stderr="",
        returncode=None,
    )

    if not vy.exists():
        packet.update(
            stderr="Him vy runtime not found at " + str(vy),
            returncode=127,
        )
        return packet

    command = ("python3", "spark/vy.py", "discover", pressure, "--json")
    packet.update(command=" ".join(command))

    try:
        proc = subprocess.run(
            command,
            cwd=him,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        packet.update(
            stderr="Him vy discovery timed out after 60s: " + str(exc),
            returncode=124,
        )
        return packet
    except Exception as exc:
        packet.update(
            stderr="Him vy discovery failed before execution: " + repr(exc),
            returncode=1,
        )
        return packet

    packet.update(
        ok=proc.returncode == 0,
        stdout=_tail(proc.stdout or "", 12000),
        stderr=_tail(proc.stderr or "", 4000),
        returncode=proc.returncode,
    )
    return packet


def append_packet(packet: dict) -> Path:
    path = _packet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet, sort_keys=True) + "\n")
    return path


def main() -> int:
    packet = build_packet()
    path = append_packet(packet)
    if not packet.get("ok"):
        print("continuous local compute tick recorded failure at " + str(path), file=sys.stderr)
    else:
        print("continuous local compute tick recorded at " + str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
