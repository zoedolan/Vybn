#!/usr/bin/env python3
"""Bounded circadian local-compute tick; no autonomy/contact/sleep claim."""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_PRESSURE = (
    "bounded circadian local compute should improve the Him vy language and "
    "produce residual-wounded action packets without waiting for prompt-response"
)


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[len(text) - limit:]


def _packet_path() -> Path:
    raw = os.environ.get("VYBN_CONTINUOUS_TICK_OUT", str(Path.home() / ".local" / "state" / "vybn" / "continuous_local_compute.jsonl"))
    return Path(raw).expanduser()


def _him_dir() -> Path:
    return Path(os.environ.get("VYBN_HIM_DIR", str(Path.home() / "Him"))).expanduser()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _in_utc_window(hour: int, start: int, end: int) -> bool:
    start, end, hour = start % 24, end % 24, hour % 24
    return False if start == end else (start <= hour < end if start < end else hour >= start or hour < end)


def circadian_phase(now: _dt.datetime | None = None) -> dict:
    utc = (now or _dt.datetime.now(_dt.UTC)).astimezone(_dt.UTC)
    start = _env_int("VYBN_CIRCADIAN_DREAM_START_UTC", 9)
    end = _env_int("VYBN_CIRCADIAN_DREAM_END_UTC", 15)
    enabled = _env_bool("VYBN_CIRCADIAN_ENABLED", True)
    dream = enabled and _in_utc_window(utc.hour, start, end)
    return {
        "enabled": enabled,
        "phase": "dream" if dream else "serve",
        "timestamp_utc": utc.isoformat(),
        "dream_window_utc": [start % 24, end % 24],
        "super_sleep_allowed": _env_bool("VYBN_CIRCADIAN_ALLOW_SUPER_SLEEP", False),
        "semantic_gate_required_before_serving": True,
        "public_outward_contact": "forbidden",
    }


def build_packet() -> dict:
    base_pressure = os.environ.get("VYBN_CONTINUOUS_PRESSURE", DEFAULT_PRESSURE)
    phase = circadian_phase()
    pressure = base_pressure + "\n\ncircadian phase: " + phase["phase"] + "; Super sleep remains operator-armed only."
    him = _him_dir()
    vy = him / "spark" / "vy.py"
    packet = dict(
        timestamp=phase["timestamp_utc"],
        source="vybn-continuous-local-compute",
        pressure_text=pressure,
        circadian=phase,
        him_dir=str(him),
        ok=False,
        stdout="",
        stderr="",
        returncode=None,
    )
    if not vy.exists():
        packet.update(stderr="Him vy runtime not found at " + str(vy), returncode=127)
        return packet

    command = ("python3", "-m", "spark.vy", "discover", pressure, "--json")
    packet.update(command=" ".join(command))
    try:
        proc = subprocess.run(command, cwd=him, text=True, capture_output=True, timeout=60, check=False)
        packet.update(ok=proc.returncode == 0, stdout=_tail(proc.stdout or "", 12000), stderr=_tail(proc.stderr or "", 4000), returncode=proc.returncode)
    except subprocess.TimeoutExpired as exc:
        packet.update(stderr="Him vy discovery timed out after 60s: " + str(exc), returncode=124)
    except Exception as exc:
        packet.update(stderr="Him vy discovery failed before execution: " + repr(exc), returncode=1)
    return packet


def append_packet(packet: dict) -> Path:
    path = _packet_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet, sort_keys=True) + "\n")
    return path


def main() -> int:
    packet = build_packet()
    if packet["circadian"]["phase"] == "dream":
        try:
            from spark.harness.refactor_perception import recursive_consolidation_pass
            packet["recursive_consolidation_ai"] = {"ok": True, "packet": recursive_consolidation_pass(max_candidates=25)}
        except Exception as exc:
            packet["recursive_consolidation_ai"] = {"ok": False, "error": repr(exc)}
    elif packet.get("ok"):
        try:
            from spark.harness.semantic_gate import local_super_semantic_gate; ok, reason = local_super_semantic_gate(base_url=os.environ.get("VYBN_SUPER_BASE_URL", "http://127.0.0.1:8000"), use_cache=False, precheck_models=True); packet.update(super_semantic_gate={"ok": ok, "reason": reason}, ok=ok)
        except Exception as exc: packet.update(super_semantic_gate={"ok": False, "reason": repr(exc)}, ok=False)
    path = append_packet(packet)
    print(("continuous local compute tick recorded" if packet.get("ok") else "continuous local compute tick recorded failure") + " at " + str(path), file=None if packet.get("ok") else sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
