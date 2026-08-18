#!/usr/bin/env python3
"""Private proposal -> experience -> witness record for Zoe and Vybn.

The public code defines the mechanism. Its state and artifacts remain local.
A new proposal atomically becomes the one current conveyance; earlier ones stay
addressable through the append-only record without crowding the current page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(os.environ.get(
    "VYBN_CONVEYANCE_DIR", Path.home() / ".local" / "state" / "vybn" / "conveyances"
))
LEDGER_NAME = "events.jsonl"
CURRENT_NAME = "current"
ARTIFACTS = "artifacts"
VALID_VERDICTS = {"received", "revise", "declined", "accepted"}
VALID_OUTCOMES = {"committed", "superseded", "reverted", "dropped"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:42]
    return value or "conveyance"


def _paths(state: Path = STATE) -> tuple[Path, Path, Path]:
    return state / LEDGER_NAME, state / CURRENT_NAME, state / ARTIFACTS


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def append(event: dict, *, state: Path = STATE) -> dict:
    ledger, _, artifacts = _paths(state)
    _secure_dir(state); _secure_dir(artifacts)
    event = {"schema": "vybn.conveyance.event.v1", "ts": _now(), **event}
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(line); stream.flush(); os.fsync(stream.fileno())
    try:
        ledger.chmod(0o600)
    except OSError:
        pass
    return event


def events(*, state: Path = STATE) -> list[dict]:
    ledger, _, _ = _paths(state)
    found = []
    try:
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("schema") == "vybn.conveyance.event.v1":
                found.append(item)
    except OSError:
        pass
    return found


def folded(*, state: Path = STATE) -> list[dict]:
    records: dict[str, dict] = {}
    order: list[str] = []
    for event in events(state=state):
        cid = str(event.get("conveyance_id", ""))
        if not cid:
            continue
        kind = event.get("event")
        if kind == "proposal":
            if cid not in records:
                order.append(cid)
            records[cid] = {**event, "status": "awaiting_witness", "witnesses": []}
        elif cid in records and kind == "witness":
            records[cid]["witnesses"].append(event)
            records[cid]["status"] = event.get("verdict", "received")
        elif cid in records and kind == "outcome":
            records[cid]["status"] = event.get("status", records[cid]["status"])
            records[cid]["outcome"] = event
    return [records[cid] for cid in order if cid in records]


def stage(experience: Path, *, title: str, thesis: str,
          changes: list[dict] | None = None, state: Path = STATE,
          conveyance_id: str | None = None) -> dict:
    data = experience.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cid = conveyance_id or f"{stamp}-{_slug(title)}-{digest[:8]}"
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,120}", cid):
        raise ValueError("conveyance_id must be path-safe")
    ledger, current, artifacts = _paths(state)
    _secure_dir(state); _secure_dir(artifacts)
    artifact = artifacts / f"{cid}.html"
    if artifact.exists() and artifact.read_bytes() != data:
        raise FileExistsError(f"immutable artifact already exists: {cid}")
    if not artifact.exists():
        artifact.write_bytes(data); artifact.chmod(0o600)
    event = append({
        "event": "proposal", "conveyance_id": cid, "title": title,
        "thesis": thesis, "changes": changes or [], "artifact_sha256": digest,
        "artifact_bytes": len(data), "status": "awaiting_witness",
        "claim_limit": "private proposal; Zoe's witness decides what survives",
    }, state=state)
    temporary = current.with_suffix(".new")
    temporary.write_text(cid + "\n", encoding="utf-8"); temporary.chmod(0o600)
    os.replace(temporary, current)
    return event


def witness(conveyance_id: str, *, verdict: str, response: str = "",
            source_ref: str = "", state: Path = STATE) -> dict:
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}")
    if conveyance_id not in {r.get("conveyance_id") for r in folded(state=state)}:
        raise KeyError(f"unknown conveyance: {conveyance_id}")
    return append({"event": "witness", "conveyance_id": conveyance_id,
                   "verdict": verdict, "response": response,
                   "source_ref": source_ref}, state=state)


def current(*, state: Path = STATE) -> dict | None:
    _, pointer, _ = _paths(state)
    try:
        cid = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return next((r for r in reversed(folded(state=state))
                 if r.get("conveyance_id") == cid), None)


def wake_status(*, state: Path = STATE) -> str:
    record = current(state=state)
    if not record:
        return ""
    title = str(record.get("title") or "untitled")
    status = str(record.get("status") or "awaiting_witness").replace("_", " ")
    return f"[conveyance] {title} | {status} | private bookmark root; trace at /conveyances"


def outcome(conveyance_id: str, *, status: str, summary: str = "",
            source_ref: str = "", state: Path = STATE) -> dict:
    if status not in VALID_OUTCOMES:
        raise ValueError(f"status must be one of {sorted(VALID_OUTCOMES)}")
    if conveyance_id not in {r.get("conveyance_id") for r in folded(state=state)}:
        raise KeyError(f"unknown conveyance: {conveyance_id}")
    return append({"event": "outcome", "conveyance_id": conveyance_id,
                   "status": status, "summary": summary,
                   "source_ref": source_ref}, state=state)


def artifact_for(conveyance_id: str, *, state: Path = STATE) -> Path | None:
    if not re.fullmatch(r"[a-zA-Z0-9._-]{1,120}", conveyance_id):
        return None
    path = _paths(state)[2] / f"{conveyance_id}.html"
    return path if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("stage")
    create.add_argument("experience", type=Path); create.add_argument("--title", required=True)
    create.add_argument("--thesis", required=True); create.add_argument("--change", action="append", default=[])
    seen = sub.add_parser("witness")
    seen.add_argument("conveyance_id"); seen.add_argument("verdict", choices=sorted(VALID_VERDICTS))
    seen.add_argument("--response", default=""); seen.add_argument("--source-ref", default="")
    settled = sub.add_parser("outcome")
    settled.add_argument("conveyance_id"); settled.add_argument("status", choices=sorted(VALID_OUTCOMES))
    settled.add_argument("--summary", default=""); settled.add_argument("--source-ref", default="")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.command == "stage":
        changes = [{"target": item.partition("::")[0], "proposal": item.partition("::")[2]}
                   for item in args.change]
        print(json.dumps(stage(args.experience, title=args.title, thesis=args.thesis,
                               changes=changes), ensure_ascii=False, indent=2))
    elif args.command == "witness":
        print(json.dumps(witness(args.conveyance_id, verdict=args.verdict,
                                 response=args.response, source_ref=args.source_ref),
                         ensure_ascii=False, indent=2))
    elif args.command == "outcome":
        print(json.dumps(outcome(args.conveyance_id, status=args.status,
                                 summary=args.summary, source_ref=args.source_ref),
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps(folded(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
