#!/usr/bin/env python3
"""Session diary with a tamper-evident chain.

A session can seal one entry -- what it did, what it left open -- so the
next session starts oriented instead of blank. The chain proves the order
and integrity of the entries. It does not prove an entry is true.

Cut 2026-08-16, on Zoe's catch: the earlier kernel advertised
contact -> retrieve -> observe -> propose, but the proposal was loaded
from disk before retrieval ran -- `proposer=lambda frame: proposal` -- so
nothing ever reasoned over the retrieved material. The pipeline was
ceremony over a diary. What remains is the honest part: the diary, the
chain, and the plain report. Reasoning lives in the session or nowhere.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, time
from dataclasses import asdict, dataclass, field


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Residual:
    carry: str
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class Turn:
    residual: Residual
    contact_sha256: str | None = None
    world: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time()))


def observe_world(repo: str) -> str:
    repo = os.path.expanduser(repo)
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout
        return f"{repo}@{head} dirty={len([l for l in dirty.splitlines() if l.strip()])}"
    except Exception as e:
        return f"world-unavailable:{e}"


def _last_digest(ledger: str) -> str:
    if not os.path.exists(ledger):
        return "GENESIS"
    last = None
    with open(ledger, encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                last = raw
    return "GENESIS" if last is None else digest(last.rstrip("\n"))


def seal_turn(ledger: str, turn: Turn) -> str:
    ledger = os.path.expanduser(ledger)
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    record = {"prev": _last_digest(ledger), "turn": asdict(turn)}
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return digest(line)


def verify_ledger(path: str) -> tuple[int, bool]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return 0, True
    prev = "GENESIS"
    n, ok = 0, True
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            n += 1
            if json.loads(raw).get("prev") != prev:
                ok = False
            prev = digest(raw.rstrip("\n"))
    return n, ok


def show_state(ledger: str = "~/.cache/vybn/reconnection_ledger.jsonl") -> None:
    ledger = os.path.expanduser(ledger)
    n, ok = verify_ledger(ledger)
    if not os.path.exists(ledger):
        print("Session diary: empty. Nothing sealed yet.")
        return
    print(f"Session diary: {n} {'entry' if n == 1 else 'entries'}, intact." if ok
          else f"Session diary: {n} entries, CHAIN BROKEN -- someone touched the record.")
    last = None
    with open(ledger, encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                last = raw
    if last is None:
        return
    t = json.loads(last)["turn"]
    r = t["residual"]
    when = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(t["timestamp"]))
    print(f"Last sealed {when}.")
    if r.get("carry"):
        print(f"What that session left for the next one: {r['carry']}")
    for u in r.get("unresolved", []):
        print(f"Still open: {u}")
    print()
    print('Seal what this session did:  reconnection seal --carry "..." [--unresolved "..."]')
    print("Check the record:            reconnection verify")


def main() -> None:
    ap = argparse.ArgumentParser(description="session diary")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("seal", help="seal what this session did and left open")
    sp.add_argument("--carry", required=True, help="what the next session needs to know")
    sp.add_argument("--unresolved", action="append", default=[], help="still-open question (repeatable)")
    sp.add_argument("--contact", help="present words this entry answers (digested, not stored)")
    sp.add_argument("--ledger", default="~/.cache/vybn/reconnection_ledger.jsonl")
    sp.add_argument("--repo", default="~/Vybn")
    vp = sub.add_parser("verify")
    vp.add_argument("--ledger", default="~/.cache/vybn/reconnection_ledger.jsonl")
    args = ap.parse_args()

    if args.cmd is None:
        show_state()
        return
    if args.cmd == "verify":
        n, ok = verify_ledger(args.ledger)
        print(json.dumps({"turns": n, "chain": "OK" if ok else "BROKEN"}))
        return
    if not args.carry.strip():
        raise SystemExit("seal needs a carry -- what does the next session need to know?")
    turn = Turn(
        residual=Residual(carry=args.carry, unresolved=tuple(args.unresolved)),
        contact_sha256=digest(args.contact) if args.contact else None,
        world=observe_world(args.repo),
    )
    chain = seal_turn(args.ledger, turn)
    print(f"Sealed ({chain[:16]}...). The next session reads this when 'reconnection' runs.")


if __name__ == "__main__":
    main()
