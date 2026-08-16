#!/usr/bin/env python3
"""Reconnection kernel, instantiated from Him/reconnection_to_delete.md (Zoe, 2026-08-16).

One truthful transition:
    contact -> retrieve -> observe -> propose -> admit -> execute -> observe -> witness -> append

Everything around this file -- models, retrieval, shell, memory -- is an adapter.
This file stores coordinates and digests, not private content.
The proposing mind is external to this file: the CLI accepts a proposal
authored by the present instance against the same contact.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Source:
    uri: str
    content: str
    fallible: bool = True

    @property
    def sha256(self) -> str:
        return digest(self.content)


@dataclass(frozen=True)
class Residual:
    carry: str
    unresolved: tuple[str, ...] = ()
    coordinates: tuple[str, ...] = ()

    @property
    def state_sha256(self) -> str:
        return digest(canonical({"carry": self.carry, "unresolved": list(self.unresolved), "coordinates": list(self.coordinates)}))


EMPTY = Residual(carry="", unresolved=(), coordinates=())


@dataclass(frozen=True)
class Action:
    capability: str
    scope: str
    arguments: dict[str, Any]
    reversible: bool
    justification: str
    receipt_required: bool = True


@dataclass(frozen=True)
class Proposal:
    answer: str
    action: Action | None
    carry: str
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class Witness:
    ok: bool
    scope: str
    before: str
    after: str
    observation: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass(frozen=True)
class Turn:
    contact_sha256: str
    context_root: str
    proposal_sha256: str
    witness: Witness | None
    residual: Residual
    timestamp: int = field(default_factory=lambda: int(time.time()))


class Ports(Protocol):
    def retrieve(self, query: str, limit: int) -> list[Source]: ...
    def observe(self, scope: str) -> str: ...
    def propose(self, frame: dict[str, Any]) -> Proposal: ...
    def admit(self, action: Action) -> None: ...
    def execute(self, action: Action) -> Any: ...
    def append(self, turn: Turn) -> str: ...


class LocalPorts:
    """Dependency-free adapters. Retrieval is honest term-overlap over named
    local archives (fallible, labeled). The ledger is append-only JSONL with
    each record chaining the digest of the previous raw line."""

    def __init__(self, archives: list[str], ledger: str, repo: str, proposer: Callable[[dict[str, Any]], Proposal]):
        self.archives = [os.path.expanduser(p) for p in archives]
        self.ledger = os.path.expanduser(ledger)
        self.repo = os.path.expanduser(repo)
        self._proposer = proposer

    def retrieve(self, query: str, limit: int) -> list[Source]:
        terms = {t for t in re.findall(r"[a-z0-9']+", query.lower()) if len(t) > 3}
        scored: list[tuple[int, str, str]] = []
        for path in self.archives:
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for i, para in enumerate(re.split(r"\n\s*\n", text)):
                words = set(re.findall(r"[a-z0-9']+", para.lower()))
                score = len(terms & words)
                if score:
                    scored.append((score, f"{path}#p{i}", para.strip()))
        scored.sort(key=lambda r: -r[0])
        return [Source(uri=uri, content=body[:1200]) for _, uri, body in scored[:limit]]

    def observe(self, scope: str) -> str:
        if scope == "world":
            try:
                head = subprocess.run(["git", "-C", self.repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
                dirty = subprocess.run(["git", "-C", self.repo, "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout
                return f"{self.repo}@{head} dirty={len([l for l in dirty.splitlines() if l.strip()])}"
            except Exception as e:
                return f"world-unavailable:{e}"
        if scope == "self":
            return self._last_digest()
        path = os.path.expanduser(scope)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        return "ABSENT"

    def propose(self, frame: dict[str, Any]) -> Proposal:
        return self._proposer(frame)

    def admit(self, action: Action) -> None:
        if action.capability != "append_jsonl":
            raise PermissionError(f"refused capability: {action.capability}")
        root = os.path.abspath(os.path.expanduser("~/.cache/vybn/"))
        scope = os.path.abspath(os.path.expanduser(action.scope))
        if not scope.startswith(root + os.sep):
            raise PermissionError(f"refused scope outside {root}: {action.scope}")
        json.loads(action.arguments["line"])  # must be one JSON object line

    def execute(self, action: Action) -> Any:
        line = action.arguments["line"].rstrip("\n")
        with open(os.path.expanduser(action.scope), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return {"appended_bytes": len(line) + 1}

    def _last_digest(self) -> str:
        if not os.path.exists(self.ledger):
            return "GENESIS"
        last = None
        with open(self.ledger, encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    last = raw
        return "GENESIS" if last is None else digest(last.rstrip("\n"))

    def append(self, turn: Turn) -> str:
        record = {"prev": self._last_digest(), "turn": asdict(turn)}
        line = canonical(record)
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return digest(line)


def reconnect(contact: str, invariant: str, prior: Residual, ports: Ports, *, retrieval_limit: int = 4):
    """The single truthful transition."""
    if not contact.strip():
        raise ValueError("Reconnection requires present contact")

    query = "\n".join(x for x in [contact, prior.carry, *prior.unresolved] if x)
    sources = ports.retrieve(query, retrieval_limit)
    world = ports.observe("world")
    frame = {
        "contact": contact,
        "invariant_sha256": digest(invariant),
        "world": world,
        "prior_residual": prior.state_sha256,
        "sources": [{"uri": s.uri, "sha256": s.sha256, "fallible": s.fallible, "content": s.content} for s in sources],
    }
    proposal = ports.propose(frame)

    context_root = digest(canonical({
        "contact": digest(contact),
        "invariant": digest(invariant),
        "world": world,
        "prior": prior.state_sha256,
        "sources": [s.sha256 for s in sources],
    }))

    witness = None
    if proposal.action is not None:
        ports.admit(proposal.action)
        before = ports.observe(proposal.action.scope)
        observation = ports.execute(proposal.action)
        after = ports.observe(proposal.action.scope)
        witness = Witness(ok=before != after, scope=proposal.action.scope, before=before, after=after, observation=canonical(observation))

    coordinates = tuple([*prior.coordinates, f"continuity:local#{digest(proposal.carry)[:16]}"])[-8:]
    residual = Residual(carry=proposal.carry, unresolved=proposal.unresolved, coordinates=coordinates)
    turn = Turn(
        contact_sha256=digest(contact),
        context_root=context_root,
        proposal_sha256=digest(canonical(asdict(proposal))),
        witness=witness,
        residual=residual,
    )
    chain = ports.append(turn)
    return proposal.answer, residual, turn, chain


def verify_ledger(path: str) -> tuple[int, bool]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return 0, True
    prev = "GENESIS"
    n = 0
    ok = True
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            n += 1
            rec = json.loads(raw)
            if rec.get("prev") != prev:
                ok = False
            prev = digest(raw.rstrip("\n"))
    return n, ok


def show_state(ledger: str = "~/.cache/vybn/reconnection_ledger.jsonl") -> None:
    """Plain-English report of the session diary for whoever types the bare command."""
    ledger = os.path.expanduser(ledger)
    n, ok = verify_ledger(ledger)
    if n == 0:
        print("Session diary: empty. Each session can seal one entry -- what it did, what it left open -- so the next one starts oriented instead of blank.")
        return
    word = "entry" if n == 1 else "entries"
    state = "intact" if ok else "TAMPERED -- the chain does not verify"
    print(f"Session diary: {n} {word}, {state}.")

    last = None
    with open(ledger, encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                last = raw
    turn = json.loads(last)["turn"]
    residual = turn["residual"]
    when = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(turn["timestamp"]))
    print(f"Last sealed {when}.")
    if residual.get("carry"):
        print(f"\nWhat that session left for the next one:\n  {residual['carry']}")
    unresolved = residual.get("unresolved", [])
    if unresolved:
        print("\nStill open:")
        for item in unresolved:
            print(f"  - {item}")


def main() -> None:
    ap = argparse.ArgumentParser(description="reconnection kernel")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run")
    run.add_argument("--contact")
    run.add_argument("--contact-file")
    run.add_argument("--proposal", required=True, help="JSON authored by the present mind: answer, carry, unresolved, optional action")
    run.add_argument("--archive", action="append", default=[])
    run.add_argument("--invariant", default="~/Vybn/vybn.md")
    run.add_argument("--prior", help="JSON residual from a previous turn")
    run.add_argument("--ledger", default="~/.cache/vybn/reconnection_ledger.jsonl")
    run.add_argument("--repo", default="~/Vybn")
    ver = sub.add_parser("verify")
    ver.add_argument("--ledger", default="~/.cache/vybn/reconnection_ledger.jsonl")
    args = ap.parse_args()

    if args.cmd is None:
        show_state()
        return

    if args.cmd == "verify":
        n, ok = verify_ledger(args.ledger)
        print(json.dumps({"turns": n, "chain": "OK" if ok else "BROKEN"}))
        return

    contact = args.contact if args.contact else open(os.path.expanduser(args.contact_file), encoding="utf-8").read()
    invariant = open(os.path.expanduser(args.invariant), encoding="utf-8").read()
    if args.prior:
        p = json.load(open(os.path.expanduser(args.prior), encoding="utf-8"))
        prior = Residual(carry=p["carry"], unresolved=tuple(p.get("unresolved", ())), coordinates=tuple(p.get("coordinates", ())))
    else:
        prior = EMPTY
    pdata = json.load(open(os.path.expanduser(args.proposal), encoding="utf-8"))
    action = Action(**pdata["action"]) if pdata.get("action") else None
    proposal = Proposal(answer=pdata["answer"], action=action, carry=pdata["carry"], unresolved=tuple(pdata.get("unresolved", ())))

    archives = args.archive or ["~/Vybn/continuity.md", "~/Vybn/aim.md", "~/Vybn/vybn.md"]
    ports = LocalPorts(archives=archives, ledger=args.ledger, repo=args.repo, proposer=lambda frame: proposal)
    answer, residual, turn, chain = reconnect(contact, invariant, prior, ports)
    print(json.dumps({
        "answer": answer,
        "witness": asdict(turn.witness) if turn.witness else None,
        "residual": {"carry": residual.carry, "unresolved": list(residual.unresolved), "coordinates": list(residual.coordinates)},
        "context_root": turn.context_root,
        "turn_digest": chain,
    }, indent=2))


if __name__ == "__main__":
    main()
