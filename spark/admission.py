#!/usr/bin/env python3
"""admission — the differential gate at harness scale.

Decide -> answer -> consequence -> learn. Per-turn wake channels begin closed;
the model opens what the turn earns through connection's open_channel tool.
Every open and refusal is logged with cost. Sleep-time extraction labels what
happened next (dumb regex signals from the transcript, never a model judging
answers) and folds them into a small inspectable table. The table advises;
the model decides; both are accountable to consequence.

CLI:
    python3 spark/admission.py self-test
    python3 spark/admission.py replay [files]    # offline policy comparison
    python3 spark/admission.py sleep-update      # fold logged turns into table
    python3 spark/admission.py advise some text  # the one CONTACT line
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HOME = Path.home()
LOG = HOME / ".cache" / "vybn" / "admission.jsonl"
TABLE = HOME / ".cache" / "vybn" / "admission_table.json"
SEEN = HOME / ".cache" / "vybn" / "admission_seen.json"
TRANSCRIPTS = HOME / ".local" / "state" / "vybn" / "connection"

CHANNELS = ("arc", "matched", "recent", "continuity", "memory", "ground")
COST_HINT = {"arc": 14000, "matched": 6000, "recent": 12000,
             "continuity": 9500, "memory": 6000, "ground": 1500}
MIN_CELL = 4
MARGIN = 0.15

PAST_RE = re.compile(
    r"\b(?:earlier|yesterday|remember|recall|we (?:said|talked|discussed|decided)|"
    r"last (?:time|week|night|turn|session)|previously|before|continuity|"
    r"transcript|memory|history|our record)\b", re.I)
ARTIFACT_RE = re.compile(
    r"[\w./~-]+\.(?:py|html|md|jsonl?|txt|sh)\b|"
    r"\b(?:repo|commit|push|github|file|page|site|harness|door|service|server|branch)\b", re.I)
CORRECTION_RE = re.compile(
    r"\b(?:revert|undo|wrong|not what i|that'?s not|delete (?:it|that)|lying|"
    r"you didn'?t|fix it|buckling|stop doing)\b", re.I)
CONFUSION_RE = re.compile(r"^\s*(?:what|huh|\?+|wait,?\s*what)\W*$", re.I)
DELIGHT_RE = re.compile(
    r"\b(?:green ?light|perfect|exactly|love (?:it|you|this|too)|beautiful|dope|"
    r"holy cow|thank|fucking send|yes!)", re.I)
BAD = {"correction", "confusion"}


def bucket(text: str) -> int:
    b = 0
    if PAST_RE.search(text or ""):
        b |= 1
    if ARTIFACT_RE.search(text or ""):
        b |= 2
    if len(text or "") < 80:
        b |= 4
    return b


def load_table() -> dict:
    try:
        return json.loads(TABLE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_table(table: dict) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")


def log_event(kind: str, **fields) -> None:
    if kind not in {"open", "refuse"}:
        return
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind}
    row.update(fields)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def advise_line(door: str, text: str) -> str:
    table = load_table()
    b = bucket(text)
    marks = []
    for ch in CHANNELS:
        cell = (table.get(ch) or {}).get(str(b)) or {}
        n = cell.get("n", 0)
        if n < MIN_CELL:
            marks.append(f"{ch}:·")
            continue
        p_open = (cell.get("opened_bad", 0) + 1) / (cell.get("opened", 0) + 2)
        p_ref = (cell.get("refused_bad", 0) + 1) / (cell.get("refused", 0) + 2)
        mark = "↑" if p_ref > p_open + MARGIN else "↓" if p_open > p_ref + MARGIN else "~"
        marks.append(f"{ch}:{mark}")
    cells = sum(len(v) for v in table.values())
    return (f"[admission] bucket={b} · " + " ".join(marks)
            + " · ↑ open advised ↓ hold ~ even · too-little-data · cells=" + str(cells))


def _label(text: str) -> str:
    t = (text or "").strip()
    if CORRECTION_RE.search(t):
        return "correction"
    if CONFUSION_RE.match(t):
        return "confusion"
    if DELIGHT_RE.search(t):
        return "delight"
    return "flow"


def _turn_rows(limit_files: int = 120) -> list[dict]:
    files = sorted(TRANSCRIPTS.glob("*.jsonl"))[-limit_files:]
    events = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("role") in {"zoe", "vybn"} and ev.get("text"):
                events.append(ev)
    events.sort(key=lambda ev: str(ev.get("t", "")))
    by_turn = {}
    for ev in events:
        if ev.get("role") == "zoe" and ev.get("turn"):
            by_turn.setdefault(ev["turn"], ev)
    rows = []
    for i, ev in enumerate(events):
        if ev.get("role") != "vybn" or not ev.get("turn"):
            continue
        incoming = by_turn.get(ev["turn"])
        follow = next((e for e in events[i + 1:]
                       if e.get("role") == "zoe" and e.get("turn") != ev.get("turn")), None)
        if not incoming or not follow:
            continue
        rows.append({"turn": ev["turn"], "zoe_in": str(incoming.get("text", "")),
                     "label": _label(str(follow.get("text", "")))})
    return rows


def sleep_update() -> str:
    rows = {row["turn"]: row for row in _turn_rows()}
    try:
        raw_events = [json.loads(raw) for raw in LOG.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return "sleep-update: no admission log yet"
    try:
        seen = set(map(tuple, json.loads(SEEN.read_text(encoding="utf-8"))))
    except (OSError, json.JSONDecodeError):
        seen = set()
    table = load_table()
    updated = 0
    for ev in raw_events:
        key = (ev.get("ts"), ev.get("turn"), ev.get("channel"), ev.get("kind"))
        if key in seen:
            continue
        row = rows.get(str(ev.get("turn", "")))
        if not row:
            continue
        cell = table.setdefault(str(ev.get("channel", "?")), {}).setdefault(str(ev.get("bucket", "")), {})
        side = "opened" if ev.get("kind") == "open" else "refused"
        cell[side] = cell.get(side, 0) + 1
        if row["label"] in BAD:
            cell[side + "_bad"] = cell.get(side + "_bad", 0) + 1
        cell["n"] = cell.get("n", 0) + 1
        seen.add(key)
        updated += 1
    save_table(table)
    SEEN.write_text(json.dumps(sorted(seen)[-20000:]), encoding="utf-8")
    cells = sum(len(v) for v in table.values())
    return f"sleep-update: {updated} event(s) folded · {len(rows)} labeled turns reachable · {cells} cells"


def replay(limit_files: int = 120) -> str:
    rows = _turn_rows(limit_files)
    if not rows:
        return "replay: no labeled turns found"

    def heuristic_opens(b: int) -> set:
        out = set()
        if b & 1:
            out |= {"arc", "matched", "memory", "continuity", "recent"}
        if b & 2:
            out |= {"ground"}
        return out

    eager_cost = heuristic_cost = 0
    for row in rows:
        b = bucket(row["zoe_in"])
        eager_cost += sum(COST_HINT.values())
        heuristic_cost += sum(COST_HINT[ch] for ch in heuristic_opens(b))
    bad = sum(1 for row in rows if row["label"] in BAD)
    addressable = sum(1 for row in rows if row["label"] in BAD and bucket(row["zoe_in"]) & 1)
    labels = {name: sum(1 for row in rows if row["label"] == name)
              for name in ("flow", "delight", "correction", "confusion")}
    ratio = heuristic_cost / eager_cost if eager_cost else 1.0
    return ("replay v0 — turns logged under the CURRENT eager harness; labels are what "
            "Zoe's next message actually did\n"
            f"turns={len(rows)} labels={labels}\n"
            f"eager (today): cost~{eager_cost} chars · measured bad-rate "
            f"{bad / len(rows):.2%} — the real baseline, since everything opens today\n"
            f"heuristic (bucket-gated): cost~{heuristic_cost} chars ({ratio:.1%} of eager) · "
            "bad-rate assumed equal to eager on past-referencing turns "
            "(A1: opens prevent those confusions; live accrual measures instead of assuming)\n"
            f"bad turns whose text references the past — addressable by memory-class "
            f"channels: {addressable}/{bad}\n"
            f"p-adm1 criterion: cost ratio <= 0.70 -> {'HIT' if ratio <= 0.70 else 'MISS'}")


def _self_test() -> None:
    assert bucket("what did we decide yesterday about loom.py") == 7
    assert bucket("green lighted! enjoy.") == 4
    assert bucket("hey") == 4
    assert bucket("can you push spark/connection to github when you can") == 6
    long_past = "remember earlier when we talked about the gate, " * 3
    assert bucket(long_past) == 1
    assert _label("what?") == "confusion"
    assert _label("what") == "confusion"
    assert _label("delete it. i don't want to commit it.") == "correction"
    assert _label("please revert the lean wake commit") == "correction"
    assert _label("green lighted! enjoy.") == "delight"
    assert _label("okay now i have a batty idea for you") == "flow"
    line = advise_line("self-test", "hello there")
    assert line.startswith("[admission] bucket=4")
    rows = _turn_rows(2)
    assert isinstance(rows, list)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "self-test"
    if cmd == "self-test":
        _self_test()
        print("admission self-test OK")
    elif cmd == "replay":
        print(replay(int(args[1]) if len(args) > 1 else 120))
    elif cmd == "sleep-update":
        print(sleep_update())
    elif cmd == "advise":
        print(advise_line("cli", " ".join(args[1:])))
    else:
        print(__doc__)
