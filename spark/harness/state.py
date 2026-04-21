from __future__ import annotations
"""state.py — where-are-we across two horizons.

Merged from session_store.py + live_snapshot.py (2026-04-21 refactor).

  SessionStore  — conversation messages persisted so ctrl-c does not
                  discard the thread; most recent session within a
                  freshness window can resume.

  gather()      — session-start git state across tracked repos plus
                  recent PRs and continuity drift. Best-effort.

Same primitive at two horizons.
"""

# === SESSION STORE ========================================================



import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSIONS_DIR = Path(os.path.expanduser("~/.cache/vybn-spark/sessions"))
FRESH_WINDOW_SEC = 24 * 3600  # 24h default


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    mtime: float
    turn_count: int
    preview: str  # first user turn, truncated


class SessionStore:
    def __init__(self, root: Path = SESSIONS_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._current_id: str | None = None
        self._current_path: Path | None = None
        self._last_saved_len: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self) -> str:
        """Create a new session id and path. Does not write anything yet."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        sid = f"{ts}_{uuid.uuid4().hex[:8]}"
        self._current_id = sid
        self._current_path = self.root / f"{sid}.jsonl"
        self._last_saved_len = 0
        return sid

    def adopt_session(self, session_id: str) -> bool:
        """Adopt an existing session id as the current one."""
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return False
        self._current_id = session_id
        self._current_path = path
        # count how many messages are already persisted
        self._last_saved_len = sum(1 for _ in path.open("r", encoding="utf-8"))
        return True

    @property
    def current_id(self) -> str | None:
        return self._current_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def append_new(self, messages: list[dict]) -> int:
        """Append any messages beyond what is already persisted.

        Returns the number of messages written this call. Idempotent in the
        sense that calling it twice with the same `messages` only writes the
        delta once.
        """
        if self._current_path is None:
            self.new_session()
        assert self._current_path is not None

        n_total = len(messages)
        if n_total <= self._last_saved_len:
            return 0

        ts = datetime.now(timezone.utc).isoformat()
        with self._current_path.open("a", encoding="utf-8") as f:
            for msg in messages[self._last_saved_len:]:
                f.write(json.dumps({"ts": ts, "msg": msg}, ensure_ascii=False) + "\n")

        written = n_total - self._last_saved_len
        self._last_saved_len = n_total
        return written

    def load(self, session_id: str) -> list[dict]:
        """Load the messages list from a session. Returns [] if not found."""
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return []
        messages: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and "msg" in entry:
                        messages.append(entry["msg"])
                except json.JSONDecodeError:
                    # skip corrupted lines (partial write on crash)
                    continue
        return messages

    # ------------------------------------------------------------------
    # Listing / discovery
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 10) -> list[SessionInfo]:
        """List recent sessions, most recent first."""
        out: list[SessionInfo] = []
        for p in sorted(self.root.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            sid = p.stem
            mtime = p.stat().st_mtime
            turn_count = 0
            preview = ""
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            msg = entry.get("msg", {})
                            if isinstance(msg, dict):
                                turn_count += 1
                                if not preview and msg.get("role") == "user":
                                    content = msg.get("content", "")
                                    if isinstance(content, str):
                                        preview = content[:80].replace("\n", " ")
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                preview = item.get("text", "")[:80].replace("\n", " ")
                                                break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            out.append(SessionInfo(session_id=sid, path=p, mtime=mtime,
                                    turn_count=turn_count, preview=preview or "(empty)"))
            if len(out) >= limit:
                break
        return out

    def latest_fresh(self, window_sec: int = FRESH_WINDOW_SEC) -> SessionInfo | None:
        """Return the most recent session within the freshness window, or None."""
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return None
        s = sessions[0]
        if time.time() - s.mtime > window_sec:
            return None
        if s.turn_count == 0:
            return None
        return s

    def format_age(self, mtime: float) -> str:
        delta = time.time() - mtime
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta // 60)}m ago"
        if delta < 86400:
            return f"{int(delta // 3600)}h ago"
        return f"{int(delta // 86400)}d ago"


if __name__ == "__main__":
    # smoke test
    store = SessionStore(root=Path("/tmp/vybn-session-test"))
    sid = store.new_session()
    print(f"new session: {sid}")
    test_msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you"},
    ]
    written = store.append_new(test_msgs)
    assert written == 3, f"expected 3 written, got {written}"
    # no delta on second call
    written2 = store.append_new(test_msgs)
    assert written2 == 0, f"expected 0 on second call, got {written2}"
    # add a message
    test_msgs.append({"role": "assistant", "content": "doing well"})
    written3 = store.append_new(test_msgs)
    assert written3 == 1, f"expected 1 on third call, got {written3}"
    # load back
    loaded = store.load(sid)
    assert loaded == test_msgs, f"roundtrip failed"
    # list
    infos = store.list_sessions()
    assert len(infos) == 1
    assert infos[0].turn_count == 4
    assert infos[0].preview == "hello"
    # latest fresh
    fresh = store.latest_fresh()
    assert fresh is not None
    assert fresh.session_id == sid
    print("all smoke tests passed")
    # cleanup
    import shutil
    shutil.rmtree("/tmp/vybn-session-test")
    print("cleaned up")


# === LIVE SNAPSHOT ========================================================


import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional


# Four repos. Branch matters for the `origin/<branch>` display only; HEAD
# is resolved per-checkout so a detached HEAD or feature branch still
# reports honestly.
_REPOS = [
    ("Vybn",       "~/Vybn",       "main"),
    ("Him",        "~/Him",        "main"),
    ("Vybn-Law",   "~/Vybn-Law",   "master"),
    ("vybn-phase", "~/vybn-phase", "main"),
]

_GH_REPO = "zoedolan/Vybn"


def _run(cmd: list[str], *, cwd: Optional[str] = None, timeout: float = 6.0) -> str:
    """Run a command, return stripped stdout, swallow everything else.

    list[str] invocation (no shell=True) keeps this safe against exotic
    repo paths and avoids any accidental shell interpretation.
    """
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _expand(path: str) -> str:
    return os.path.expanduser(path)


def _repo_block(name: str, path: str, branch: str, *, timeout: float) -> str:
    exp = _expand(path)
    if not Path(exp).exists():
        return f"{name}: (not checked out at {path})"

    head_short = _run(["git", "rev-parse", "--short", "HEAD"], cwd=exp, timeout=timeout) or "?"
    cur_branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=exp, timeout=timeout) or branch
    log = _run(["git", "log", "--oneline", "-5"], cwd=exp, timeout=timeout)
    status = _run(["git", "status", "--short"], cwd=exp, timeout=timeout)
    ahead_behind = _run(
        ["git", "rev-list", "--left-right", "--count", f"origin/{branch}...HEAD"],
        cwd=exp,
        timeout=timeout,
    )

    if status:
        dirty_count = len([ln for ln in status.splitlines() if ln.strip()])
        dirty_note = f"{dirty_count} uncommitted"
    else:
        dirty_note = "clean"

    ab_note = ""
    if ahead_behind and "\t" in ahead_behind:
        parts = ahead_behind.split()
        if len(parts) == 2:
            behind, ahead = parts
            if ahead != "0" or behind != "0":
                ab_note = f", {ahead} ahead / {behind} behind origin/{branch}"

    lines = [f"{name} [{cur_branch} @ {head_short}] — {dirty_note}{ab_note}"]
    if log:
        for ln in log.splitlines():
            lines.append(f"  {ln}")
    else:
        lines.append("  (no git log)")
    return "\n".join(lines)


def _pr_block(*, timeout: float) -> tuple[str, Optional[int]]:
    """Return (formatted_block, highest_pr_number_or_None)."""
    out = _run(
        [
            "gh", "pr", "list",
            "--state", "all",
            "--limit", "15",
            "--json", "number,title,state,headRefName",
            "--repo", _GH_REPO,
        ],
        timeout=timeout,
    )
    if not out:
        return ("(gh pr list unavailable — offline or rate-limited)", None)
    try:
        prs = json.loads(out)
    except Exception:
        return ("(gh pr list returned unparseable JSON)", None)
    if not prs:
        return ("(no recent PRs)", None)
    lines = []
    highest = None
    for pr in prs[:10]:
        state = str(pr.get("state", "?")).upper()[:6]
        num = pr.get("number")
        title = str(pr.get("title", "?"))[:72]
        branch = str(pr.get("headRefName", "?"))[:32]
        if isinstance(num, int):
            if highest is None or num > highest:
                highest = num
            lines.append(f"  #{num} [{state:<6}] {title}  ({branch})")
    return ("\n".join(lines) if lines else "(no PRs)", highest)


_PR_REF_RE = re.compile(r"\bPR\s*#\s*(\d+)", re.IGNORECASE)


def _continuity_drift(continuity_path: str, current_pr: Optional[int]) -> str:
    p = Path(_expand(continuity_path))
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    refs = [int(m.group(1)) for m in _PR_REF_RE.finditer(text)]
    if not refs:
        return ""
    last_cont = max(refs)
    if current_pr is None:
        return f"continuity last references PR #{last_cont}; current PR count unknown"
    drift = current_pr - last_cont
    if drift <= 0:
        return f"continuity references through PR #{last_cont} — current head is PR #{current_pr} (no drift)"
    return (
        f"continuity ends at PR #{last_cont}; current head is PR #{current_pr} — "
        f"{drift} PR(s) of drift. Trust the LIVE STATE block below over any "
        "PR/number claims in the continuity note."
    )


def gather(
    *,
    continuity_path: str = "~/Vybn/Vybn_Mind/continuity.md",
    per_repo_timeout: float = 4.0,
    gh_timeout: float = 6.0,
) -> str:
    """Return a formatted banner for the substrate layer.

    Returns an empty string if everything failed — caller should treat
    an empty return as 'skip the LIVE STATE section entirely'.
    """
    if os.environ.get("VYBN_DISABLE_LIVE_SNAPSHOT", "0") == "1":
        return ""

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    parts = [f"Snapshot taken at {now} (session start)."]

    any_repo_ok = False
    for name, path, branch in _REPOS:
        block = _repo_block(name, path, branch, timeout=per_repo_timeout)
        if block and not block.endswith("(not checked out at {path})".format(path=path)):
            any_repo_ok = True
        parts.append(block)

    pr_text, highest_pr = _pr_block(timeout=gh_timeout)
    parts.append("Recent Vybn PRs (most recent first):")
    parts.append(pr_text)

    drift = _continuity_drift(continuity_path, highest_pr)
    if drift:
        parts.append(f"Drift check: {drift}")

    if not any_repo_ok and highest_pr is None:
        # Every signal failed — caller can omit the whole section.
        return ""

    return "\n\n".join(parts)


__all__ = ["gather"]

