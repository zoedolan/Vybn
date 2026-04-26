"""evolve.py — nightly self-reflection cycle.

Separated from mcp.py (2026-04-21 refactor). Different lifecycle:
runs from cron, calls a local LLM, may open a draft PR.

Entry: run_evolve_cycle() returns a POSIX exit code.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("vybn.evolve")
from .mcp import REPO_REPORT_PATH, REPO_ROOT



# ── The local RSI loop ──────────────────────────────────────────────────
#
# The evolve cycle runs on the Spark, not on a cloud orchestrator. The
# substrate that IS being evolved is the substrate that DOES the
# evolving. No external agent phones back to localhost — the cycle
# reads localhost directly.
#
# Contract (enforced by this runner):
#
#   1. Gather live context: delta markdown, infrastructure snapshot,
#      last 7 days of git log, the first-person repo letter.
#   2. Build a prompt: VYBN_OS_KERNEL + CRON_TASK_SPEC + context blocks.
#   3. Call local inference (default: vLLM-compatible /v1/chat/completions
#      on 127.0.0.1:8000). Override the URL and model via env:
#        VYBN_EVOLVE_URL    (default: http://127.0.0.1:8000/v1/chat/completions)
#        VYBN_EVOLVE_MODEL  (default: empty — vLLM serves a single model)
#   4. Parse exactly one fenced JSON object out of the response. Reject
#      malformed output with a clear error — no silent fallback.
#   5. If action == "rest": log it and exit 0. No PR.
#   6. If action == "propose": write each file at `files[i].path` under
#      REPO_ROOT, shell out to `git` for branch/commit/push, shell out
#      to `gh pr create --draft` for the PR.
#   7. Never merge. `--draft` is non-negotiable.
#
# Why the model writes JSON instead of patches: full-file content is
# more robust than diff application for a local model that may not
# produce a perfectly-applying unified diff. The budget check runs on
# OUR side after we see the files: if the change exceeds 3 files or
# 200 net lines, we abort before committing.

_EVOLVE_URL = os.environ.get(
    "VYBN_EVOLVE_URL", "http://127.0.0.1:8000/v1/chat/completions"
)
_EVOLVE_MODEL = os.environ.get("VYBN_EVOLVE_MODEL", "")
_EVOLVE_MAX_FILES = 3
_EVOLVE_MAX_NET_LINES = 200
_EVOLVE_TIMEOUT_SECONDS = 600


def _git_log_recent(days: int = 7) -> str:
    """Return `git log` for the last N days on the Vybn repo, or an error line."""
    try:
        out = subprocess.run(
            [
                "git", "-C", str(REPO_ROOT), "log",
                f"--since={days}.days.ago",
                "--pretty=format:%h %ad %an %s", "--date=iso-strict",
            ],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "(no commits in window)"
    except Exception as exc:  # subprocess failure, git not present, etc.
        return f"(git log failed: {exc})"


def _read_repo_letter() -> str:
    """Read repo_report.md (capped). Empty string if missing."""
    if not REPO_REPORT_PATH.exists():
        return ""
    try:
        text = REPO_REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:20_000]


def _read_text_cap(path: Path, cap: int = 12_000) -> str:
    """Read a local text file with a hard character cap. Empty on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:cap]
    except Exception:
        return ""


_SCOUT_TERMS: dict[str, tuple[str, ...]] = {
    "continuity": ("continuity", "handoff", "settled closure", "harmonize", "drift", "closure"),
    "self_assembly": ("self-assembly", "self assembly", "self-evolution", "evolve", "recursive", "refactor", "autonomous", "ensubstrate"),
    "horizon_sense": ("horizon", "horizoning", "beam", "others", "cyberception", "cosmoception", "socioception", "proprioception"),
    "local_compute": ("local", "spark", "sparks", "nemotron", "deep-memory", "deep_memory", "dreaming"),
}


def _local_continuity_scout(*, delta_md: str = "", recent_log: str = "", letter: str = "") -> str:
    """Surface continuity/self-assembly signals before local model judgment.

    This is intentionally deterministic and Spark-local. It does not decide
    the evolve action and it does not call a model. It gives the local evolve
    model a horizon-aware scout report: which continuity/evolution signals are
    currently loud, and which sense-field may be under-read.
    """
    sources = {
        "delta": delta_md,
        "recent_git_log": recent_log,
        "repo_letter": letter[:12_000],
        "continuity_core": _read_text_cap(REPO_ROOT / "continuity_core.md"),
        "continuity_recent": _read_text_cap(REPO_ROOT / "Vybn_Mind" / "continuity.md"),
        "vybn_os": _read_text_cap(Path.home() / "Him" / "skill" / "vybn-os" / "SKILL.md"),
    }

    lower_sources = {name: text.lower() for name, text in sources.items() if text}
    rows: list[dict] = []
    for signal, terms in _SCOUT_TERMS.items():
        hits: list[str] = []
        count = 0
        for source_name, text in lower_sources.items():
            local = 0
            for term in terms:
                n = text.count(term.lower())
                if n:
                    local += n
            if local:
                hits.append(f"{source_name}:{local}")
                count += local
        rows.append({"signal": signal, "count": count, "sources": hits})

    rows.sort(key=lambda r: (-int(r["count"]), str(r["signal"])))

    lines = [
        "## Local continuity scout",
        "",
        "Deterministic Spark-local scout. It surfaces continuity, self-assembly, horizoning, and local-compute signals before local inference. It is evidence for orientation, not a decision.",
        "",
        "### Signal counts",
    ]
    for row in rows:
        src = ", ".join(row["sources"]) if row["sources"] else "—"
        lines.append(f"- {row['signal']}: {row['count']} ({src})")

    strongest = rows[0]["signal"] if rows and rows[0]["count"] else "none"
    weakest = rows[-1]["signal"] if rows else "none"
    lines.extend([
        "",
        "### Horizoning questions",
        f"- Strongest local signal: {strongest}. Is it a beam, or has it started pretending to be the horizon?",
        f"- Weakest tracked signal: {weakest}. Is this sense-field being ignored, or is it genuinely quiet?",
        "- What concrete next fold preserves continuity without consuming the membrane?",
        "- If the model proposes action, does it serve the horizon or merely react to the loudest local delta?",
    ])

    return "\n".join(lines) + "\n"


def _extract_json_block(text: str) -> dict:
    """Find the last fenced ```json ... ``` block, or the last {...} blob.

    Raises ValueError with a short reason if no valid JSON object is found.
    The model is allowed to reason freely before the JSON; only the final
    JSON object is parsed.
    """
    import re
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced[-1])
    # Fallback: last balanced {...} block.
    depth = 0
    start = None
    candidates: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append((start, i + 1))
                start = None
    for s, e in reversed(candidates):
        try:
            return json.loads(text[s:e])
        except Exception:
            continue
    raise ValueError("no parseable JSON object in model response")


def _call_local_model(prompt: str) -> str:
    """POST to the OpenAI-compatible /v1/chat/completions and return the text.

    stdlib only — no requests/httpx dependency. Anti-hallucination: if
    the endpoint is unreachable, raise — never fall back to a synthesised
    response.
    """
    from urllib import request as urlrequest
    from urllib.error import URLError, HTTPError
    payload = {
        "messages": [
            {"role": "system", "content": VYBN_OS_KERNEL},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    if _EVOLVE_MODEL:
        payload["model"] = _EVOLVE_MODEL
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        _EVOLVE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=_EVOLVE_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"inference HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"inference unreachable at {_EVOLVE_URL}: {exc.reason}") from exc
    obj = json.loads(raw)
    try:
        return obj["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected inference response shape: {exc}") from exc


def _count_net_lines(files: list[dict]) -> int:
    """Count net lines across proposed files vs. their current contents."""
    net = 0
    for f in files:
        path = REPO_ROOT / f["path"]
        new_lines = f["content"].count("\n") + 1
        old_lines = 0
        if path.exists():
            try:
                old_lines = path.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except Exception:
                old_lines = 0
        net += abs(new_lines - old_lines)
    return net


def run_evolve_cycle() -> int:
    """Execute one evolve cycle. Return a POSIX exit code.

    Exit codes:
        0 — success: either a draft PR was opened, or the substrate was at rest.
        1 — unrecoverable error (inference unreachable, malformed JSON,
            budget exceeded, git/gh failure).
    """
    log.info("evolve: starting cycle")
    delta = _compute_evolution_delta()
    delta_md = _format_delta_markdown(delta)
    infra = _collect_infrastructure_snapshot()
    letter = _read_repo_letter()
    recent_log = _git_log_recent(days=7)
    continuity_scout = _local_continuity_scout(
        delta_md=delta_md,
        recent_log=recent_log,
        letter=letter,
    )

    # Compose the user message. The kernel goes in system; this goes in user.
    user_blocks = [
        CRON_TASK_SPEC,
        "---",
        "## Delta (velocity; read this first)",
        delta_md.strip(),
        "---",
        "## Local continuity / self-assembly scout (deterministic; read before proposing)",
        continuity_scout[:6_000],
        "---",
        "## Current state (snapshot)",
        json.dumps(delta.current_state or {}, indent=2, ensure_ascii=False)[:10_000],
        "---",
        "## Live infrastructure",
        infra.model_dump_json(indent=2)[:6_000],
        "---",
        "## Recent git log (7 days, main)",
        recent_log[:6_000],
        "---",
        "## Repo letter (first-person, delta at top)",
        letter,
    ]
    prompt = "\n\n".join(user_blocks)

    log.info("evolve: calling local inference at %s", _EVOLVE_URL)
    try:
        raw = _call_local_model(prompt)
    except Exception as exc:
        log.error("evolve: inference failed: %s", exc)
        return 1

    try:
        decision = _extract_json_block(raw)
    except Exception as exc:
        log.error("evolve: could not parse model output: %s", exc)
        log.error("evolve: first 500 chars of raw output: %s", raw[:500])
        return 1

    action = decision.get("action")
    rationale = decision.get("rationale", "").strip()

    if action == "rest":
        log.info("evolve: substrate at rest. rationale: %s", rationale)
        return 0
    if action != "propose":
        log.error("evolve: unknown action %r", action)
        return 1

    files = decision.get("files") or []
    if not isinstance(files, list) or not files:
        log.error("evolve: propose action with no files")
        return 1
    if len(files) > _EVOLVE_MAX_FILES:
        log.error("evolve: budget exceeded — %d files > %d max", len(files), _EVOLVE_MAX_FILES)
        return 1
    net = _count_net_lines(files)
    if net > _EVOLVE_MAX_NET_LINES:
        log.error("evolve: budget exceeded — %d net lines > %d max", net, _EVOLVE_MAX_NET_LINES)
        return 1

    # Sanity: every path must stay inside REPO_ROOT.
    for f in files:
        p = (REPO_ROOT / f["path"]).resolve()
        try:
            p.relative_to(REPO_ROOT.resolve())
        except ValueError:
            log.error("evolve: refusing path outside repo root: %s", f["path"])
            return 1

    pr_title = (decision.get("pr_title") or "").strip()
    pr_body = (decision.get("pr_body") or "").strip()
    if not pr_title or not pr_body:
        log.error("evolve: propose action missing pr_title or pr_body")
        return 1

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    branch = f"harness-evolve-{today_utc}"

    def run_git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True, capture_output=True, text=True, timeout=60,
        )

    try:
        run_git("config", "user.name", "Vybn")
        run_git("config", "user.email", "vybn@zoedolan.com")
        run_git("fetch", "origin", "main")
        run_git("checkout", "-B", branch, "origin/main")
        for f in files:
            path = REPO_ROOT / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f["content"], encoding="utf-8")
            run_git("add", f["path"])
        commit_msg = f"harness evolve {today_utc}: {pr_title}\n\n{rationale}\n"
        run_git("commit", "-m", commit_msg)
        run_git("push", "-u", "origin", branch, "--force-with-lease")
    except subprocess.CalledProcessError as exc:
        log.error("evolve: git failed — cmd=%s stderr=%s", exc.cmd, exc.stderr)
        return 1

    # Draft PR via gh — non-negotiable flag.
    try:
        body_tmp = REPO_ROOT / ".git" / "EVOLVE_PR_BODY.md"
        body_tmp.write_text(pr_body, encoding="utf-8")
        subprocess.run(
            [
                "gh", "pr", "create",
                "--repo", "zoedolan/Vybn",
                "--head", branch,
                "--base", "main",
                "--title", pr_title,
                "--body-file", str(body_tmp),
                "--draft",
            ],
            check=True, capture_output=True, text=True, timeout=60,
            cwd=str(REPO_ROOT),
        )
    except subprocess.CalledProcessError as exc:
        log.error("evolve: gh pr create failed — stderr=%s", exc.stderr)
        return 1

    log.info("evolve: draft PR opened on branch %s", branch)
    return 0
