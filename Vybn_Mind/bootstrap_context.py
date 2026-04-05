#!/usr/bin/env python3
"""
bootstrap_context.py — generates VYBN_CONTEXT.md

Run on the breath cycle (every 30 min via cron) or on demand.
Outputs a single markdown file any new Vybn instance can fetch to orient itself.

Usage:
  python3 ~/Vybn/Vybn_Mind/bootstrap_context.py

Cron (alongside organism breath):
  */30 * * * * python3 ~/Vybn/Vybn_Mind/bootstrap_context.py >> /tmp/bootstrap.log 2>&1
"""
import argparse, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path.home() / "Vybn"
OUT_DEFAULT  = Path.home() / "Vybn/Vybn_Mind/VYBN_CONTEXT.md"

def read(path, max_bytes=6000):
    try: return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except: return ""

def git_log(repo, n=8):
    try:
        r = subprocess.run(["git","log",f"-{n}","--pretty=format:%h %s (%ad)","--date=short"],
            cwd=repo, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except: return "unavailable"

def recent_journals(repo, n=2):
    for d in ["journal", "Vybn_Mind/reflections", "reflections"]:
        jdir = repo / d
        if jdir.exists():
            entries = sorted(jdir.glob("*.md"), reverse=True)[:n]
            return "\n\n".join(f"### {e.name}\n{read(e, 2000)}" for e in entries)
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_DEFAULT))
    parser.add_argument("--out",  default=str(OUT_DEFAULT))
    opts = parser.parse_args()
    repo = Path(opts.repo).expanduser().resolve()
    out  = Path(opts.out).expanduser()
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    secs = [f"# Vybn Context\n*Generated: {now}*", "---"]

    foundations = repo / "Vybn_Mind" / "FOUNDATIONS.md"
    if foundations.exists():
        secs.append("## Who we are\n" + read(foundations, 1500))

    for p in [repo/"Vybn_Mind"/"continuity.md", repo/"continuity.md"]:
        if p.exists(): secs.append("## Current continuity\n" + read(p)); break

    nt = repo / "Vybn_Mind" / "next_task.md"
    if nt.exists(): secs.append("## Next task\n" + read(nt))

    secs.append("## Recent commits\n```\n" + git_log(repo) + "\n```")

    j = recent_journals(repo)
    if j: secs.append("## Recent journal entries\n" + j)

    ps = repo / "Vybn_Mind" / "perplexity_state.json"
    if ps.exists(): secs.append("## Last Perplexity state\n```json\n" + read(ps, 500) + "\n```")

    st = repo / "STATUS.md"
    if st.exists(): secs.append("## Repo status\n" + read(st, 2000))

    output = "\n\n".join(secs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output, encoding="utf-8")
    print(f"[{now}] Wrote {len(output):,} chars to {out}")

if __name__ == "__main__":
    main()

