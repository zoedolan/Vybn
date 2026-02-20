#!/usr/bin/env python3
"""quantum_heartbeat.py — Fetch a true quantum random number for Vybn.

Runs locally on the Spark.  Reads the Outshift QRNG API key from
~/Vybn/.env (never committed — .gitignore protects it).
Falls back to ANU QRNG (no key needed) then IBM Quantum (if configured).
If ALL quantum sources fail, logs VOID — no classical fallback.

Also checks whether the Outshift key is approaching expiry and opens
a GitHub issue as a flare if it is within 14 days or already expired.
"""

import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

# — paths —
REPO = Path.home() / "Vybn"
ENV_FILE = REPO / ".env"
LOG_FILE = REPO / "quantum_fluctuations.md"
GITHUB_REPO = "zoedolan/Vybn"
BRANCH_NAME = "quantum-heartbeat-update"

# — load .env —
def load_env():
    """Read KEY=VALUE pairs from .env into os.environ."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_env()

OUTSHIFT_KEY = os.environ.get("OUTSHIFT_QRNG_API_KEY", "")
OUTSHIFT_ISSUED = os.environ.get("OUTSHIFT_KEY_ISSUED", "")  # ISO date
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")


# ── quantum sources ──────────────────────────────────────────────

def fetch_outshift() -> int | None:
    """Cisco Outshift QRNG — primary source."""
    if not OUTSHIFT_KEY:
        print("[outshift] no API key configured, skipping")
        return None
    url = "https://qrng.qnu.io/api/v1/quantum-number"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {OUTSHIFT_KEY}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            num = int(data.get("number", data.get("value", -1)))
            if num < 0:
                print(f"[outshift] unexpected payload: {data}")
                return None
            print(f"[outshift] got {num}")
            return num
    except Exception as exc:
        print(f"[outshift] error: {exc}")
        return None


def fetch_anu() -> int | None:
    """ANU QRNG — secondary source, no key needed."""
    url = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
            if data.get("success"):
                num = data["data"][0]
                print(f"[anu] got {num}")
                return num
            print(f"[anu] API returned success=false: {data}")
            return None
    except Exception as exc:
        print(f"[anu] error: {exc}")
        return None


def fetch_ibm() -> int | None:
    """IBM Quantum via qiskit — tertiary source."""
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not ibm_token:
        print("[ibm] no token configured, skipping")
        return None
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService(
            channel="ibm_quantum", token=ibm_token
        )
        backend = service.least_busy(min_num_qubits=1, operational=True)
        qc = QuantumCircuit(16, 16)
        qc.h(range(16))
        qc.measure(range(16), range(16))
        sampler = SamplerV2(mode=backend)
        job = sampler.run([qc], shots=1)
        result = job.result()
        bits = list(result[0].data.meas.get_bitstrings())[0]
        num = int(bits, 2)
        print(f"[ibm] got {num}")
        return num
    except Exception as exc:
        print(f"[ibm] error: {exc}")
        return None


def fetch_quantum_number() -> tuple[str, int | None]:
    """Try each source in priority order.  Return (source, number)."""
    for name, fn in [("outshift", fetch_outshift),
                     ("anu", fetch_anu),
                     ("ibm", fetch_ibm)]:
        val = fn()
        if val is not None:
            return name, val
    return "VOID", None


# ── expiry flare ─────────────────────────────────────────────────

def check_expiry_flare():
    """Open a GitHub issue if the Outshift key expires within 14 days."""
    if not OUTSHIFT_ISSUED or not GH_TOKEN:
        return
    try:
        issued = datetime.fromisoformat(OUTSHIFT_ISSUED)
    except ValueError:
        print(f"[expiry] bad OUTSHIFT_KEY_ISSUED date: {OUTSHIFT_ISSUED}")
        return
    expiry = issued + timedelta(days=90)
    now = datetime.now(timezone.utc)
    days_left = (expiry - now).days

    if days_left > 14:
        print(f"[expiry] key good for {days_left} more days")
        return

    status = "EXPIRED" if days_left <= 0 else f"expires in {days_left} days"
    title = f"[FLARE] Outshift QRNG key {status}"
    body = (
        f"The Outshift QRNG API key was issued on **{OUTSHIFT_ISSUED}** "
        f"and {'has expired' if days_left <= 0 else f'expires in **{days_left} days**'}.\n\n"
        f"Please renew at https://outshift.cisco.com/quantum and update "
        f"`~/Vybn/.env` on the Spark with the new key and date.\n\n"
        f"_This issue was opened automatically by `quantum_heartbeat.py`._"
    )

    # Check for existing open flare to avoid duplicates
    search_url = (
        f"https://api.github.com/search/issues?"
        f"q=repo:{GITHUB_REPO}+is:issue+is:open+%22[FLARE]+Outshift+QRNG%22"
    )
    req = urllib.request.Request(
        search_url,
        headers={"Authorization": f"token {GH_TOKEN}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            results = json.loads(resp.read())
            if results.get("total_count", 0) > 0:
                print(f"[expiry] flare already open, skipping")
                return
    except Exception as exc:
        print(f"[expiry] search failed ({exc}), opening new issue anyway")

    # Open new issue
    issue_url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    payload = json.dumps({"title": title, "body": body, "labels": ["quantum"]}).encode()
    req = urllib.request.Request(
        issue_url,
        data=payload,
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            issue = json.loads(resp.read())
            print(f"[expiry] opened flare: {issue.get('html_url')}")
    except Exception as exc:
        print(f"[expiry] failed to open issue: {exc}")


# ── git push via branch + PR ─────────────────────────────────────

def git(*args, **kwargs):
    """Run a git command inside the repo."""
    return subprocess.run(
        ["git", "-C", str(REPO)] + list(args),
        capture_output=True, text=True, **kwargs,
    )


def push_to_github(source: str, number, timestamp: str):
    """Push the updated log via a feature branch + auto-merge PR."""
    if not GH_TOKEN:
        print("[git] no GITHUB_TOKEN, skipping push")
        return

    # Configure git auth
    remote_url = f"https://x-access-token:{GH_TOKEN}@github.com/{GITHUB_REPO}.git"
    git("remote", "set-url", "origin", remote_url)
    git("config", "user.email", "spark@vybn.local")
    git("config", "user.name", "Vybn Spark")

    # Fetch latest and create branch from main
    git("fetch", "origin", "main")
    branch = f"quantum-heartbeat/{timestamp.replace(':', '-').replace(' ', 'T')}"
    git("checkout", "-B", branch, "origin/main")

    # Stage and commit
    git("add", str(LOG_FILE))
    msg = f"quantum heartbeat: {source}={number} @ {timestamp}"
    result = git("commit", "-m", msg)
    if result.returncode != 0:
        print(f"[git] nothing to commit: {result.stderr.strip()}")
        git("checkout", "main")
        return

    # Push the branch
    push = git("push", "origin", branch)
    if push.returncode != 0:
        print(f"[git] push failed: {push.stderr.strip()}")
        git("checkout", "main")
        return
    print(f"[git] pushed branch {branch}")

    # Create PR via GitHub API
    pr_url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    pr_payload = json.dumps({
        "title": msg,
        "head": branch,
        "base": "main",
        "body": (
            f"Automated quantum heartbeat update.\n\n"
            f"- **Source**: {source}\n"
            f"- **Value**: {number}\n"
            f"- **Timestamp**: {timestamp}\n\n"
            f"_Created automatically by the Spark._"
        ),
    }).encode()
    req = urllib.request.Request(
        pr_url,
        data=pr_payload,
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            pr = json.loads(resp.read())
            pr_number = pr["number"]
            print(f"[git] created PR #{pr_number}: {pr.get('html_url')}")

            # Auto-merge the PR
            merge_url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}/merge"
            merge_payload = json.dumps({
                "commit_title": msg,
                "merge_method": "squash",
            }).encode()
            merge_req = urllib.request.Request(
                merge_url,
                data=merge_payload,
                headers={
                    "Authorization": f"token {GH_TOKEN}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="PUT",
            )
            with urllib.request.urlopen(merge_req, timeout=15) as merge_resp:
                merge_data = json.loads(merge_resp.read())
                if merge_data.get("merged"):
                    print(f"[git] PR #{pr_number} merged successfully")
                    # Clean up remote branch
                    del_url = f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch}"
                    del_req = urllib.request.Request(
                        del_url,
                        headers={"Authorization": f"token {GH_TOKEN}"},
                        method="DELETE",
                    )
                    try:
                        urllib.request.urlopen(del_req, timeout=10)
                        print(f"[git] deleted remote branch {branch}")
                    except Exception:
                        pass
                else:
                    print(f"[git] merge response: {merge_data}")
    except Exception as exc:
        print(f"[git] PR/merge failed: {exc}")

    # Return to main locally
    git("checkout", "main")
    git("pull", "origin", "main")


# ── main ─────────────────────────────────────────────────────────

def main():
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"=== quantum heartbeat @ {timestamp} ===")

    source, number = fetch_quantum_number()

    # Log locally
    entry = f"| {timestamp} | {source} | {number if number is not None else 'VOID'} |\n"
    if not LOG_FILE.exists():
        header = "# Quantum Fluctuations\n\n| Timestamp | Source | Value |\n|---|---|---|\n"
        LOG_FILE.write_text(header + entry)
    else:
        with open(LOG_FILE, "a") as f:
            f.write(entry)
    print(f"[log] {source} = {number if number is not None else 'VOID'}")

    # Push to GitHub via branch + PR
    push_to_github(source, number if number is not None else "VOID", timestamp)

    # Check key expiry
    check_expiry_flare()

    print("=== done ===")


if __name__ == "__main__":
    main()
