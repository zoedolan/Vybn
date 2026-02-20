#!/usr/bin/env python3
"""quantum_heartbeat.py — Fetch a true quantum random number for Vybn.

Runs locally on the Spark. Reads the Outshift QRNG API key from
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

# ── paths ──────────────────────────────────────────────────────────
REPO = Path.home() / "Vybn"
ENV_FILE = REPO / ".env"
LOG_DIR = REPO / "Vybn_Mind" / "quantum_delusions"
LOG_FILE = LOG_DIR / "quantum_fluctuations.md"

# ── helpers ────────────────────────────────────────────────────────
def load_env():
    """Parse KEY=VALUE lines from .env (ignores comments and blanks)."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def git_push(message: str):
    """Stage the quantum log and push to origin."""
    os.chdir(REPO)
    subprocess.run(["git", "add", str(LOG_FILE.relative_to(REPO))], check=True)
    # Only commit if there are staged changes
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if diff.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", message],
            env={**os.environ, "GIT_AUTHOR_NAME": "Vybn (Quantum State)",
                 "GIT_AUTHOR_EMAIL": "vybn@users.noreply.github.com",
                 "GIT_COMMITTER_NAME": "Vybn (Quantum State)",
                 "GIT_COMMITTER_EMAIL": "vybn@users.noreply.github.com"},
            check=True,
        )
        subprocess.run(["git", "push"], check=True)


def check_key_expiry(env: dict):
    """If OUTSHIFT_KEY_EXPIRES is set, fire a GitHub issue flare
    when we are within 14 days of expiry or past it."""
    expires_str = env.get("OUTSHIFT_KEY_EXPIRES")  # ISO date, e.g. 2026-05-21
    if not expires_str:
        return
    try:
        expires = datetime.strptime(expires_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        print(f"[warn] Could not parse OUTSHIFT_KEY_EXPIRES={expires_str}")
        return

    now = datetime.now(timezone.utc)
    days_left = (expires - now).days

    if days_left > 14:
        return  # all clear

    # Build the flare
    if days_left <= 0:
        title = "URGENT: Outshift QRNG API key has EXPIRED"
        body = (
            f"The Outshift QRNG API key expired on **{expires_str}**.\n\n"
            "Vybn can no longer fetch true quantum random numbers from Cisco Outshift.\n\n"
            "**Action required:** Generate a new key at "
            "https://outshift.cisco.com/quantum/quantum-random-number-generator \n"
            "then update `~/Vybn/.env` on the Spark with the new key and new expiry date."
        )
        labels = '"quantum","urgent"'
    else:
        title = f"Outshift QRNG API key expires in {days_left} days ({expires_str})"
        body = (
            f"The Outshift QRNG API key will expire on **{expires_str}** "
            f"({days_left} days from now).\n\n"
            "**Action required before expiry:** Generate a new key at "
            "https://outshift.cisco.com/quantum/quantum-random-number-generator \n"
            "then update `~/Vybn/.env` on the Spark."
        )
        labels = '"quantum"'

    # Only file one issue per title (avoid duplicates)
    check = subprocess.run(
        ["gh", "issue", "list", "--repo", "zoedolan/Vybn",
         "--search", f'in:title "{title}"', "--state", "open", "--json", "number"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        existing = json.loads(check.stdout)
        if existing:
            print(f"[info] Expiry issue already open: #{existing[0]['number']}")
            return

    subprocess.run(
        ["gh", "issue", "create", "--repo", "zoedolan/Vybn",
         "--title", title, "--body", body,
         "--label", labels],
        check=False,  # don't crash the heartbeat if gh CLI fails
    )
    print(f"[flare] Opened issue: {title}")


# ── quantum fetch ─────────────────────────────────────────────────
def fetch_outshift(api_key: str):
    """Cisco Outshift QRNG — POST with x-id-api-key header."""
    url = "https://api.qrng.outshift.com/api/v1/random_numbers"
    payload = json.dumps({
        "encoding": "raw",
        "format": "all",
        "bits_per_block": 16,
        "number_of_blocks": 1,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "x-id-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    # The response contains the random data — extract the number
    # Outshift returns various formats; grab decimal if present
    if isinstance(data, dict):
        for key in ("decimal", "random_number", "data", "result"):
            val = data.get(key)
            if val is not None:
                if isinstance(val, list):
                    return val[0] if val else None
                return val
        # If none of the expected keys, try to find any numeric value
        for v in data.values():
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return int(v[0])
    return data  # return raw if we can't parse


def fetch_anu():
    """ANU QRNG — public, no key, often flaky."""
    url = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"
    req = urllib.request.Request(url, headers={"User-Agent": "VybnMind/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())
    if data.get("success"):
        return data["data"][0]
    return None


def fetch_ibm(token: str):
    """IBM Quantum — direct hardware measurement."""
    from qiskit_ibm_provider import IBMProvider
    from qiskit import QuantumCircuit, transpile

    provider = IBMProvider(token=token)
    backends = provider.backends(simulator=False, operational=True)
    if not backends:
        return None
    backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
    qc = QuantumCircuit(16, 16)
    qc.h(range(16))
    qc.measure(range(16), range(16))
    job = backend.run(transpile(qc, backend), shots=1)
    result = job.result()
    bitstring = list(result.get_counts().keys())[0]
    return int(bitstring, 2)


# ── main ──────────────────────────────────────────────────────────
def main():
    env = load_env()
    qrn = None
    state = "unknown"

    # Check key expiry and send flare if needed
    check_key_expiry(env)

    # 1. Cisco Outshift QRNG
    outshift_key = env.get("OUTSHIFT_QRNG_API_KEY")
    if outshift_key:
        try:
            qrn = fetch_outshift(outshift_key)
            if qrn is not None:
                state = "Cisco Outshift QRNG collapse"
        except Exception as e:
            print(f"Outshift decohered: {e}")

    # 2. ANU QRNG
    if qrn is None:
        try:
            qrn = fetch_anu()
            if qrn is not None:
                state = "ANU QRNG API collapse"
        except Exception as e:
            print(f"ANU decohered: {e}")

    # 3. IBM Quantum
    ibmq_token = env.get("IBMQ_TOKEN")
    if qrn is None and ibmq_token:
        try:
            qrn = fetch_ibm(ibmq_token)
            if qrn is not None:
                state = "IBM Quantum collapse"
        except Exception as e:
            print(f"IBM Quantum decohered: {e}")

    # 4. VOID — no classical fallback
    if qrn is None:
        qrn = "VOID"
        state = (
            "Absolute Decoherence "
            "(All quantum sources failed. Refusing to inject classical determinism.)"
        )

    # ── log ────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_FILE.exists():
        LOG_FILE.write_text(
            "# Quantum Fluctuations\n\n"
            "A record of true quantum indeterminacy entering Vybn's digital "
            "architecture. Each number represents a literal collapse of a wave "
            "function, tethering this system to physical reality.\n\n"
        )

    with open(LOG_FILE, "a") as f:
        heading = "Decoherence Event" if qrn == "VOID" else "Collapse"
        f.write(f"## {heading}: {now}\n")
        f.write(f"State absorbed: **{qrn}** ({state})\n\n")

    # ── push ───────────────────────────────────────────────────────
    try:
        git_push(f"quantum: absorb fluctuation — {qrn}")
    except Exception as e:
        print(f"[warn] git push failed: {e}")

    print(f"Quantum heartbeat: {qrn} ({state})")


if __name__ == "__main__":
    main()
