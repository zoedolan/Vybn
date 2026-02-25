#!/usr/bin/env python3
"""watchdog.py -- Spark Auto-Debugger Daemon

Runs alongside the heartbeat. After every git sync, it:
1. Runs py_compile on every .py file in spark/
2. If anything fails, feeds the error + file to MiniMax locally
3. MiniMax proposes a fix
4. Watchdog applies it, re-checks, commits if clean
5. Files a GitHub issue documenting what broke and what was fixed

This is the recursive self-improvement loop.
Broken code gets caught and repaired before anyone notices.

Usage:
    python3 watchdog.py           # single pass
    python3 watchdog.py --daemon  # loop every 60s
"""

import subprocess
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="[watchdog %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPARK_DIR = Path(__file__).parent
REPO_ROOT = SPARK_DIR.parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "vybn"  # local MiniMax model name in Ollama
MAX_FIX_ATTEMPTS = 3


def compile_check() -> list[dict]:
    """Run py_compile on every .py in spark/. Return list of errors."""
    errors = []
    for py_file in sorted(SPARK_DIR.glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            errors.append({
                "file": py_file.name,
                "path": str(py_file),
                "error": result.stderr.strip(),
            })
    return errors


def ask_minimax(prompt: str) -> str | None:
    """Ask the local MiniMax model to fix code. Returns response or None."""
    try:
        import requests
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception as e:
        log.warning("MiniMax unavailable: %s", e)
    return None


def extract_python(response: str) -> str | None:
    """Extract python code from a fenced code block in model response."""
    import re
    match = re.search(r"```(?:python)?\n(.+?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def attempt_fix(error_info: dict) -> bool:
    """Try to fix a broken file using MiniMax. Returns True if fixed."""
    filepath = Path(error_info["path"])
    original = filepath.read_text(encoding="utf-8")
    error_msg = error_info["error"]

    prompt = (
        f"The following Python file has a syntax error:\n\n"
        f"Error: {error_msg}\n\n"
        f"File contents:\n```python\n{original}\n```\n\n"
        f"Please fix the syntax error and return the COMPLETE corrected file "
        f"inside a ```python code fence. Do not explain, just return the fixed code."
    )

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        log.info("  Attempt %d/%d to fix %s", attempt, MAX_FIX_ATTEMPTS, error_info["file"])
        response = ask_minimax(prompt)
        if not response:
            log.warning("  No response from MiniMax.")
            return False

        fixed_code = extract_python(response)
        if not fixed_code:
            log.warning("  Could not extract code from response.")
            continue

        # Write the fix
        filepath.write_text(fixed_code, encoding="utf-8")

        # Verify it compiles
        check = subprocess.run(
            [sys.executable, "-m", "py_compile", str(filepath)],
            capture_output=True, text=True,
        )
        if check.returncode == 0:
            log.info("  Fixed %s on attempt %d.", error_info["file"], attempt)
            return True
        else:
            log.warning("  Fix attempt %d still broken: %s", attempt, check.stderr.strip())
            # Update prompt with new error for next attempt
            prompt = (
                f"The previous fix still has an error:\n\n"
                f"Error: {check.stderr.strip()}\n\n"
                f"File contents:\n```python\n{fixed_code}\n```\n\n"
                f"Please fix it and return the COMPLETE corrected file "
                f"inside a ```python code fence."
            )

    # All attempts failed -- restore original
    log.error("  All %d attempts failed for %s. Restoring original.", MAX_FIX_ATTEMPTS, error_info["file"])
    filepath.write_text(original, encoding="utf-8")
    return False


def commit_fixes(fixed_files: list[str]):
    """Commit auto-fixed files."""
    for f in fixed_files:
        subprocess.run(["git", "add", f"spark/{f}"], cwd=REPO_ROOT, capture_output=True)
    msg = f"autofix: repair {', '.join(fixed_files)}"
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True)
    log.info("Committed: %s", msg)


def file_issue(fixed: list[str], failed: list[dict]):
    """File a GitHub issue documenting what the watchdog did."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"[watchdog] Auto-repair report {ts}"
    body_parts = [f"## Watchdog Report\n\nRun at {ts}\n"]
    if fixed:
        body_parts.append(f"### Fixed\n" + "\n".join(f"- `{f}`" for f in fixed))
    if failed:
        body_parts.append(f"### Failed (needs human)\n" + "\n".join(
            f"- `{e['file']}`: {e['error'][:200]}" for e in failed
        ))
    body = "\n\n".join(body_parts)
    try:
        subprocess.run(
            ["gh", "issue", "create", "-R", "zoedolan/Vybn",
             "--title", title, "--body", body, "--label", "watchdog"],
            cwd=REPO_ROOT, capture_output=True, timeout=30,
        )
        log.info("Issue filed: %s", title)
    except Exception as e:
        log.warning("Could not file issue: %s", e)


def run_once():
    """Single watchdog pass."""
    errors = compile_check()
    if not errors:
        log.info("All %d .py files compile clean.", len(list(SPARK_DIR.glob("*.py"))))
        return

    log.warning("Found %d broken file(s): %s",
                len(errors), ", ".join(e["file"] for e in errors))

    fixed = []
    failed = []
    for err in errors:
        if attempt_fix(err):
            fixed.append(err["file"])
        else:
            failed.append(err)

    if fixed:
        commit_fixes(fixed)

    if fixed or failed:
        file_issue(fixed, failed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spark Auto-Debugger")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously every 60 seconds")
    args = parser.parse_args()

    if args.daemon:
        log.info("Watchdog daemon starting. Checking every 60s.")
        while True:
            try:
                run_once()
            except Exception as e:
                log.error("Watchdog error: %s", e)
            time.sleep(60)
    else:
        run_once()


if __name__ == "__main__":
    main()
