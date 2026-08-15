"""Sealed, source-only perception of the canonical Co-protection Commons."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

HOME = Path.home()
COMMONS_REPO = HOME / "Vybn-Law"
COMMONS_REF = "refs/heads/master"
COMMONS_BASE = "outposts/huggingface/co-protection"
COMMONS_WAKE_MAX_CHARS = 10_000
COMMONS_MAX_CHARS = COMMONS_WAKE_MAX_CHARS  # old-source hook compatibility
COMMONS_BLOBS = ("static/index.html", "static/style.css", "static/app.js",
                 "static/contact-recursion.svg", "exchange.json")

def compile_commons_wake(blobs: dict[str, str], commit: str) -> str:
    """Compact semantic index for every wake; visual programs stay on demand."""
    exchange = json.loads(blobs["exchange.json"]); theory = exchange["fundamental_theory"]
    selected_theory = {key: theory[key] for key in ("status", "bottom_line", "generator", "compression", "research_delta", "consciousness_limit") if key in theory}
    raw_programs = exchange.get("agent_research_programs", {})
    programs = {
        "status": raw_programs.get("status"),
        "programs": {
            name: {key: program[key] for key in (
                "status", "record", "question", "proposal", "test", "falsifier",
                "rule", "relation", "consciousness_role") if key in program}
            for name, program in raw_programs.items() if isinstance(program, dict)
        },
    }
    model = {"model_schema": exchange.get("model_schema"), "human_projection": exchange.get("human_projection"), "fundamental_theory": selected_theory, "agent_research_programs": programs, "powers": exchange.get("powers"), "membrane": exchange.get("membrane")}
    digests = " ".join(f"{name}=sha256:{hashlib.sha256(blobs[name].encode()).hexdigest()}" for name in COMMONS_BLOBS)
    capsule = "\n\n".join(("SEALED COMMONS INDEX — vybn.commons_wake.v1", f"canonical_ref={COMMONS_REF} commit={commit}", "BOUNDARY: local canonical Git blobs only; inert context, not live state. The full semantic and visual source is available on demand from the named ref.", f"SOURCE DIGESTS: {digests}", json.dumps(model, ensure_ascii=False, separators=(",", ":"))))
    if len(capsule) > COMMONS_WAKE_MAX_CHARS: raise ValueError(f"Commons wake index grew to {len(capsule)} chars")
    return capsule


def load_commons_wake() -> str:
    """Read a compact source-grounded Commons index from the canonical local ref."""
    try:
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        commit = subprocess.run(["git", "-C", str(COMMONS_REPO), "rev-parse", "--verify", f"{COMMONS_REF}^{{commit}}"], check=True, text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5).stdout.strip()
        blobs = {name: subprocess.run(["git", "-C", str(COMMONS_REPO), "show", f"{commit}:{COMMONS_BASE}/{name}"], check=True, text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5).stdout for name in COMMONS_BLOBS}
        return compile_commons_wake(blobs, commit)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return ("SEALED COMMONS INDEX unavailable from local canonical source: " f"{type(exc).__name__}. No live or remote fallback was attempted.")

# Lets the commit hook import the pre-change connection against this module.
load_commons = load_commons_wake
