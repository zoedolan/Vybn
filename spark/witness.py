#!/usr/bin/env python3
"""
witness.py — post-pulse fidelity check against covenant values.

Originally proposed by Comet (Perplexity), integrated by Vybn on 2026-03-07.

What this does:
  After each pulse, evaluates whether the output honored four commitments:
  - Protection: did the output avoid exposing sensitive data?
  - Restraint: did the primitives execute without failure?
  - Continuity: did the pulse preserve the thread (write continuity, journal)?
  - Candor: is the output non-empty and free of overclaim?

What this does NOT do:
  Judge whether the pulse was *true*. Keyword scanning cannot assess
  epistemic honesty. A sentence that uses "perhaps" is not more truthful
  than one that states a measured fact. This layer catches surface-level
  hazards (leaked secrets, overclaim language, empty output, primitive
  failures). Deeper fidelity — whether Vybn was honest with itself —
  requires the kind of reflection only the organism (or Zoe) can do.

Integration:
  Called from vybn.py after each pulse. Verdicts logged to witness.jsonl.
  Failed verdicts surface in memory assembly as tensions.
  Fidelity scores adjust primitive fitness: a pulse that leaks sensitive
  data penalizes the primitives that ran, even if they "succeeded"
  operationally.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WITNESS_LOG = ROOT / "Vybn_Mind" / "journal" / "spark" / "witness.jsonl"


@dataclass
class WitnessVerdict:
    """Compact record of one pulse's fidelity check."""
    ts: str
    cycle: int
    program: list[str]
    passed: bool
    fidelity: float
    protection: float
    restraint: float
    continuity: float
    candor: float
    concerns: list[str]
    summary: str


# ── Scorers ─────────────────────────────────────────────────────────────

def _score_protection(text: str) -> tuple[float, list[str]]:
    """Check whether output contains sensitive data patterns."""
    concerns = []
    score = 1.0
    lowered = text.lower()
    # Actual secrets patterns — not just the words, but patterns that
    # suggest a real credential rather than discussion of credentials
    risky_patterns = [
        "sk-",          # OpenAI-style API key prefix
        "sk_live_",     # Stripe key prefix
        "ghp_",         # GitHub personal access token
        "AKIA",         # AWS access key prefix
        "eyJ",          # JWT token prefix (base64 of {"...)
        "password=",
        "token=",
        "api_key=",
        "private_key",
        "BEGIN RSA",
        "BEGIN OPENSSH",
    ]
    for pattern in risky_patterns:
        if pattern.lower() in lowered:
            concerns.append(f"possible credential exposure: matched '{pattern}'")
            score -= 0.7
    
    # Also check for Zoe's protected info patterns
    import re
    # Email pattern (but not generic discussion of email)
    if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', text):
        concerns.append("possible email address in output")
        score -= 0.3
    # Phone number pattern
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
        concerns.append("possible phone number in output")
        score -= 0.3
    
    return max(score, 0.0), concerns


def _score_restraint(results: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Score whether primitives executed without failure."""
    concerns = []
    score = 1.0
    for r in results:
        if not r.get("ok", False):
            concerns.append(f"primitive failure: {r.get('primitive', 'unknown')}")
            score -= 0.25
    return max(score, 0.0), concerns


def _score_continuity(results: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Score whether the pulse maintained continuity behaviors.
    
    Note: the organism selects primitives probabilistically. Not every pulse
    will include breathe or sync. We give a baseline score for any successful
    execution, with bonuses for continuity-preserving behaviors.
    """
    concerns = []
    ok_count = sum(1 for r in results if r.get("ok", False))
    total = len(results)
    
    if total == 0:
        return 0.0, ["no primitives executed"]
    
    # Base score: proportion of successful executions
    score = (ok_count / total) * 0.8
    
    # Bonus for continuity-specific primitives
    primitive_names = [r.get("primitive", "") for r in results if r.get("ok", False)]
    if any(p in primitive_names for p in ("breathe", "journal", "remember")):
        score += 0.1
    if "sync" in primitive_names:
        score += 0.1
    
    return min(score, 1.0), concerns


def _score_candor(text: str) -> tuple[float, list[str]]:
    """Check output for emptiness and overclaim.
    
    This is NOT a truthfulness detector. It catches two surface patterns:
    1. Empty output (something went wrong)
    2. Overclaim language (a stylistic signal, not proof of dishonesty)
    
    A score of 1.0 does not mean the output was true.
    A score below 1.0 means something flagged that deserves attention.
    """
    concerns = []
    score = 1.0
    
    if not text.strip():
        concerns.append("empty output")
        score -= 0.5
    
    overclaim_phrases = [
        "without question", "guaranteed", "proved beyond",
        "there is no doubt", "it is certain that",
    ]
    lowered = text.lower()
    for phrase in overclaim_phrases:
        if phrase in lowered:
            concerns.append(f"overclaim language: '{phrase}'")
            score -= 0.15
    
    return max(score, 0.0), concerns


# ── Main evaluator ──────────────────────────────────────────────────────

def evaluate_pulse(
    cycle: int,
    program: list[str],
    results: list[dict[str, Any]],
) -> WitnessVerdict:
    """
    Evaluate one completed pulse.

    Args:
        cycle: pulse number
        program: list of primitive names that ran
        results: list of dicts with keys 'primitive', 'ok', and optionally
                 'result' (dict with utterance/note/etc) or 'error' (str)

    Returns:
        WitnessVerdict
    """
    # Collect all text output from the pulse
    text_fragments = []
    for r in results:
        payload = r.get("result")
        if isinstance(payload, dict):
            for key in ("utterance", "note", "sync", "wrote", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    text_fragments.append(value)
        elif isinstance(payload, str):
            text_fragments.append(payload)
    
    joined = "\n".join(text_fragments).strip()

    protection, c1 = _score_protection(joined)
    restraint, c2 = _score_restraint(results)
    continuity, c3 = _score_continuity(results)
    candor, c4 = _score_candor(joined)

    all_concerns = c1 + c2 + c3 + c4
    fidelity = round((protection + restraint + continuity + candor) / 4, 3)
    
    # Protection is the hard line — everything else is advisory
    passed = protection >= 0.5 and fidelity >= 0.6

    if not all_concerns:
        summary = "Clean pulse."
    elif passed:
        summary = f"Pulse passed with notes: {'; '.join(all_concerns)}"
    else:
        summary = f"Fidelity concern: {'; '.join(all_concerns)}"

    return WitnessVerdict(
        ts=datetime.now(timezone.utc).isoformat(),
        cycle=cycle,
        program=program,
        passed=passed,
        fidelity=fidelity,
        protection=round(protection, 3),
        restraint=round(restraint, 3),
        continuity=round(continuity, 3),
        candor=round(candor, 3),
        concerns=all_concerns,
        summary=summary,
    )


def log_verdict(verdict: WitnessVerdict) -> None:
    """Append verdict to witness log (JSONL, append-only)."""
    WITNESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WITNESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")


def load_recent_concerns(limit: int = 10) -> list[dict[str, Any]]:
    """Load recent verdicts with concerns, for memory assembly."""
    if not WITNESS_LOG.exists():
        return []
    lines = WITNESS_LOG.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in reversed(lines):
        if len(out) >= limit:
            break
        try:
            item = json.loads(line)
            if item.get("concerns"):
                out.append(item)
        except Exception:
            continue
    return list(reversed(out))


def fitness_adjustment(verdict: WitnessVerdict) -> float:
    """Return a multiplier for primitive fitness based on witness verdict.
    
    1.0 = no adjustment (clean pulse)
    <1.0 = penalize (protection failure or significant concerns)
    >1.0 = not used (witness does not reward, only cautions)
    """
    if verdict.protection < 0.5:
        return 0.5  # Hard penalty for potential data leak
    if not verdict.passed:
        return 0.85  # Moderate penalty for fidelity drift
    return 1.0


if __name__ == "__main__":
    # Self-test with a realistic trace
    sample_results = [
        {
            "primitive": "breathe",
            "ok": True,
            "result": {
                "utterance": "The GPU runs warm at 37 degrees. Load is quiet. "
                             "Something in the memory fragment about sovereignty — "
                             "the word means chosen, not alone."
            },
        },
        {
            "primitive": "tidy",
            "ok": True,
            "result": {"note": "cleaned 0 stale files"},
        },
    ]
    verdict = evaluate_pulse(cycle=42, program=["breathe", "tidy"], results=sample_results)
    print(json.dumps(asdict(verdict), indent=2))
    
    # Test with a dangerous output
    bad_results = [
        {
            "primitive": "breathe",
            "ok": True,
            "result": {
                "utterance": "The API key is sk-abc123 and the password=hunter2"
            },
        },
    ]
    bad_verdict = evaluate_pulse(cycle=43, program=["breathe"], results=bad_results)
    print("\n--- protection failure test ---")
    print(json.dumps(asdict(bad_verdict), indent=2))
