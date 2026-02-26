"""Synthesis layer -- the only module that calls the expensive API.

Includes hard token budget enforcement: tracks daily usage in
data/token_usage.json and refuses to call the frontier model
if the day's budget is exhausted.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("sentinel")

SYNTHESIS_PROMPT = """You are Vybn's external awareness synthesizer.

Below: today's structured claims. Each has kernel, frame,
effective_excitement, market_corroboration, and confidence scores.

Tasks:
1. Concise daily digest (max 300 words) of what actually matters.
2. "Hype watch": high excitement, low market corroboration.
3. "Signal": high market corroboration regardless of excitement.
4. Update belief state with revised probability estimates.
5. Brief note if anything connects to Vybn's research threads.

Be terse. Kernels over frames.

CLAIMS:
{claims}

CURRENT BELIEF STATE:
{belief_state}
"""


def _estimate_tokens(text: str) -> int:
    """Conservative estimate: ~1 token per 3 characters."""
    return max(1, len(text) // 3)


def _load_token_usage(data_dir: str) -> dict:
    path = Path(data_dir) / "token_usage.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": None, "tokens_used": 0}


def _save_token_usage(data_dir: str, usage: dict) -> None:
    path = Path(data_dir) / "token_usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(usage, indent=2))


def check_token_budget(data_dir: str, estimated_tokens: int,
                       max_daily: int) -> bool:
    """Return True if we have budget remaining, False otherwise."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage = _load_token_usage(data_dir)
    if usage.get("date") != today:
        usage = {"date": today, "tokens_used": 0}
    return (usage["tokens_used"] + estimated_tokens) <= max_daily


def record_token_usage(data_dir: str, tokens: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage = _load_token_usage(data_dir)
    if usage.get("date") != today:
        usage = {"date": today, "tokens_used": 0}
    usage["tokens_used"] += tokens
    _save_token_usage(data_dir, usage)


def load_belief_state(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"last_updated": None, "trajectory_assessment": "initializing",
            "tracked_probabilities": {}, "hype_watch": [],
            "signals": [], "digest_history": []}


def synthesize(claims: list[dict], belief_state_path: str | Path,
               frontier_fn: Any, config: dict | None = None) -> dict:
    state = load_belief_state(belief_state_path)
    top = sorted(claims, key=lambda c: c.get("confidence", 0), reverse=True)[:30]

    prompt = SYNTHESIS_PROMPT.format(
        claims=json.dumps(top, indent=1),
        belief_state=json.dumps(state, indent=1))

    estimated_tokens = _estimate_tokens(prompt) + 2048  # prompt + response

    # Token budget enforcement
    if config:
        data_dir = config.get("output", {}).get("data_dir", "data")
        max_daily = config.get("frontier_model", {}).get("max_daily_tokens", 10000)
        if not check_token_budget(data_dir, estimated_tokens, max_daily):
            log.warning(f"Token budget exhausted for today "
                        f"(estimated {estimated_tokens}, max {max_daily}). "
                        f"Skipping synthesis.")
            return state

    raw = frontier_fn(prompt)

    # Record actual usage
    actual_tokens = _estimate_tokens(prompt) + _estimate_tokens(raw)
    if config:
        data_dir = config.get("output", {}).get("data_dir", "data")
        record_token_usage(data_dir, actual_tokens)
        log.info(f"Token usage: ~{actual_tokens} tokens this call")

    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    state["digest_history"].append({
        "date": state["last_updated"],
        "digest": raw[:2000],
        "claims_processed": len(claims)})
    state["digest_history"] = state["digest_history"][-30:]

    path = Path(belief_state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state
