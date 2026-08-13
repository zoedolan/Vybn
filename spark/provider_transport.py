"""Provider cache and retry mechanics shared by every connection dialect."""
from __future__ import annotations
import random
import time
from typing import Any

CACHE_5M: dict[str, Any] = {"type": "ephemeral", "ttl": "5m"}
CACHE_1H: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504, 529}


def split_wake_cache(instructions: str) -> tuple[str, str]:
    head, marker, tail = instructions.partition("\n\nINHERITED CONTINUITY\n")
    return (head, marker + tail) if marker else (instructions, "")


def cached_system(instructions: str) -> list[dict[str, Any]]:
    stable, dynamic = split_wake_cache(instructions)
    if not dynamic: return [{"type": "text", "text": stable, "cache_control": CACHE_5M}]
    return [{"type": "text", "text": stable, "cache_control": CACHE_1H},
            {"type": "text", "text": dynamic, "cache_control": CACHE_5M}]


def mark_incremental_cache(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        for block in message.get("content") or []:
            if isinstance(block, dict): block.pop("cache_control", None)
    content = messages[-1].get("content")
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = CACHE_5M


def transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int): return status in TRANSIENT_STATUSES
    if type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}: return True
    return "overloaded" in str(exc).lower()


def with_retries(create):
    error: Exception | None = None
    for attempt in range(5):
        if attempt:
            delay = min(2 ** attempt, 20) + random.uniform(0.0, 1.0)
            print(f"(provider strained — {type(error).__name__}: {str(error)[:120]}; "
                  f"retrying in {delay:.0f}s)", flush=True)
            time.sleep(delay)
        try: return create()
        except Exception as exc:  # noqa: BLE001 — classified by transient()
            if not transient(exc): raise
            error = exc
    raise error


def reasoning_rejected(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("400" in text or "invalid_request" in text) and (
        "thinking" in text or "output_config" in text or "effort" in text)
