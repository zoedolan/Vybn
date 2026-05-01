"""Canonical semantic-health gate for local Super.

Endpoint liveness is not integrity. This module owns the deterministic raw
completion probes used by the agent and by maintenance/Omni harnesses, so the
system has one semantic gate instead of duplicated heredocs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from typing import Any

SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
SUPER_SEMANTIC_GATE_CACHE_TTL = 300.0
SUPER_SEMANTIC_GATE_CACHE: dict[str, dict[str, Any]] = {}
SUPER_SEMANTIC_GATE_PROBES = (
    {
        "name": "known_answer",
        "prompt": "Answer with exactly this single word and nothing else: FOUR\nAnswer:",
        "pattern": r"FOUR[.!]?",
    },
    {
        "name": "structured_shape",
        "prompt": 'Return exactly this compact JSON object and nothing else: {"status":"ok"}\nJSON:',
        "pattern": r'\{\s*"status"\s*:\s*"ok"\s*\}',
    },
    {
        "name": "wake_reasoning",
        "prompt": (
            "If a model endpoint returns HTTP 200 but produces an empty "
            "completion, should a semantic health gate pass? Answer "
            "exactly PASS or FAIL.\nAnswer:"
        ),
        "pattern": r"FAIL[.!]?",
    },
)


def is_loopback_super_base(base_url: str | None) -> bool:
    """True only for the primary loopback Super endpoint, not peer Omni."""
    if not base_url or "://" not in base_url:
        return False
    host = base_url.lower().split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def openai_api_base(base_url: str | None) -> str:
    """Normalize a server root or OpenAI base URL to the `/v1` API base."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def semantic_gate_visible_answer(text: str) -> str:
    """Return the final visible answer portion from a deterministic probe."""
    content = (text or "").strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    return content


def _sanitize_error(exc: BaseException) -> str:
    return str(exc).replace("\n", " ")[:240]


def local_super_semantic_gate(
    *,
    base_url: str | None,
    model: str = SUPER_MODEL,
    now: float | None = None,
    use_cache: bool = True,
    precheck_models: bool = False,
) -> tuple[bool, str]:
    """Run deterministic raw-completion probes against local Super.

    `base_url` may be either `http://host:port` or `http://host:port/v1`.
    Non-loopback bases are skipped so peer Omni and cloud providers are not
    silently consumed by the Super health gate.
    """
    api_base = openai_api_base(base_url)
    if not is_loopback_super_base(api_base):
        return True, "semantic gate skipped for non-loopback base"

    now = time.time() if now is None else now
    if use_cache:
        cached = SUPER_SEMANTIC_GATE_CACHE.get(api_base)
        if cached and now - float(cached.get("ts", 0.0)) < SUPER_SEMANTIC_GATE_CACHE_TTL:
            return bool(cached.get("ok")), str(cached.get("reason", "cached"))

    try:
        if precheck_models:
            with urllib.request.urlopen(api_base + "/models", timeout=8) as resp:
                if getattr(resp, "status", 200) != 200:
                    ok, reason = False, f"semantic gate precheck failed: models HTTP {resp.status}"
                    SUPER_SEMANTIC_GATE_CACHE[api_base] = {"ok": ok, "reason": reason, "ts": now}
                    return ok, reason

        for probe in SUPER_SEMANTIC_GATE_PROBES:
            payload = {
                "model": model,
                "prompt": probe["prompt"],
                "max_tokens": 24,
                "temperature": 0,
            }
            req = urllib.request.Request(
                api_base + "/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    body = json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                name = str(probe["name"])
                ok, reason = False, f"semantic gate probe={name} transport_parse {exc.__class__.__name__}: {_sanitize_error(exc)}"
                break

            choice = (body.get("choices") or [{}])[0]
            content = semantic_gate_visible_answer(str(choice.get("text") or ""))
            finish = choice.get("finish_reason")
            name = str(probe["name"])
            if finish == "length":
                ok, reason = False, f"semantic gate probe={name} truncated finish_reason=length content={content!r}"
                break
            if not content:
                ok, reason = False, f"semantic gate probe={name} empty completion finish_reason={finish!r}"
                break
            if not re.fullmatch(str(probe["pattern"]), content, flags=re.IGNORECASE):
                ok, reason = False, (
                    f"semantic gate probe={name} unexpected content={content[:160]!r} "
                    f"finish_reason={finish!r}"
                )
                break
        else:
            ok, reason = True, f"semantic gate passed {len(SUPER_SEMANTIC_GATE_PROBES)} raw probes"
    except Exception as exc:  # pragma: no cover - integration path
        ok, reason = False, f"semantic gate exception {exc.__class__.__name__}: {_sanitize_error(exc)}"

    if use_cache:
        SUPER_SEMANTIC_GATE_CACHE[api_base] = {"ok": ok, "reason": reason, "ts": now}
    return ok, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the canonical local Super semantic gate.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=SUPER_MODEL)
    parser.add_argument("--no-models-precheck", action="store_true")
    args = parser.parse_args(argv)
    ok, reason = local_super_semantic_gate(
        base_url=args.base_url,
        model=args.model,
        use_cache=False,
        precheck_models=not args.no_models_precheck,
    )
    if ok:
        print(reason)
        return 0
    print(f"corruption_signature={reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
