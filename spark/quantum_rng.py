"""
quantum_rng.py — Pull true quantum random numbers from Cisco's
Outshift QRNG hardware via their public API.

Rate-limited: at most one call every five minutes.  Between calls the
last fetched value is returned instantly so callers never block.

Environment
-----------
Set QRNG_API_KEY to your Outshift API key (free tier).
  export QRNG_API_KEY="your-key-here"

Usage
-----
    from quantum_rng import quantum_random

    value = await quantum_random()          # async
    value = quantum_random_sync()           # sync wrapper
    print(value)                            # e.g. 738201
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

QRNG_ENDPOINT = "https://api.qrng.outshift.com/api/v1/random_numbers"
COOLDOWN_SECONDS = 300  # five minutes


@dataclass
class _QRNGState:
    """Singleton mutable state shared across every call."""
    last_fetch: float = 0.0
    cached_value: Optional[int] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_state = _QRNGState()


def _api_key() -> str:
    key = os.environ.get("QRNG_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "QRNG_API_KEY is not set. Sign up at "
            "https://qrng.outshift.com to get a free key."
        )
    return key


async def quantum_random(
    bits: int = 32,
    encoding: str = "raw",
    fmt: str = "all",
) -> int:
    """
    Return a quantum-sourced random integer.

    If fewer than five minutes have elapsed since the last API call,
    the previously fetched value is returned immediately and no
    network request is made.

    Parameters
    ----------
    bits : int
        Bits per block (max 10 000 per Outshift docs).
    encoding : str
        "raw" | "aes-ctr" — server-side post-processing mode.
    fmt : str
        "all" returns every available representation; you can also
        pass "hex", "decimal", "octal", or "binary".
    """
    async with _state.lock:
        now = time.monotonic()
        elapsed = now - _state.last_fetch

        if _state.cached_value is not None and elapsed < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - elapsed
            logger.debug(
                "Rate-limited — returning cached quantum value "
                "(%.0fs until next fetch allowed)", remaining
            )
            return _state.cached_value

        payload = {
            "encoding": encoding,
            "format": fmt,
            "bits_per_block": bits,
            "number_of_blocks": 1,
        }
        headers = {
            "Content-Type": "application/json",
            "x-id-api-key": _api_key(),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                QRNG_ENDPOINT, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()

        # The API returns a structure like:
        #   {"random_numbers": [{"decimal": 738201, "hex": "...", ...}]}
        # Pull the decimal representation out of the first block.
        numbers = body.get("random_numbers", body.get("data", []))
        if isinstance(numbers, list) and numbers:
            entry = numbers[0]
            value = (
                entry.get("decimal")
                if isinstance(entry, dict)
                else entry
            )
        else:
            value = numbers

        _state.cached_value = int(value)
        _state.last_fetch = time.monotonic()
        logger.info("Fetched fresh quantum random value: %s", _state.cached_value)
        return _state.cached_value


def quantum_random_sync(**kwargs) -> int:
    """Blocking convenience wrapper around the async function."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, quantum_random(**kwargs)).result()
    return asyncio.run(quantum_random(**kwargs))


def seconds_until_next_fetch() -> float:
    """How many seconds until the rate limiter will allow a fresh call."""
    elapsed = time.monotonic() - _state.last_fetch
    return max(0.0, COOLDOWN_SECONDS - elapsed)


# ── CLI quick-test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    if not os.environ.get("QRNG_API_KEY"):
        print(
            "Set QRNG_API_KEY first.\n"
            "  export QRNG_API_KEY='your-outshift-key'\n"
            "Sign up free: https://qrng.outshift.com",
            file=sys.stderr,
        )
        sys.exit(1)

    val = quantum_random_sync()
    print(f"Quantum random value: {val}")
    print(f"Next fetch available in: {seconds_until_next_fetch():.0f}s")

    # Second call should be instant / cached
    val2 = quantum_random_sync()
    print(f"Cached value (should match): {val2}")
