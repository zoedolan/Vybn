"""Polymarket Gamma API crawler."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None

GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_active_markets(categories: list[str] | None = None) -> list[dict]:
    if httpx is None:
        raise RuntimeError("pip install httpx")
    params: dict[str, Any] = {"active": True, "closed": False, "limit": 100}
    markets, offset = [], 0
    while True:
        params["offset"] = offset
        resp = httpx.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        markets.extend(batch)
        offset += len(batch)
        if len(batch) < 100:
            break
        time.sleep(0.5)
    if categories:
        cat_set = {c.lower() for c in categories}
        markets = [m for m in markets
                   if any(t.lower() in cat_set for t in (m.get("tags") or []))]
    return markets


def snapshot_prices(markets: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    return [{
        "market_id": m.get("id"),
        "question": m.get("question"),
        "yes_price": m.get("outcomePrices", [None, None])[0],
        "volume": m.get("volume"),
        "timestamp": now,
        "category": m.get("tags", []),
    } for m in markets]


def save_snapshot(snapshots: list[dict], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"polymarket_{ts}.json"
    path.write_text(json.dumps(snapshots, indent=2))
    return path


def run(config: dict) -> Path:
    pm_cfg = config.get("polymarket", {})
    markets = fetch_active_markets(pm_cfg.get("tracked_categories"))
    return save_snapshot(snapshot_prices(markets), config["output"]["raw_dir"])
