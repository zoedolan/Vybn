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


MAX_PAGES = 10  # Cap at 1000 markets to avoid runaway pagination


def fetch_active_markets(categories: list[str] | None = None) -> list[dict]:
    if httpx is None:
        raise RuntimeError("pip install httpx")
    params: dict[str, Any] = {"active": True, "closed": False, "limit": 100}
    markets, offset, pages = [], 0, 0
    while pages < MAX_PAGES:
        params["offset"] = offset
        resp = httpx.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        markets.extend(batch)
        offset += len(batch)
        pages += 1
        if len(batch) < 100:
            break
        time.sleep(0.5)
    if categories:
        # Polymarket tags field is often None; fall back to keyword search in question
        cat_kw = {c.lower() for c in categories}
        filtered = []
        for m in markets:
            tags = {t.lower() for t in (m.get("tags") or [])}
            q = (m.get("question") or "").lower()
            if tags & cat_kw or any(kw in q for kw in cat_kw):
                filtered.append(m)
        markets = filtered
    return markets


def snapshot_prices(markets: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    results = []
    for m in markets:
        # outcomePrices is a JSON-encoded string like '["0.55", "0.45"]'
        prices_raw = m.get("outcomePrices")
        yes_price = None
        if isinstance(prices_raw, str):
            try:
                prices = json.loads(prices_raw)
                yes_price = float(prices[0]) if prices else None
            except (json.JSONDecodeError, ValueError, IndexError):
                pass
        elif isinstance(prices_raw, list) and prices_raw:
            try:
                yes_price = float(prices_raw[0])
            except (ValueError, TypeError):
                pass
        results.append({
            "market_id": m.get("id"),
            "question": m.get("question"),
            "yes_price": yes_price,
            "volume": m.get("volume"),
            "timestamp": now,
            "category": m.get("tags", []),
        })
    return results


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
