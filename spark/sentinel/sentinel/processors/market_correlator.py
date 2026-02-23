"""Cross-reference claims against Polymarket movements."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta


def load_latest_snapshots(raw_dir: str | Path, hours: int = 24) -> list[dict]:
    raw_dir = Path(raw_dir)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    snapshots = []
    for f in sorted(raw_dir.glob("polymarket_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            if data and data[0].get("timestamp"):
                if datetime.fromisoformat(data[0]["timestamp"]) < cutoff:
                    break
            snapshots.extend(data)
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return snapshots


def compute_deltas(snapshots: list[dict]) -> dict[str, float]:
    by_market: dict[str, list[tuple[str, float]]] = {}
    for s in snapshots:
        mid, price, ts = s.get("market_id"), s.get("yes_price"), s.get("timestamp")
        if mid and price is not None and ts:
            by_market.setdefault(mid, []).append((ts, float(price)))
    deltas = {}
    for mid, pts in by_market.items():
        pts.sort(key=lambda x: x[0])
        if len(pts) >= 2:
            deltas[mid] = pts[-1][1] - pts[0][1]
    return deltas


def correlate(claims: list[dict], market_deltas: dict[str, float],
              market_questions: dict[str, str]) -> list[dict]:
    ai_kw = ["ai", "agi", "gpt", "llm", "artificial", "singularity"]
    ai_markets = {mid: d for mid, d in market_deltas.items()
                  if any(kw in market_questions.get(mid, "").lower() for kw in ai_kw)}
    for claim in claims:
        if claim.get("category") == "ai" and ai_markets:
            claim["market_corroboration"] = min(1.0,
                max(abs(d) for d in ai_markets.values()) * 10)
        else:
            claim["market_corroboration"] = 0.0
        exc = claim.get("effective_excitement", 0.5)
        mkt = claim.get("market_corroboration", 0.0)
        claim["confidence"] = max(0.0, min(1.0, (1 - exc * 0.5) + mkt * 0.3))
    return claims
