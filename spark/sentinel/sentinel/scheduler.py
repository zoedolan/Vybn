"""Scheduler: python -m sentinel.scheduler"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    yaml = None

from sentinel.crawlers import polymarket, news
from sentinel.processors.claim_extractor import process_news_bundle
from sentinel.processors.market_correlator import (
    load_latest_snapshots, compute_deltas, correlate)
from sentinel.synthesizer import synthesize

log = logging.getLogger("sentinel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_config(path: str = "config.yaml") -> dict:
    if not yaml: raise RuntimeError("pip install pyyaml")
    return yaml.safe_load(Path(path).read_text())


def make_local_model_fn(config: dict):
    import httpx
    base = config.get("local_model", {}).get("endpoint", "http://localhost:8080")
    temp = config.get("local_model", {}).get("temperature", 0.1)
    def call(prompt: str) -> str:
        resp = httpx.post(f"{base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}],
                  "temperature": temp, "max_tokens": 4096}, timeout=120)
        return resp.json()["choices"][0]["message"]["content"]
    return call


def make_frontier_fn(config: dict):
    import anthropic
    client = anthropic.Anthropic()
    model = config.get("frontier_model", {}).get("model", "claude-sonnet-4-20250514")
    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    return call


def run_cycle(config: dict):
    log.info("Sentinel cycle start")
    polymarket.run(config)
    news_path = news.run(config)
    claims = process_news_bundle(news_path, config, make_local_model_fn(config))
    snapshots = load_latest_snapshots(config["output"]["raw_dir"])
    deltas = compute_deltas(snapshots)
    questions = {s["market_id"]: s["question"] for s in snapshots if s.get("market_id")}
    claims = correlate(claims, deltas, questions)
    sd = Path(config["output"]["structured_dir"])
    sd.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (sd / f"claims_{ts}.json").write_text(json.dumps(claims, indent=2))
    log.info(f"{len(claims)} claims saved")
    state = synthesize(claims, config["output"]["belief_state"], make_frontier_fn(config))
    log.info(f"Belief state updated: {state.get('last_updated')}")


def main():
    config = load_config()
    interval = config.get("polymarket", {}).get("poll_interval_minutes", 15) * 60
    while True:
        try: run_cycle(config)
        except Exception as e: log.error(f"Cycle failed: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    main()
