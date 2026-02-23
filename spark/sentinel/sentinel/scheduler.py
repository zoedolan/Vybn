"""Scheduler: python -m sentinel.scheduler

Modes:
  --once         Run a single cycle and exit
  --local-only   Skip all frontier model calls
  --dry-run      Validate config and exit
"""
import argparse
import json
import signal
import sys
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

# --- Graceful shutdown ---
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    log.info(f"Signal {signum} received, finishing current cycle before exit...")
    _shutdown_requested = True


def load_config(path: str = "config.yaml") -> dict:
    if not yaml:
        raise RuntimeError("pip install pyyaml")
    return yaml.safe_load(Path(path).read_text())


def ensure_data_dirs(config: dict) -> None:
    """Create the gitignored data directories if they don't exist."""
    for key in ("data_dir", "structured_dir", "raw_dir"):
        d = config.get("output", {}).get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)


def make_local_model_fn(config: dict):
    import httpx
    base = config.get("local_model", {}).get("endpoint", "http://localhost:8081")
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


def save_scheduler_state(data_dir: str, state: dict) -> None:
    path = Path(data_dir) / "scheduler_state.json"
    path.write_text(json.dumps(state, indent=2))


def load_scheduler_state(data_dir: str) -> dict:
    path = Path(data_dir) / "scheduler_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_successful_cycle": None, "cycles_completed": 0, "last_error": None}


def run_cycle(config: dict, local_only: bool = False):
    log.info("Sentinel cycle start")
    ensure_data_dirs(config)

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

    if not local_only:
        state = synthesize(claims, config["output"]["belief_state"],
                           make_frontier_fn(config))
        log.info(f"Belief state updated: {state.get('last_updated')}")
    else:
        log.info("Local-only mode: skipping frontier synthesis")


def main():
    parser = argparse.ArgumentParser(description="Sentinel scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle and exit")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip frontier model calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and exit")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        log.info("Config loaded successfully. Dry run complete.")
        log.info(f"  Local model endpoint: {config.get('local_model', {}).get('endpoint')}")
        log.info(f"  Data dir: {config.get('output', {}).get('data_dir')}")
        log.info(f"  Frontier model: {config.get('frontier_model', {}).get('model')}")
        ensure_data_dirs(config)
        log.info("  Data directories created.")
        return

    # Install signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    data_dir = config.get("output", {}).get("data_dir", "data")
    sched_state = load_scheduler_state(data_dir)

    interval = config.get("polymarket", {}).get("poll_interval_minutes", 15) * 60

    while True:
        try:
            run_cycle(config, local_only=args.local_only)
            sched_state["last_successful_cycle"] = datetime.now(timezone.utc).isoformat()
            sched_state["cycles_completed"] += 1
            sched_state["last_error"] = None
            save_scheduler_state(data_dir, sched_state)
        except Exception as e:
            log.error(f"Cycle failed: {e}")
            sched_state["last_error"] = str(e)
            save_scheduler_state(data_dir, sched_state)

        if args.once or _shutdown_requested:
            log.info("Exiting.")
            break

        # Sleep in 1-second increments to check for shutdown
        for _ in range(interval):
            if _shutdown_requested:
                break
            time.sleep(1)

        if _shutdown_requested:
            log.info("Shutdown requested during sleep. Exiting.")
            break


if __name__ == "__main__":
    main()
