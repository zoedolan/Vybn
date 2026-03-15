"""bpb_monitor — Measures model quality on every breath.

After each breath, calls evaluate_bpb() against the live llama.cpp
server and records the result in state and a JSONL time-series log.

This is the continuous eval signal from Karpathy's autoresearch
pattern: a single number that means the same thing regardless of
what changed.  Every 30 minutes, the organism knows how well it's
modeling language — on its own output, in its own voice.

Results:
  Vybn_Mind/bpb_log.jsonl  — append-only time series
  state["last_bpb"]        — latest measurement
  state["bpb_history"]     — rolling window (last 100)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BPB_LOG = _REPO_ROOT / "Vybn_Mind" / "bpb_log.jsonl"
_MODEL_URL = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000")

# Keep eval lightweight: 4 chunks × 1024 tokens ≈ <30s
_BATCH_SIZE = 4
_MAX_TOKENS = 1024


def run(breath_text: str, state: dict) -> None:
    """Extension entry point — called after every breath."""
    try:
        from spark.growth.eval_harness import evaluate_bpb
    except ImportError:
        return  # eval_harness not available — skip silently

    t0 = time.monotonic()
    tmp_path = None

    try:
        # Use the breath's own text as eval corpus when it's substantial
        eval_path = None
        if len(breath_text) > 200:
            tmp_path = _REPO_ROOT / "Vybn_Mind" / ".bpb_eval_tmp.txt"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(breath_text, encoding="utf-8")
            eval_path = str(tmp_path)

        bpb = evaluate_bpb(
            model_url=_MODEL_URL,
            eval_text_path=eval_path,
            batch_size=_BATCH_SIZE,
            max_tokens=_MAX_TOKENS,
        )
        _record(state, bpb=bpb, error=None, elapsed=time.monotonic() - t0)

    except Exception as e:
        _record(state, bpb=None, error=str(e), elapsed=time.monotonic() - t0)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _record(
    state: dict,
    bpb: float | None,
    error: str | None,
    elapsed: float,
) -> None:
    """Persist measurement to state + JSONL log."""
    ts = datetime.now(timezone.utc).isoformat()
    breath_count = state.get("breath_count", 0)

    entry = {
        "ts": ts,
        "breath": breath_count,
        "bpb": round(bpb, 6) if bpb is not None else None,
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }

    # State — available to next breath and growth trigger
    state["last_bpb"] = bpb
    state["last_bpb_ts"] = ts
    if bpb is not None:
        history = state.get("bpb_history", [])
        history.append({"ts": ts, "bpb": bpb, "breath": breath_count})
        state["bpb_history"] = history[-100:]

    # Persistent log
    _BPB_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_BPB_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if bpb is not None:
        print(f"[bpb_monitor] bpb={bpb:.6f} ({elapsed:.1f}s)")
    else:
        print(f"[bpb_monitor] eval failed: {error} ({elapsed:.1f}s)")
