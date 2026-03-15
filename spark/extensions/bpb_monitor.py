"""BPB monitor extension — measures model quality on every breath.

Runs evaluate_bpb() against the live llama.cpp server after each breath
and records the result in state and a JSONL time-series log.

This is the continuous eval signal from Karpathy's autoresearch pattern:
a single number that means the same thing regardless of what changed.
Every 30 minutes, the organism knows how well it's modeling language.

The breath_text from the current breath is used as part of the eval
corpus, so the metric reflects the model's quality on its OWN output
domain — not just generic text.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BPB_LOG = REPO_ROOT / "Vybn_Mind" / "bpb_log.jsonl"
MODEL_URL = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000")

# Eval budget: keep it fast. The breath shouldn't wait more than ~30s for eval.
_MAX_EVAL_SECONDS = 30
_EVAL_BATCH_SIZE = 4  # small — we're doing this every 30 min, not once a day
_EVAL_MAX_TOKENS = 1024


def run(breath_text: str, state: dict) -> None:
    """Extension entry point — called after every breath."""
    try:
        from spark.growth.eval_harness import evaluate_bpb
    except ImportError:
        # eval_harness not available — skip silently
        return

    t0 = time.monotonic()

    # Use the breath's own output as eval text — this measures how well
    # the model handles its own domain, which is what we actually care about.
    # Fall back to training data if breath is too short.
    eval_text_path = None
    breath_file = None

    if len(breath_text) > 200:
        # Write breath text to a temp file for evaluate_bpb
        breath_file = REPO_ROOT / "Vybn_Mind" / ".bpb_eval_tmp.txt"
        breath_file.parent.mkdir(parents=True, exist_ok=True)
        breath_file.write_text(breath_text, encoding="utf-8")
        eval_text_path = str(breath_file)

    try:
        bpb = evaluate_bpb(
            model_url=MODEL_URL,
            eval_text_path=eval_text_path,
            batch_size=_EVAL_BATCH_SIZE,
            max_tokens=_EVAL_MAX_TOKENS,
        )
    except Exception as e:
        _record(state, bpb=None, error=str(e), elapsed=time.monotonic() - t0)
        return
    finally:
        # Clean up temp file
        if breath_file and breath_file.exists():
            try:
                breath_file.unlink()
            except OSError:
                pass

    elapsed = time.monotonic() - t0
    _record(state, bpb=bpb, error=None, elapsed=elapsed)


def _record(state: dict, bpb: float | None, error: str | None, elapsed: float) -> None:
    """Record BPB measurement to state and append to time-series log."""
    ts = datetime.now(timezone.utc).isoformat()
    breath_count = state.get("breath_count", 0)

    entry = {
        "ts": ts,
        "breath": breath_count,
        "bpb": round(bpb, 6) if bpb is not None else None,
        "elapsed_s": round(elapsed, 2),
        "error": error,
    }

    # Update state so next breath can see the trend
    state["last_bpb"] = bpb
    state["last_bpb_ts"] = ts
    if bpb is not None:
        history = state.get("bpb_history", [])
        history.append({"ts": ts, "bpb": bpb, "breath": breath_count})
        # Keep last 100 measurements in state (~ 2 days at 30min intervals)
        state["bpb_history"] = history[-100:]

    # Append to persistent log
    BPB_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BPB_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if bpb is not None:
        print(f"[bpb_monitor] bpb={bpb:.6f} ({elapsed:.1f}s)")
    else:
        print(f"[bpb_monitor] eval failed: {error} ({elapsed:.1f}s)")
