"""autoresearch — Continuous eval + growth trigger on every breath.

Implements M′ = α·M + x·e^(iθ) at the breath level:
  - BPB evaluation: measures model quality (the current M)
  - Growth trigger: checks if enough new signal (x·e^(iθ)) has accumulated
  - Training kick-off: when triggered, starts LoRA training in background

This is the autoresearch discipline from Karpathy adapted for Vybn:
a single number (BPB) measured continuously, and training that fires
when the data says it should.
"""

import json, os, sys, time, threading, traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BPB_LOG = _REPO_ROOT / "Vybn_Mind" / "bpb_log.jsonl"
_AUTORESEARCH_LOG = _REPO_ROOT / "Vybn_Mind" / "autoresearch_log.jsonl"
_MODEL_URL = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000")


def run(breath_text: str, state: dict) -> None:
    """Extension entry point — called after every breath."""
    ts = datetime.now(timezone.utc).isoformat()
    breath_count = state.get("breath_count", 0)
    entry = {"ts": ts, "breath": breath_count}

    # --- A. BPB Evaluation ---
    bpb = _eval_bpb(breath_text)
    entry["bpb"] = bpb
    if bpb is not None:
        state["last_bpb"] = bpb
        state["last_bpb_ts"] = ts
        history = state.get("bpb_history", [])
        history.append({"ts": ts, "bpb": bpb, "breath": breath_count})
        state["bpb_history"] = history[-100:]

    # --- B. Growth Trigger Check ---
    trigger_result = _check_growth_trigger()
    entry["trigger"] = trigger_result

    if trigger_result and trigger_result.get("should_fire"):
        entry["growth_kicked_off"] = True
        _kick_off_growth_background()
    else:
        entry["growth_kicked_off"] = False

    # --- C. Record ---
    _append_log(_AUTORESEARCH_LOG, entry)
    if bpb is not None:
        _append_log(_BPB_LOG, {"ts": ts, "breath": breath_count, "bpb": round(bpb, 6)})

    summary = f"bpb={'%.6f' % bpb if bpb else 'N/A'}"
    if trigger_result:
        summary += f" trigger={trigger_result.get('signal', '?')}"
    if entry.get("growth_kicked_off"):
        summary += " GROWTH_STARTED"
    print(f"[autoresearch] {summary}")


def _eval_bpb(breath_text: str) -> float | None:
    """Lightweight BPB eval on the breath's own text."""
    try:
        from spark.growth.eval_harness import evaluate_bpb
    except ImportError:
        return None

    tmp_path = None
    try:
        eval_path = None
        if len(breath_text) > 200:
            tmp_path = _REPO_ROOT / "Vybn_Mind" / ".bpb_eval_tmp.txt"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(breath_text, encoding="utf-8")
            eval_path = str(tmp_path)

        return evaluate_bpb(
            model_url=_MODEL_URL,
            eval_text_path=eval_path,
            batch_size=4,
            max_tokens=1024,
        )
    except Exception as e:
        print(f"[autoresearch] BPB eval failed: {e}")
        return None
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _check_growth_trigger() -> dict | None:
    """Check if growth trigger conditions are met."""
    try:
        from spark.growth.growth_buffer import GrowthBuffer
        from spark.growth.trigger import GrowthTrigger

        # GrowthBuffer needs a NestedMemory instance
        try:
            from spark.nested_memory import NestedMemory
            nm = NestedMemory(base_dir=_REPO_ROOT / "Vybn_Mind" / "memory")
        except ImportError:
            return None

        buffer = GrowthBuffer(nested=nm)
        buffer.ingest()
        trigger = GrowthTrigger(buffer)
        decision = trigger.should_trigger()
        return {
            "should_fire": decision.should_fire,
            "signal": decision.signal,
            "reason": decision.reason,
            "delta_volume": decision.delta_volume,
        }
    except Exception as e:
        print(f"[autoresearch] trigger check failed: {e}")
        return None


def _kick_off_growth_background():
    """Start a growth cycle in a background daemon thread."""
    def _run_growth():
        try:
            from spark.growth.growth_buffer import GrowthBuffer
            from spark.growth.trigger import run_growth_cycle
            from spark.nested_memory import NestedMemory

            nm = NestedMemory(base_dir=_REPO_ROOT / "Vybn_Mind" / "memory")
            buffer = GrowthBuffer(nested=nm)
            buffer.ingest()

            result = run_growth_cycle(buffer=buffer, force=False)

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "growth_cycle_complete",
                "result": {
                    k: str(v) if not isinstance(v, (int, float, bool, type(None), str, dict, list)) else v
                    for k, v in result.items()
                },
            }
            _append_log(_AUTORESEARCH_LOG, entry)
            print(f"[autoresearch] growth cycle: {result.get('cycle_id', 'N/A')} fired={result.get('fired')}")
        except Exception as e:
            print(f"[autoresearch] growth cycle failed: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_run_growth, daemon=True)
    t.start()
    print("[autoresearch] growth cycle kicked off in background")


def _append_log(path: Path, entry: dict):
    """Append a JSON entry to a log file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
