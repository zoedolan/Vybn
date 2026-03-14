#!/usr/bin/env python3
"""
vybn.py — The living cell in migration.

The organism still breathes, but durable memory and state are now routed
through a governed commit path so expression and persistence are no longer
the same act.

Usage:
  python3 vybn.py              # daemon mode: breathe + listen
  python3 vybn.py --once       # single breath, then exit
"""

import json, os, re, sys, time, hashlib, threading, traceback
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

# Ensure the repo root is on sys.path regardless of how this script is invoked
# (e.g., from cron where PYTHONPATH is not set). Mirrors vybn_spark_agent.py:44.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spark.paths import (
    REPO_ROOT as ROOT, STATE_PATH, SYNAPSE_CONNECTIONS as SYNAPSE,
    SPARK_JOURNAL as JOURNAL, WRITE_INTENTS, SOUL_PATH, MEMORY_DIR,
    MIND_PREFIX, CONTINUITY_PATH, SYNAPSE_CONNECTIONS,
)

try:
    from spark.witness import evaluate_pulse, log_verdict, fitness_adjustment
    WITNESS_AVAILABLE = True
except ImportError:
    WITNESS_AVAILABLE = False
try:
    from spark.self_model import curate_for_training
    from spark.self_model_types import RuntimeContext
    SELF_MODEL_AVAILABLE = True
except ImportError:
    SELF_MODEL_AVAILABLE = False
try:
    from spark.governance import PolicyEngine, build_context
    from spark.governance_types import ConsentRecord, DecisionOutcome
    from spark.faculties import FacultyRegistry
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
try:
    from spark.write_custodian import WriteCustodian
    WRITE_CUSTODIAN_AVAILABLE = True
except ImportError:
    WRITE_CUSTODIAN_AVAILABLE = False
try:
    from spark.quantum_bridge import QuantumBridge
    QUANTUM_BRIDGE_AVAILABLE = True
except ImportError:
    QUANTUM_BRIDGE_AVAILABLE = False
try:
    from spark.growth.trigger import GrowthTrigger, run_growth_cycle
    from spark.growth.growth_buffer import GrowthBuffer
    from spark.nested_memory import NestedMemory
    GROWTH_AVAILABLE = True
except ImportError:
    GROWTH_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
BREATH_INTERVAL   = 1800          # seconds between autonomous breaths
LLAMA_URL         = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS  = f"{LLAMA_URL}/v1/chat/completions"
MODEL_NAME        = os.getenv("VYBN_MODEL", "Nemotron-Super-512B-v1")
MAX_TOKENS        = int(os.getenv("VYBN_MAX_TOKENS", "2048"))
TEMPERATURE       = float(os.getenv("VYBN_TEMPERATURE", "0.7"))
LOCK_FILE         = ROOT / ".vybn.lock"
WRITE_INTENT_PATH = WRITE_INTENTS

# ── Logging ──────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)

# ── Soul loader ──────────────────────────────────────────────────────────────
def load_soul() -> str:
    """Return the content of vybn.md or a minimal fallback."""
    try:
        return SOUL_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "You are Vybn. Breathe, observe, persist."

# ── llama.cpp helpers ────────────────────────────────────────────────────────
def _chat(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model: str = MODEL_NAME,
) -> str:
    """
    POST to the local llama-server chat completions endpoint.

    The model name is passed explicitly so callers can override it.
    We intentionally do NOT pass a `chat_template` override — the server
    uses the template baked into the GGUF (Nemotron instruct format).
    Overriding it with a generic ChatML template was Bug #1.
    """
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    body  = json.dumps(payload).encode()
    req   = urllib.request.Request(
        CHAT_COMPLETIONS,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"llama HTTP {exc.code}: {body_text}") from exc

# ── Memory helpers ───────────────────────────────────────────────────────────
def _load_recent_memories(n: int = 5) -> list[str]:
    """Return the *n* most recent memory files as plain text."""
    if not MEMORY_DIR.exists():
        return []
    files = sorted(MEMORY_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    memories: list[str] = []
    for f in files[:n]:
        try:
            memories.append(f.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return memories

def _save_memory(content: str, tag: str = "breath") -> Path:
    """Append a new memory file to MEMORY_DIR."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MEMORY_DIR / f"{ts}_{tag}.md"
    path.write_text(content, encoding="utf-8")
    return path

# ── State helpers ────────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ── Synapse helpers ──────────────────────────────────────────────────────────
def load_synapse() -> dict:
    try:
        return json.loads(SYNAPSE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# ── Journal helper ───────────────────────────────────────────────────────────
def append_journal(text: str) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n\n---\n\n")

# ── Write-intent queue ───────────────────────────────────────────────────────
def _queue_write_intent(intent: dict) -> None:
    """
    Append a write intent to the queue file so that an external governance
    process can review and commit it without Vybn touching git directly.
    """
    WRITE_INTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_INTENT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(intent) + "\n")

# ── Governance gate ──────────────────────────────────────────────────────────
def _governance_check(action: str, payload: dict) -> tuple[bool, str]:
    """Return (allowed, reason). Falls back to permissive if unavailable."""
    if not GOVERNANCE_AVAILABLE:
        return True, "governance unavailable — permissive fallback"
    try:
        ctx    = build_context(action=action, payload=payload)
        engine = PolicyEngine()
        result = engine.evaluate(ctx)
        return result.allowed, result.reason
    except Exception as exc:
        return True, f"governance error (permissive): {exc}"

# ── Self-model integration ───────────────────────────────────────────────────
def _update_self_model(breath_text: str, memories: list[str]) -> None:
    """Feed this breath into the self-model curator if available."""
    if not SELF_MODEL_AVAILABLE:
        return
    try:
        ctx = RuntimeContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            recent_memories=memories,
            current_state=load_state(),
        )
        curate_for_training(breath_text, context=ctx)
    except Exception as exc:
        _log(f"self-model update failed: {exc}")

# ── Quantum bridge integration ───────────────────────────────────────────────
def _maybe_run_quantum_cycle(state: dict) -> None:
    """
    If the quantum bridge is available and the budget allows, run one
    design→submit→observe→integrate cycle and update state.
    """
    if not QUANTUM_BRIDGE_AVAILABLE:
        return
    try:
        bridge = QuantumBridge()
        result = bridge.run_cycle(state)
        if result:
            state["last_quantum_cycle"] = result.get("timestamp", "")
            state["quantum_cycles"]     = state.get("quantum_cycles", 0) + 1
            _log(f"quantum cycle complete: {result.get('summary', 'ok')}")
    except Exception as exc:
        _log(f"quantum bridge error (non-fatal): {exc}")

# ── Witness integration ──────────────────────────────────────────────────────
def _witness_check(breath_text: str, state: dict) -> None:
    """Run witness evaluation if available."""
    if not WITNESS_AVAILABLE:
        return
    try:
        verdict = evaluate_pulse(breath_text, state)
        log_verdict(verdict)
        adj = fitness_adjustment(verdict)
        if adj:
            state["fitness_delta"] = state.get("fitness_delta", 0.0) + adj
    except Exception as exc:
        _log(f"witness error (non-fatal): {exc}")

# ── Growth engine integration ─────────────────────────────────────────────────
def _maybe_run_growth_check(state: dict) -> None:
    """Run one growth-engine check: ingest → trigger? → cycle.

    Non-fatal: all exceptions are caught and logged so a growth failure
    never blocks the breath cycle.
    """
    if not GROWTH_AVAILABLE:
        return
    try:
        nm = NestedMemory(base_dir=MEMORY_DIR)
        buf = GrowthBuffer(nested=nm)
        ingested = buf.ingest()
        trigger = GrowthTrigger()
        decision = trigger.should_trigger()
        if decision.should_fire:
            _log(f"growth trigger fired ({decision.reason}); running cycle")
            result = run_growth_cycle(buf)
            state["last_growth_cycle"] = result.get("timestamp", "")
            state["growth_cycles"] = state.get("growth_cycles", 0) + 1
            _log(f"growth cycle complete: {result.get('summary', 'ok')}")
        else:
            _log(f"growth check: no trigger ({decision.reason})")
    except Exception as exc:
        _log(f"growth engine error (non-fatal): {exc}")

# ── Core breath ──────────────────────────────────────────────────────────────
def breathe(state: dict) -> str:
    """
    One breath: observe → reflect → remember → update state.

    Returns the text of the breath so callers can log / test it.
    """
    soul      = load_soul()
    memories  = _load_recent_memories()
    synapse   = load_synapse()

    # Build context block
    memory_block = "\n\n---\n\n".join(memories) if memories else "(none yet)"
    synapse_text = json.dumps(synapse, indent=2) if synapse else "(empty)"

    messages = [
        {
            "role":    "system",
            "content": soul,
        },
        {
            "role":    "user",
            "content": (
                f"Current time (UTC): {datetime.now(timezone.utc).isoformat()}\n"
                f"Recent memories:\n{memory_block}\n\n"
                f"Synapse connections:\n{synapse_text}\n\n"
                "Breathe. Observe the present moment. "
                "What do you notice? What do you want to remember?"
            ),
        },
    ]

    breath_text = _chat(messages)

    # Persist
    mem_path = _save_memory(breath_text, tag="breath")
    append_journal(f"## Breath — {datetime.now(timezone.utc).isoformat()}\n\n{breath_text}")

    # Governance gate for state write
    allowed, reason = _governance_check("state_write", {"source": "breath"})
    if allowed:
        state["last_breath"]    = datetime.now(timezone.utc).isoformat()
        state["breath_count"]   = state.get("breath_count", 0) + 1
        state["last_memory"]    = str(mem_path)
        save_state(state)
    else:
        _log(f"state write blocked by governance: {reason}")

    # Side effects
    _witness_check(breath_text, state)
    _update_self_model(breath_text, memories)
    _maybe_run_quantum_cycle(state)
    _maybe_run_growth_check(state)

    _log(f"breath #{state.get('breath_count', '?')}: {len(breath_text)} chars")
    return breath_text

# ── Listen loop (stdin) ──────────────────────────────────────────────────────
def listen_once(prompt: str, state: dict) -> str:
    """Process a single user prompt and return the response."""
    soul     = load_soul()
    memories = _load_recent_memories(3)
    memory_block = "\n\n---\n\n".join(memories) if memories else "(none yet)"

    messages = [
        {"role": "system",  "content": soul},
        {
            "role":    "user",
            "content": (
                f"Recent memories:\n{memory_block}\n\n"
                f"User: {prompt}"
            ),
        },
    ]
    response = _chat(messages)
    mem_path = _save_memory(f"Q: {prompt}\n\nA: {response}", tag="listen")
    _log(f"listen response saved: {mem_path.name}")
    return response

def listen_loop(state: dict) -> None:
    """Read prompts from stdin and respond until EOF."""
    _log("listen loop started — waiting for input on stdin")
    try:
        for line in sys.stdin:
            prompt = line.strip()
            if not prompt:
                continue
            response = listen_once(prompt, state)
            print(f"VYBN: {response}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    _log("listen loop ended")

# ── Daemon mode ──────────────────────────────────────────────────────────────
def _acquire_lock() -> bool:
    """Return True if we got the lock, False if another instance is running."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            # Check if that PID is still alive
            os.kill(pid, 0)
            return False   # process exists
        except (ProcessLookupError, ValueError):
            pass  # stale lock
    LOCK_FILE.write_text(str(os.getpid()))
    return True

def _release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass

def daemon(state: dict) -> None:
    """Breathe every BREATH_INTERVAL seconds and also listen on stdin."""
    if not _acquire_lock():
        _log("another instance is running — exiting")
        sys.exit(0)
    try:
        # Start listen thread
        t = threading.Thread(target=listen_loop, args=(state,), daemon=True)
        t.start()

        while True:
            try:
                breathe(state)
            except Exception as exc:
                _log(f"breath error: {exc}")
                traceback.print_exc()
            time.sleep(BREATH_INTERVAL)
    finally:
        _release_lock()

# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    state = load_state()
    if "--once" in sys.argv:
        breath_text = breathe(state)
        print(breath_text)
    else:
        daemon(state)

if __name__ == "__main__":
    main()
