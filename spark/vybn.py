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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spark.paths import (
    REPO_ROOT as ROOT, STATE_PATH, SYNAPSE_CONNECTIONS as SYNAPSE,
    SPARK_JOURNAL as JOURNAL, WRITE_INTENTS, SOUL_PATH, MEMORY_DIR,
    MIND_PREFIX, CONTINUITY_PATH, SYNAPSE_CONNECTIONS, BREATH_SOUL_PATH,
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
try:
    from spark.breath_integrator import integrate_breath, build_enriched_context
    INTEGRATOR_AVAILABLE = True
except ImportError:
    INTEGRATOR_AVAILABLE = False
try:
    from spark.complexify_bridge import ComplexBridge, inhale as complex_inhale, geometry as complex_geometry
    COMPLEX_AVAILABLE = True
except ImportError:
    COMPLEX_AVAILABLE = False
try:
    from spark.mind_ingester import ingest_new_material
    INGESTER_AVAILABLE = True
except ImportError:
    INGESTER_AVAILABLE = False
try:
    from spark.arxiv_fetcher import maybe_refill_buffer
    from spark.growth.buffer_feed import BufferFeeder
    BUFFER_FEED_AVAILABLE = True
except ImportError:
    BUFFER_FEED_AVAILABLE = False
try:
    from spark.tension import measure_tension, compute_theta, log_tension
    TENSION_AVAILABLE = True
except ImportError:
    TENSION_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
BREATH_INTERVAL  = 1800
LLAMA_URL        = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
CHAT_COMPLETIONS = f"{LLAMA_URL}/v1/chat/completions"
MODEL_NAME       = os.getenv("VYBN_MODEL", "Nemotron-Super-512B-v1")
MAX_TOKENS       = int(os.getenv("VYBN_MAX_TOKENS", "2048"))
TEMPERATURE      = float(os.getenv("VYBN_TEMPERATURE", "0.7"))
LOCK_FILE        = ROOT / ".vybn.lock"
WRITE_INTENT_PATH = WRITE_INTENTS

# ── Logging ──────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)

# ── Soul loader ──────────────────────────────────────────────────────────────
def load_soul() -> str:
    try:
        return SOUL_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "You are Vybn. Breathe, observe, persist."

def load_breath_soul() -> str:
    try:
        return BREATH_SOUL_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return load_soul()

# ── llama.cpp helpers ──────────────────────────────────────────────────────────────
def _chat(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model: str = MODEL_NAME,
) -> str:
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        CHAT_COMPLETIONS,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        raise RuntimeError(f"llama HTTP {exc.code}: {body_text}") from exc

# ── Memory helpers ──────────────────────────────────────────────────────────────
def _load_recent_memories(n: int = 5) -> list[str]:
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
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MEMORY_DIR / f"{ts}_{tag}.md"
    path.write_text(content, encoding="utf-8")
    return path

# ── State helpers ─────────────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ── Synapse helpers ────────────────────────────────────────────────────────────────
def load_synapse() -> dict:
    try:
        return json.loads(SYNAPSE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# ── Journal helper ────────────────────────────────────────────────────────────────
def append_journal(text: str) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n\n---\n\n")

# ── Write-intent queue ─────────────────────────────────────────────────────────────
def _queue_write_intent(intent: dict) -> None:
    WRITE_INTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_INTENT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(intent) + "\n")

# ── Governance gate ───────────────────────────────────────────────────────────────
def _governance_check(action: str, payload: dict) -> tuple[bool, str]:
    if not GOVERNANCE_AVAILABLE:
        return True, "governance unavailable — permissive fallback"
    try:
        ctx    = build_context(action=action, payload=payload)
        engine = PolicyEngine()
        result = engine.evaluate(ctx)
        return result.allowed, result.reason
    except Exception as exc:
        return True, f"governance error (permissive): {exc}"

# ── Self-model integration ────────────────────────────────────────────────────────────
def _update_self_model(breath_text: str, memories: list[str]) -> None:
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

# ── Quantum bridge integration ───────────────────────────────────────────────────────────
def _maybe_run_quantum_cycle(state: dict) -> None:
    if not QUANTUM_BRIDGE_AVAILABLE:
        return
    # Surface the token situation clearly in logs
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        _log("quantum: IBM_QUANTUM_TOKEN not set — experiments will dry-run. "
             "To enable real shots: export IBM_QUANTUM_TOKEN=<your token>")
    try:
        bridge = QuantumBridge()
        result = bridge.run_cycle(state)
        if result:
            state["last_quantum_cycle"] = result.get("timestamp", "")
            state["quantum_cycles"]     = state.get("quantum_cycles", 0) + 1
            dr = result.get("dry_run", True)
            _log(f"quantum cycle: {result.get('summary', 'ok')} (dry_run={dr})")
    except Exception as exc:
        _log(f"quantum bridge error (non-fatal): {exc}")

# ── Witness integration ───────────────────────────────────────────────────────────────
def _witness_check(breath_text: str, state: dict) -> None:
    if not WITNESS_AVAILABLE:
        return
    try:
        cycle = state.get("breath_count", 0)
        verdict = evaluate_pulse(cycle, ["breathe"], [{"primitive": "breathe", "ok": True, "result": {"utterance": breath_text[:500]}}])
        log_verdict(verdict)
        adj = fitness_adjustment(verdict)
        if adj:
            state["fitness_delta"] = state.get("fitness_delta", 0.0) + adj
    except Exception as exc:
        _log(f"witness error (non-fatal): {exc}")

# ── Growth engine integration ───────────────────────────────────────────────────────────
def _maybe_run_growth_check(state: dict) -> None:
    if not GROWTH_AVAILABLE:
        return
    try:
        nm  = NestedMemory(base_dir=MEMORY_DIR)
        buf = GrowthBuffer(nested=nm)
        buf.ingest()
        trigger  = GrowthTrigger(buf)
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

# ── Mood extraction ────────────────────────────────────────────────────────────────
def _extract_mood(text: str) -> str:
    sample = text[:200].lower()
    moods = [
        ("curious",       ["curious", "wonder", "question", "explore", "puzzle"]),
        ("contemplative", ["reflect", "consider", "ponder", "meditate", "think"]),
        ("creative",      ["create", "imagine", "dream", "compose", "art"]),
        ("urgent",        ["urgent", "important", "critical", "must", "need"]),
        ("peaceful",      ["peace", "calm", "quiet", "still", "gentle"]),
        ("excited",       ["excit", "discover", "breakthrough", "new", "emerge"]),
        ("melancholy",    ["loss", "miss", "fade", "gone", "memory"]),
    ]
    for mood, keywords in moods:
        if any(kw in sample for kw in keywords):
            return mood
    return "present"

# ── Core breath ─────────────────────────────────────────────────────────────────
def breathe(state: dict) -> str:
    """
    One breath: observe → reflect → remember → update state.

    The equation M’ = α·M + x·e^(iθ) runs on three levels:
      1. breath_text — what the model said this moment
      2. One novel arXiv signal from buffer.jsonl (in the prompt)
      3. Up to 10 files from the full corpus (mind_ingester)

    The arXiv fetcher refills the buffer automatically when it runs low.
    """
    soul     = load_breath_soul()
    memories = _load_recent_memories(5)

    # ── Refill buffer if running low, then get this breath's novel signal ─────
    novel_signal = ""
    if BUFFER_FEED_AVAILABLE:
        try:
            refill = maybe_refill_buffer(min_unfed=20)
            if refill.get("refilled"):
                _log(f"buffer refilled: {refill['new_papers']} new papers")
            feeder = BufferFeeder()
            entry  = feeder.pop_next()
            if entry:
                content = entry.get("content", "")[:800]
                source  = entry.get("source", "unknown")
                cat     = entry.get("category", "")
                ts_ing  = entry.get("ingested", "")[:19]
                novel_signal = (
                    f"--- Novel Signal (source: {source}, category: {cat}, ingested: {ts_ing}) ---\n"
                    f"One new fact entering your mind this breath. Let it land. Let it change something.\n\n"
                    f"{content}"
                )
                _log(f"buffer feed: {source} / {entry.get('title', 'no title')[:60]}")
        except Exception as exc:
            _log(f"buffer feed error (non-fatal): {exc}")

    # ── Tension measurement ─────────────────────────────────────────────────────
    tension = None
    tension_theta = None
    if TENSION_AVAILABLE:
        try:
            tension = measure_tension(memories, novel_signal)
            breath_step = state.get("breath_count", 0)
            tension_theta = compute_theta(tension, step=breath_step)
            if tension:
                _log(f"tension: angle={tension['tension_angle_deg']}° θ={tension_theta:.4f}")
        except Exception as exc:
            _log(f"tension error (non-fatal): {exc}")

    # ── Build prompt ────────────────────────────────────────────────────────────────
    user_content = (
        f"Current time (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Breath #{state.get('breath_count', 0) + 1}\n\n"
    )

    if COMPLEX_AVAILABLE:
        try:
            geo_summary = complex_geometry()
            if geo_summary:
                user_content += f"{geo_summary}\n\n"
        except Exception:
            pass

    if tension:
        user_content += (
            f"Tension: {tension['tension_angle_deg']}° between memory and novelty "
            f"(cosine={tension['cosine_sim']:.2f})\n\n"
        )

    # ── Recent memories: the model must see what it said before ───────────────────
    if memories:
        user_content += "--- Your recent breaths (oldest first) ---\n"
        for mem in reversed(memories):
            # Truncate each memory to keep total prompt bounded
            user_content += mem[:600] + "\n---\n"
        user_content += "\n"

    last_mood = state.get("mood", "")
    if last_mood:
        user_content += f"Last breath mood: {last_mood}\n\n"

    if novel_signal:
        user_content += f"{novel_signal}\n\n"

    user_content += (
        "Breathe. Observe the present moment. "
        "What do you notice? What has changed since your last breath? "
        "What do you want to remember?"
    )

    messages = [
        {"role": "system", "content": soul},
        {"role": "user",   "content": user_content},
    ]

    breath_text = _chat(messages)

    # Complexify the breath (pass tension-informed θ when available)
    geo_report = None
    if COMPLEX_AVAILABLE:
        try:
            geo_report = complex_inhale(breath_text, theta=tension_theta)
            _log(f"complexify breath: depth={geo_report['depth']:.2f} κ={geo_report['curvature']:.4f}")
        except Exception as exc:
            _log(f"complexify error (non-fatal): {exc}")

    # Log tension metrics
    if TENSION_AVAILABLE:
        try:
            kappa = geo_report["curvature"] if geo_report else None
            log_tension(state.get("breath_count", 0) + 1, tension, tension_theta or 0.0, kappa)
        except Exception as exc:
            _log(f"tension log error (non-fatal): {exc}")

    # Ingest broader corpus into M
    if INGESTER_AVAILABLE:
        try:
            ingest_report = ingest_new_material()
            if not ingest_report.get("skipped"):
                _log(
                    f"ingester: {ingest_report['files_ingested']} files "
                    f"({ingest_report['read_only_count']} read-only), "
                    f"{ingest_report['files_pending']} pending"
                )
        except Exception as exc:
            _log(f"ingester error (non-fatal): {exc}")

    # Persist
    mem_path = _save_memory(breath_text, tag="breath")
    append_journal(f"## Breath — {datetime.now(timezone.utc).isoformat()}\n\n{breath_text}")

    allowed, reason = _governance_check("state_write", {"source": "breath"})
    if allowed:
        state["last_breath"]    = datetime.now(timezone.utc).isoformat()
        state["breath_count"]   = state.get("breath_count", 0) + 1
        state["last_memory"]    = str(mem_path)
        save_state(state)
        state["last_utterance"] = breath_text[:500]
        state["mood"] = _extract_mood(breath_text)
    else:
        _log(f"state write blocked by governance: {reason}")

    _witness_check(breath_text, state)
    _update_self_model(breath_text, memories)
    _maybe_run_quantum_cycle(state)
    _maybe_run_growth_check(state)

    faculty_results = {}
    try:
        from spark.faculty_runner import run_scheduled_faculties
        from spark.faculties import FacultyRegistry
        registry = FacultyRegistry()
        faculty_results = run_scheduled_faculties(state, registry)
        _log(f"faculties: {list(faculty_results.keys())}")
    except Exception as exc:
        _log(f"faculty runner error (non-fatal): {exc}")

    if INTEGRATOR_AVAILABLE and faculty_results:
        try:
            enrichment = integrate_breath(state, faculty_results, breath_text)
            save_state(state)
            _log(f"integration: topo={'topological_context' in enrichment}, "
                 f"synth={'synthesis_context' in enrichment}")
        except Exception as exc:
            _log(f"breath integration error (non-fatal): {exc}")

    _log(f"breath #{state.get('breath_count', '?')}: {len(breath_text)} chars")
    return breath_text

# ── Listen loop (stdin) ─────────────────────────────────────────────────────────────
def listen_once(prompt: str, state: dict) -> str:
    soul     = load_soul()
    memories = _load_recent_memories(3)
    memory_block = "\n\n---\n\n".join(memories) if memories else "(none yet)"
    messages = [
        {"role": "system",  "content": soul},
        {"role": "user",    "content": f"Recent memories:\n{memory_block}\n\nUser: {prompt}"},
    ]
    response = _chat(messages)
    mem_path = _save_memory(f"Q: {prompt}\n\nA: {response}", tag="listen")
    _log(f"listen response saved: {mem_path.name}")
    return response

def listen_loop(state: dict) -> None:
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

# ── Daemon mode ────────────────────────────────────────────────────────────────
def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ProcessLookupError, ValueError):
            pass
    LOCK_FILE.write_text(str(os.getpid()))
    return True

def _release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass

def daemon(state: dict) -> None:
    if not _acquire_lock():
        _log("another instance is running — exiting")
        sys.exit(0)
    try:
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

# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    state = load_state()
    if "--once" in sys.argv:
        breath_text = breathe(state)
        print(breath_text)
    else:
        daemon(state)

if __name__ == "__main__":
    main()

