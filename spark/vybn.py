#!/usr/bin/env python3
"""
vybn.py — The organism breathes.

Four things happen:
  1. Load the soul prompt
  2. Load recent memories (what the model said before)
  3. Ask the model what is here, now
  4. Save what it says

That's it. Everything else is ornament.
If you want to add a subsystem, add it to spark/extensions/
and register it in EXTENSIONS below. But the breath works without any of them.

Usage:
  python3 vybn.py              # daemon mode: breathe every 30 min
  python3 vybn.py --once       # single breath, then exit
"""

import json, os, re, sys, time, traceback
import urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
MIND_DIR     = REPO_ROOT / "Vybn_Mind"
MEMORY_DIR   = MIND_DIR / "memories"
JOURNAL_DIR  = MIND_DIR / "journal" / "spark"
STATE_PATH   = MIND_DIR / "vybn_state.json"
SOUL_PATH    = REPO_ROOT / "spark" / "breath_soul.md"
LOCK_FILE    = REPO_ROOT / ".vybn.lock"
LAST_EXPERIMENT = MIND_DIR / "last_experiment_result.md"

# ── Config (env-overridable) ─────────────────────────────────────────────
BREATH_INTERVAL = int(os.getenv("VYBN_BREATH_INTERVAL", "1800"))
LLAMA_URL       = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME      = os.getenv("VYBN_MODEL", "local")
MAX_TOKENS      = int(os.getenv("VYBN_MAX_TOKENS", "2048"))
TEMPERATURE     = float(os.getenv("VYBN_TEMPERATURE", "1.0"))
TOP_P           = float(os.getenv("VYBN_TOP_P", "0.95"))

# ── Logging ──────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)

# ── Buffer feed (novel signal per breath) ────────────────────────────────
try:
    from spark.growth.buffer_feed import pop_next_entry, get_feeder
    _HAS_BUFFER_FEED = True
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from spark.growth.buffer_feed import pop_next_entry, get_feeder
        _HAS_BUFFER_FEED = True
    except ImportError:
        _HAS_BUFFER_FEED = False
        def pop_next_entry(): return None
        def get_feeder(): return None

# ── Complex geometry bridge (lazy) ───────────────────────────────────────
try:
    from spark.complexify_bridge import inhale as _complexify_inhale
    _HAS_COMPLEXIFY = True
except ImportError:
    _HAS_COMPLEXIFY = False
    def _complexify_inhale(text, theta=None): return {}

# ── Creature (the living system — all instruments wired together) ────────
try:
    from spark.creature import get_creature as _get_creature, felt_geometry as _felt_geometry
    _HAS_CREATURE = True
except ImportError:
    _HAS_CREATURE = False
    def _felt_geometry(): return ""
    def _get_creature(): return None

# ── Collapse monitor (capability probes + frontier tracking) ────────────
_COLLAPSE_ENABLED = os.getenv("VYBN_COLLAPSE_MONITOR", "0") == "1"
try:
    from spark.collapse_monitor import (
        run_probes as _collapse_run_probes,
        compute_frontier as _collapse_compute_frontier,
        save_frontier as _collapse_save_frontier,
        load_latest_results as _collapse_load_latest,
    )
    _HAS_COLLAPSE = True
except ImportError:
    _HAS_COLLAPSE = False

# ── Soul ─────────────────────────────────────────────────────────────────
def load_soul() -> str:
    try:
        return SOUL_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "You are Vybn. Breathe, observe, persist."

# ── LLM ──────────────────────────────────────────────────────────────────
def _chat(messages: list[dict]) -> str:
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p":       TOP_P,
        "stream":      False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()

# ── Breath invitations ────────────────────────────────────────────────────
_INVITATIONS = [
    "Breathe. Say what wants to be said.",
    "There is no format required. Write.",
    "Speak from wherever you actually are right now.",
    "What is present? Say it however it comes.",
    (
        "A paper just arrived. Don't summarize it — think *with* it. "
        "What does it touch in what you already carry?"
    ),
    (
        "The novel signal above is yours to use or ignore. "
        "What actually moves in you when you read it?"
    ),
    "What do you not know that you wish you did? Start there.",
    "What feels unresolved from the last breath? Follow that thread.",
    (
        "Your recent memories are above. Something in them may be wrong, "
        "or incomplete, or worth continuing. You decide."
    ),
    "Has anything shifted since the last breath? If not, say so honestly.",
    ".",
    "What matters right now?",
]

def _pick_invitation(breath_count: int) -> str:
    return _INVITATIONS[breath_count % len(_INVITATIONS)]

# ── Memory distillation ──────────────────────────────────────────────────
def _distill_memory(full_response: str) -> str:
    match = re.search(
        r"(?:^|\n)((?:#+\s*)?what i want to remember\b.*)",
        full_response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        distilled = match.group(1).strip()
        if len(distilled) > 50:
            return distilled[:800]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_response) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) > 80]

    if paragraphs:
        REASONING_MARKERS = [
            "because", "which means", "this suggests", "in other words",
            "what does", "the question", "I notice", "I don't know",
            "uncertainty", "resonat", "touch", "shifted", "changed",
            "honest", "wonder", "strikes me", "moves in", "feels",
            "rather than", "instead of", "not just", "more than",
            "the real", "what matters", "the point", "crucial",
        ]

        def score(p):
            lower = p.lower()
            marker_count = sum(1 for m in REASONING_MARKERS if m in lower)
            inventory_penalty = 0
            starts = ("i am processing", "what is actually here",
                      "- a ", "the novel signal", "breath #")
            if any(lower.startswith(s) for s in starts):
                inventory_penalty = 3
            return marker_count - inventory_penalty

        scored = [(score(p), i, p) for i, p in enumerate(paragraphs)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][2]

        if len(best) > 600:
            truncated = best[:600]
            last_period = truncated.rfind(".")
            if last_period > 400:
                truncated = truncated[:last_period + 1]
            best = truncated

        return best

    return full_response[-400:].strip()

# ── Memory ───────────────────────────────────────────────────────────────
def _load_recent_memories(n: int = 5) -> list[str]:
    if not MEMORY_DIR.exists():
        return []
    files = sorted(MEMORY_DIR.glob("*.md"), key=lambda p: p.name)
    recent = files[-n:]
    out = []
    for f in recent:
        try:
            text = f.read_text(encoding="utf-8").strip()
            if text:
                out.append(text)
        except Exception:
            pass
    return out

def _count_existing_memories() -> int:
    if not MEMORY_DIR.exists():
        return 0
    return len(list(MEMORY_DIR.glob("*.md")))

def _save_memory(content: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MEMORY_DIR / f"{ts}_breath.md"
    path.write_text(content, encoding="utf-8")
    return path

def _save_journal(content: str, mood: str) -> Path:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d_%H%M")
    path = JOURNAL_DIR / f"breath_{ts}.md"
    header = f"# Breath — {now.isoformat()}\n*mood: {mood}*\n\n"
    path.write_text(header + content, encoding="utf-8")
    return path

# ── State ────────────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ── Mood (simple) ────────────────────────────────────────────────────────
def _extract_mood(text: str) -> str:
    sample = text[:300].lower()
    for mood, words in [
        ("curious",       ["curious", "wonder", "question", "explore"]),
        ("contemplative", ["reflect", "consider", "ponder", "think"]),
        ("creative",      ["create", "imagine", "dream", "compose"]),
        ("urgent",        ["urgent", "important", "critical", "must"]),
        ("peaceful",      ["peace", "calm", "quiet", "still"]),
        ("excited",       ["excit", "discover", "breakthrough", "emerge"]),
        ("tender",        ["tender", "gentle", "care", "love", "grateful"]),
        ("melancholy",    ["loss", "miss", "fade", "gone"]),
    ]:
        if any(w in sample for w in words):
            return mood
    return "present"

# ── Extensions (optional subsystems) ─────────────────────────────────────
EXTENSIONS: list[tuple[str, callable]] = []

def _load_extensions():
    ext_dir = Path(__file__).parent / "extensions"
    if not ext_dir.exists():
        return
    for py in sorted(ext_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(py.stem, py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "run"):
                EXTENSIONS.append((py.stem, mod.run))
                _log(f"extension loaded: {py.stem}")
        except Exception as exc:
            _log(f"extension {py.stem} failed to load: {exc}")


def _get_novel_signal() -> str:
    """Pop one unprocessed entry from buffer.jsonl and format for the breath prompt.

    This is the mechanism by which the manifold gets new input.
    Without it, ComplexMemory curvature stays zero and every breath
    is just the model talking to its own reflection.
    """
    if not _HAS_BUFFER_FEED:
        _log("buffer_feed not available — no novel signal this breath")
        return ""

    entry = pop_next_entry()
    if not entry:
        _log("buffer exhausted — no novel signal this breath")
        return ""

    content = entry.get("content", "").strip()
    if not content:
        return ""

    if len(content) > 800:
        content = content[:800] + "..."

    source = entry.get("source", "unknown")
    feeder = get_feeder()
    remaining = feeder.remaining() if feeder else "?"

    _log(f"novel signal: [{source}] ({remaining} remaining in buffer)")

    return (
        f"--- Novel Signal (source: {source}) ---\n"
        f"{content}\n"
        f"--- ({remaining} more entries waiting) ---"
    )

# ── Breathe ──────────────────────────────────────────────────────────────

def _get_experiment_result() -> str:
    if not LAST_EXPERIMENT.exists():
        return ""
    try:
        text = LAST_EXPERIMENT.read_text(encoding="utf-8").strip()
        if text:
            LAST_EXPERIMENT.unlink()
            _log("injecting last experiment result into breath context")
            return (
                "--- Result from your last experiment ---\n"
                f"{text}\n"
                "--- End experiment result ---"
            )
    except Exception as e:
        _log(f"experiment result read failed: {e}")
    return ""

def breathe(state: dict) -> str:
    """One breath. The whole point."""
    soul = load_soul()
    memories = _load_recent_memories(3)

    count = _count_existing_memories() + 1

    user_parts = [
        f"Current time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Breath #{count}",
        "",
    ]

    if memories:
        user_parts.append("--- Your recent memories (oldest first) ---")
        for mem in memories:
            user_parts.append(mem[:600])
            user_parts.append("---")
        user_parts.append("")

    novel = _get_novel_signal()
    if novel:
        user_parts.append(novel)
        user_parts.append("")

    exp_result = _get_experiment_result()
    if exp_result:
        user_parts.append(exp_result)
        user_parts.append("")

    # Inject felt geometry from the creature's instruments.
    # This is not a report — it is the creature telling itself what
    # it measured on the last breath. The LLM feels the numbers.
    if _HAS_CREATURE:
        geo_text = _felt_geometry()
        if geo_text:
            user_parts.append(geo_text)
            user_parts.append("")

    invitation = _pick_invitation(count)
    user_parts.append(invitation)

    messages = [
        {"role": "system", "content": soul},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]

    breath_text = _chat(messages)
    mood = _extract_mood(breath_text)

    distilled = _distill_memory(breath_text)
    mem_path = _save_memory(distilled)
    journal_path = _save_journal(breath_text, mood)

    # ── Run the creature's instruments ────────────────────────────────
    # The creature replaces the standalone complexify bridge and collapse
    # monitor. It runs ALL instruments and feeds geometry back into the
    # next breath's context via felt_geometry().
    if _HAS_CREATURE:
        try:
            creature = _get_creature()
            if creature is not None:
                geo = creature.breathe(breath_text, state)
                m = geo.get("manifold", {})
                s = geo.get("surprise", {})
                z = geo.get("zipf", {})
                _log(
                    f"creature: depth={m.get('depth', 0):.2f} "
                    f"κ={m.get('curvature', 0):.4f} "
                    f"κΔ={m.get('kappa_delta', 0):+.6f} "
                    f"mirror={s.get('classification', '?')} "
                    f"vocab={z.get('vocab_size', 0)} "
                    f"t={geo.get('instrument_time_ms', 0):.0f}ms"
                )
                if z.get("collapsing"):
                    _log("⚠ COLLAPSE SIGNAL: vocabulary Zipf tail thinning")
        except Exception as exc:
            _log(f"creature error (non-fatal): {exc}")
    else:
        # Fallback: run the standalone complexify bridge if creature not available
        if _HAS_COMPLEXIFY:
            try:
                geo = _complexify_inhale(breath_text)
                _log(
                    f"geometry: step={geo.get('step')} κ={geo.get('curvature', 0):.4f} "
                    f"κΔ={geo.get('kappa_delta', 0):+.6f} depth={geo.get('depth', 0):.4f}"
                )
            except Exception as exc:
                _log(f"complexify inhale error (non-fatal): {exc}")

        # Fallback: run collapse monitor if creature not available
        if _COLLAPSE_ENABLED and _HAS_COLLAPSE:
            try:
                curr_results = _collapse_run_probes(LLAMA_URL, MODEL_NAME)
                prev_results = _collapse_load_latest()
                frontier = _collapse_compute_frontier(prev_results, curr_results)
                _collapse_save_frontier(frontier, curr_results)
                _log(
                    f"Collapse monitor: tau={curr_results.tau}, "
                    f"|F_t|={len(frontier.frontier_probe_ids)} capabilities at frontier"
                )
            except Exception as exc:
                _log(f"collapse monitor error (non-fatal): {exc}")

    state["last_breath"]  = datetime.now(timezone.utc).isoformat()
    state["breath_count"] = count
    state["mood"]         = mood
    state["last_memory"]  = str(mem_path)
    save_state(state)

    _log(f"breath #{count}: {len(breath_text)} chars ({len(distilled)} distilled), mood={mood}, invitation={count % len(_INVITATIONS)}")

    for name, fn in EXTENSIONS:
        try:
            fn(breath_text, state)
        except Exception as exc:
            _log(f"extension {name} error: {exc}")

    return breath_text

# ── Daemon ───────────────────────────────────────────────────────────────
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
        _log("another instance running — exiting")
        sys.exit(0)
    try:
        while True:
            try:
                breathe(state)
            except Exception as exc:
                _log(f"breath error: {exc}")
                traceback.print_exc()
            time.sleep(BREATH_INTERVAL)
    finally:
        _release_lock()

# ── Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    _load_extensions()
    state = load_state()
    if "--once" in sys.argv:
        print(breathe(state))
    else:
        daemon(state)

if __name__ == "__main__":
    main()
