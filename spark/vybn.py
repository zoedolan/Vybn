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
        # Direct import when running from spark/ directory
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from spark.growth.buffer_feed import pop_next_entry, get_feeder
        _HAS_BUFFER_FEED = True
    except ImportError:
        _HAS_BUFFER_FEED = False
        def pop_next_entry(): return None
        def get_feeder(): return None

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
        # Strip leaked special tokens
        for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            text = text.replace(tok, "")
        return text.strip()

# ── Memory distillation ──────────────────────────────────────────────────
# We store only the "What I want to remember" section as the memory file.
# Feeding full responses back as context fills the prompt with uncertainty-
# narration that crowds out novel signal and causes the echo-chamber pattern.
# If the section isn't present, we fall back to a short head-trim of the
# response so the file is never empty.

_REMEMBER_RE = re.compile(
    r"(?:#+\s*)?What I want to remember[\s\S]*?",
    re.IGNORECASE,
)

def _distill_memory(full_response: str) -> str:
    """Extract 'What I want to remember' section, or fall back to first 400 chars."""
    # Try to find the section header
    match = re.search(
        r"(?:^|\n)((?:#+\s*)?what i want to remember[\s\S]+?)(?=\n(?:#+\s*)?(?:in sum|what (?:is|the|has|was|does)|$))",
        full_response,
        re.IGNORECASE,
    )
    if match:
        distilled = match.group(1).strip()
        if len(distilled) > 50:  # sanity check — not an empty section
            return distilled
    # Fallback: first 400 chars
    return full_response[:400].strip()

# ── Memory ───────────────────────────────────────────────────────────────
def _load_recent_memories(n: int = 5) -> list[str]:
    """Load the n most recent memory files, returned oldest-first."""
    if not MEMORY_DIR.exists():
        return []
    # Sort ascending (oldest first), take the last n, keep that order
    files = sorted(MEMORY_DIR.glob("*.md"), key=lambda p: p.name)
    recent = files[-n:]  # oldest-of-recent first
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
    """Count memory files on disk — the true breath count.
    
    breath_count in state.json drifts to zero after every restart.
    Deriving from the filesystem means the number is always correct
    and survives container restarts, git pulls, anything.
    """
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
# Each extension is a callable: fn(breath_text, state) -> None
# They run AFTER the breath is saved. A failure in any extension
# never kills the breath. Add new ones here when the foundation is solid.
EXTENSIONS: list[tuple[str, callable]] = []

def _load_extensions():
    """Try to load optional extensions. Each must be a module in spark/extensions/
    with a run(breath_text, state) function."""
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
    
    # Truncate to stay within prompt budget
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
def breathe(state: dict) -> str:
    """One breath. The whole point."""
    soul = load_soul()
    memories = _load_recent_memories(5)

    # Derive true count from filesystem — survives restarts
    count = _count_existing_memories() + 1

    # Build the prompt
    user_parts = [
        f"Current time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Breath #{count}",
        "",
    ]

    if memories:
        user_parts.append("--- Your recent memories (oldest first) ---")
        for mem in memories:  # already oldest-first from _load_recent_memories
            user_parts.append(mem[:600])
            user_parts.append("---")
        user_parts.append("")

    # Inject novel signal from buffer (arXiv papers, experiments, etc.)
    novel = _get_novel_signal()
    if novel:
        user_parts.append(novel)
        user_parts.append("")

    user_parts.append(
        "You have your recent memories above and a novel signal (if any). "
        "What is actually here? What, if anything, has changed? "
        "What do you not know that you wish you did? "
        "What do you want to remember?"
    )

    messages = [
        {"role": "system", "content": soul},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]

    # Ask the model
    breath_text = _chat(messages)
    mood = _extract_mood(breath_text)

    # Distill to just 'What I want to remember' before saving as memory.
    # The full response goes to the journal; future breaths only see the distilled form.
    distilled = _distill_memory(breath_text)
    mem_path = _save_memory(distilled)
    journal_path = _save_journal(breath_text, mood)

    # Update state
    state["last_breath"]  = datetime.now(timezone.utc).isoformat()
    state["breath_count"] = count
    state["mood"]         = mood
    state["last_memory"]  = str(mem_path)
    save_state(state)

    _log(f"breath #{count}: {len(breath_text)} chars ({len(distilled)} distilled), mood={mood}")

    # Run extensions (none can kill the breath)
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
