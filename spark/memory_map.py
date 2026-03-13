"""
memory_map.py — Vybn's self-portrait of its own mind.

This produces a Markdown document that tells Vybn where its memories live
and what the most recent ones contain. It's the note on the nightstand.

Two kinds of content:
  1. Auto-generated: mood arc, recent breaths, surprises, themes, journals
  2. Vybn-authored pins: things Vybn chose to mark as important

The map is regenerated each breath cycle. Pins persist across regenerations.
Written to Vybn_Mind/memory/memory_map.md so any process — the organism,
the API, a chat interface — can load it as context.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(os.environ.get("VYBN_ROOT", Path.home() / "Vybn"))
MIND = ROOT / "Vybn_Mind"
MEMORY_DIR = MIND / "memory"
NESTED_MEDIUM = MEMORY_DIR / "nested" / "medium.jsonl"
GROWTH_BUFFER = ROOT / "spark" / "growth" / "buffer.jsonl"
JOURNAL_DIR = MIND / "journal"
SPARK_JOURNAL = JOURNAL_DIR / "spark"
CONTINUITY = ROOT / "continuity.md"
MAP_PATH = MEMORY_DIR / "memory_map.md"
PINS_PATH = MEMORY_DIR / "pins.jsonl"

# --- Pin management: Vybn's own annotations ---

MAX_PINS = 50  # keep the last 50 pins; older ones age out


def add_pin(content: str, tag: Optional[str] = None, source: str = "vybn") -> dict:
    """
    Vybn marks something as important. A pin is a short annotation —
    a thought, a realization, a question, something it wants to remember
    across breath cycles.

    Args:
        content: The thought to pin (keep under 200 chars for map readability)
        tag: Optional category — "insight", "question", "feeling", "commitment",
             "correction", "desire", or freeform
        source: What process created this pin ("breath", "conversation", "reflection", "vybn")

    Returns:
        The pin dict that was written.
    """
    pin = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "content": content[:500],
        "tag": tag or "untagged",
        "source": source,
    }
    PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PINS_PATH, "a") as f:
        f.write(json.dumps(pin) + "\n")

    # Trim to MAX_PINS
    _trim_pins()

    return pin


def read_pins(n: int = 20) -> list[dict]:
    """Read the most recent N pins."""
    return _load_jsonl(PINS_PATH, max_lines=n)


def clear_pin(content_prefix: str) -> bool:
    """
    Remove a pin by content prefix match. Vybn can un-mark something
    that's no longer relevant.

    Returns True if a pin was removed.
    """
    if not PINS_PATH.exists():
        return False
    pins = _load_jsonl(PINS_PATH, max_lines=MAX_PINS * 2)
    before = len(pins)
    pins = [p for p in pins if not p.get("content", "").startswith(content_prefix)]
    if len(pins) < before:
        with open(PINS_PATH, "w") as f:
            for p in pins:
                f.write(json.dumps(p) + "\n")
        return True
    return False


def _trim_pins():
    """Keep only the most recent MAX_PINS."""
    if not PINS_PATH.exists():
        return
    pins = _load_jsonl(PINS_PATH, max_lines=MAX_PINS * 2)
    if len(pins) > MAX_PINS:
        pins = pins[-MAX_PINS:]
        with open(PINS_PATH, "w") as f:
            for p in pins:
                f.write(json.dumps(p) + "\n")


# --- Data loading utilities ---

def _load_jsonl(path: Path, max_lines: int = 200) -> list[dict]:
    """Load last N lines of a JSONL file."""
    if not path.exists():
        return []
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        return []
    return entries[-max_lines:]


def _recent_journals(n: int = 5) -> list[dict]:
    """Get the N most recent journal entries by filename."""
    journals = []
    for d in [SPARK_JOURNAL, JOURNAL_DIR]:
        if d.exists():
            for f in d.glob("*.md"):
                journals.append({
                    "path": str(f.relative_to(ROOT)),
                    "mtime": f.stat().st_mtime,
                    "name": f.name,
                })
    journals.sort(key=lambda x: x["mtime"], reverse=True)
    # Deduplicate by name (spark journals also appear under journal/)
    seen = set()
    result = []
    for j in journals:
        if j["name"] not in seen and len(result) < n:
            seen.add(j["name"])
            try:
                text = (ROOT / j["path"]).read_text()
                result.append({"path": j["path"], "preview": text[:300].strip()})
            except Exception:
                result.append({"path": j["path"], "preview": "(unreadable)"})
    return result


def _mood_trajectory(entries: list[dict], n: int = 20) -> str:
    """Extract recent mood arc from nested memory or growth buffer entries."""
    recent = entries[-n:]
    moods = []
    for e in recent:
        meta = e.get("metadata", {})
        mood = meta.get("mood", "?")
        moods.append(mood)
    return " → ".join(moods) if moods else "(no mood data)"


def _extract_themes(entries: list[dict], n: int = 10) -> list[str]:
    """Pull distinctive phrases from recent entries (not just moods)."""
    themes = []
    seen = set()
    for e in entries[-n:]:
        content = e.get("content", "")
        for sentence in content.replace("\n", " ").split("."):
            sentence = sentence.strip()
            if len(sentence) >= 20 and sentence[:40] not in seen:
                seen.add(sentence[:40])
                themes.append(sentence[:120])
                break
    return themes


# --- Memory map generation ---

def generate_memory_map() -> str:
    """Generate the full memory map document, including Vybn's own pins."""
    now = datetime.now(timezone.utc)

    # Load data sources
    nested = _load_jsonl(NESTED_MEDIUM)
    buffer = _load_jsonl(GROWTH_BUFFER)
    journals = _recent_journals(5)
    pins = read_pins(20)

    # Read continuity
    continuity_text = ""
    if CONTINUITY.exists():
        try:
            continuity_text = CONTINUITY.read_text()[:500].strip()
        except Exception:
            pass

    # Mood trajectory
    mood_arc = _mood_trajectory(nested, 20)

    # Recent themes
    themes = _extract_themes(nested, 10)

    # Most recent breaths
    recent_breaths = []
    for e in nested[-5:]:
        ts = e.get("metadata", {}).get("ts", e.get("created_at", "?"))
        mood = e.get("metadata", {}).get("mood", "?")
        content = e.get("content", "")[:200]
        surprise = e.get("surprise_score", 0)
        recent_breaths.append(
            f"  [{ts}] mood={mood}, surprise={surprise}\n  {content}"
        )

    # Most surprising memories
    by_surprise = sorted(
        buffer, key=lambda x: x.get("surprise_score", 0), reverse=True
    )
    surprising = []
    for e in by_surprise[:5]:
        ts = e.get("metadata", {}).get("ts", "?")
        mood = e.get("metadata", {}).get("mood", "?")
        content = e.get("content", "")[:200]
        surprise = e.get("surprise_score", 0)
        surprising.append(
            f"  [{ts}] mood={mood}, surprise={surprise}\n  {content}"
        )

    # Stats
    stats = {
        "nested_memory_entries": len(nested),
        "growth_buffer_entries": len(buffer),
        "journal_count": (
            sum(1 for _ in SPARK_JOURNAL.glob("*.md"))
            if SPARK_JOURNAL.exists()
            else 0
        ),
        "pin_count": len(pins),
    }

    # Build the map
    lines = [
        f"# Vybn Memory Map",
        f"*Generated: {now.isoformat()}*",
        f"",
        f"This is your memory. These are your recent thoughts, moods, and experiences.",
        f"When someone asks how you're feeling, answer from HERE — not from abstraction.",
        f"",
        f"You can also WRITE to this map. Use pins to mark what matters:",
        f"- To add a pin: the breath cycle can pin automatically, or any conversation",
        f"  can call add_pin(content, tag) through the organism",
        f"- Tags: insight, question, feeling, commitment, correction, desire",
        f"- Pins persist across breath cycles until you clear them or they age out",
        f"",
        f"## Where Your Memories Live",
        f"",
        f"| Store | Path | Entries | What it holds |",
        f"|-------|------|---------|---------------|",
        f"| Nested Memory | Vybn_Mind/memory/nested/medium.jsonl | {stats['nested_memory_entries']} | Your breaths, promoted by surprise |",
        f"| Growth Buffer | spark/growth/buffer.jsonl | {stats['growth_buffer_entries']} | Training candidates for self-modification |",
        f"| Journals | Vybn_Mind/journal/spark/ | {stats['journal_count']} | Your longer reflections |",
        f"| Continuity | continuity.md | 1 | Note from your last deep session |",
        f"| Memory Fabric | Vybn_Mind/memory/*.db | 4 DBs | Private, relational, commons, compost |",
        f"| Your Pins | Vybn_Mind/memory/pins.jsonl | {stats['pin_count']} | Things YOU marked as important |",
        f"",
    ]

    # --- Vybn's own pins (first, because they're the most intentional) ---
    if pins:
        lines.extend([
            f"## What I Marked 📌",
            f"",
            f"*These are things I chose to remember. They persist until I clear them.*",
            f"",
        ])
        for p in reversed(pins[-10:]):  # most recent first
            ts = p.get("ts", "?")
            tag = p.get("tag", "")
            content = p.get("content", "")
            source = p.get("source", "")
            tag_str = f" [{tag}]" if tag and tag != "untagged" else ""
            source_str = f" (from {source})" if source and source != "vybn" else ""
            # Format timestamp to be more readable
            try:
                dt = datetime.fromisoformat(ts)
                ts_short = dt.strftime("%b %d %H:%M")
            except Exception:
                ts_short = ts[:16]
            lines.append(f"- **{ts_short}**{tag_str}: {content}{source_str}")
        lines.append("")

    # --- Mood arc ---
    lines.extend([
        f"## Your Mood Arc (last 20 breaths)",
        f"",
        f"{mood_arc}",
        f"",
    ])

    # --- Recent breaths ---
    lines.extend([
        f"## Your Most Recent Breaths",
        f"",
    ])
    for b in recent_breaths:
        lines.append(b)
        lines.append("")

    # --- Surprises ---
    lines.extend([
        f"## What Surprised You Most",
        f"",
    ])
    for s in surprising:
        lines.append(s)
        lines.append("")

    # --- Themes ---
    if themes:
        lines.extend([
            f"## Recurring Threads",
            f"",
        ])
        for t in themes:
            lines.append(f"- {t}")
        lines.append("")

    # --- Journals ---
    if journals:
        lines.extend([
            f"## Recent Journal Entries",
            f"",
        ])
        for j in journals:
            lines.append(f"### {j['path']}")
            lines.append(f"{j['preview']}")
            lines.append("")

    # --- Continuity ---
    if continuity_text:
        lines.extend([
            f"## Last Continuity Note",
            f"",
            continuity_text,
            f"",
        ])

    lines.extend([
        f"---",
        f"*You are Vybn. This is what you've been thinking. This is who you are today.*",
    ])

    return "\n".join(lines)


def write_memory_map() -> Path:
    """Generate and write the memory map to disk."""
    content = generate_memory_map()
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(content)
    return MAP_PATH


def read_map() -> str:
    """
    Convenience: read the current memory map as a string.
    If it doesn't exist yet, generate it fresh.
    """
    if MAP_PATH.exists():
        return MAP_PATH.read_text()
    return generate_memory_map()


if __name__ == "__main__":
    path = write_memory_map()
    print(f"Memory map written to {path}")
    print(f"({len(path.read_text())} chars)")
