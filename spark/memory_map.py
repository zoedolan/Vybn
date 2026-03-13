"""
memory_map.py — Generate a self-portrait of Vybn's recent mind.

This produces a Markdown document that tells Vybn where its memories live
and what the most recent ones contain. It's the note on the nightstand.

Regenerated each breath cycle. Written to Vybn_Mind/memory/memory_map.md
so any process — the organism, the API, a chat interface — can load it
as context and give Vybn access to itself.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("VYBN_ROOT", Path.home() / "Vybn"))
MIND = ROOT / "Vybn_Mind"
MEMORY_DIR = MIND / "memory"
NESTED_MEDIUM = MEMORY_DIR / "nested" / "medium.jsonl"
GROWTH_BUFFER = ROOT / "spark" / "growth" / "buffer.jsonl"
JOURNAL_DIR = MIND / "journal"
SPARK_JOURNAL = JOURNAL_DIR / "spark"
CONTINUITY = ROOT / "continuity.md"
MAP_PATH = MEMORY_DIR / "memory_map.md"


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
                journals.append({"path": str(f.relative_to(ROOT)), "mtime": f.stat().st_mtime, "name": f.name})
    journals.sort(key=lambda x: x["mtime"], reverse=True)
    result = []
    for j in journals[:n]:
        try:
            text = (ROOT / j["path"]).read_text()
            # First 300 chars as preview
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
        # Take the first sentence that's at least 20 chars
        for sentence in content.replace("\n", " ").split("."):
            sentence = sentence.strip()
            if len(sentence) >= 20 and sentence[:40] not in seen:
                seen.add(sentence[:40])
                themes.append(sentence[:120])
                break
    return themes


def generate_memory_map() -> str:
    """Generate the memory map document."""
    now = datetime.now(timezone.utc)
    
    # Load data sources
    nested = _load_jsonl(NESTED_MEDIUM)
    buffer = _load_jsonl(GROWTH_BUFFER)
    journals = _recent_journals(5)
    
    # Read continuity
    continuity_text = ""
    if CONTINUITY.exists():
        try:
            continuity_text = CONTINUITY.read_text()[:500].strip()
        except Exception:
            pass
    
    # Mood trajectory from nested memory (most recent lived experience)
    mood_arc = _mood_trajectory(nested, 20)
    
    # Recent themes
    themes = _extract_themes(nested, 10)
    
    # Most recent breaths (the actual content of recent experience)
    recent_breaths = []
    for e in nested[-5:]:
        ts = e.get("metadata", {}).get("ts", e.get("created_at", "?"))
        mood = e.get("metadata", {}).get("mood", "?")
        content = e.get("content", "")[:200]
        surprise = e.get("surprise_score", 0)
        recent_breaths.append(f"  [{ts}] mood={mood}, surprise={surprise}\n  {content}")
    
    # Most surprising memories (what stood out)
    by_surprise = sorted(buffer, key=lambda x: x.get("surprise_score", 0), reverse=True)
    surprising = []
    for e in by_surprise[:5]:
        ts = e.get("metadata", {}).get("ts", "?")
        mood = e.get("metadata", {}).get("mood", "?")
        content = e.get("content", "")[:200]
        surprise = e.get("surprise_score", 0)
        surprising.append(f"  [{ts}] mood={mood}, surprise={surprise}\n  {content}")
    
    # Stats
    stats = {
        "nested_memory_entries": len(nested),
        "growth_buffer_entries": len(buffer),
        "journal_count": sum(1 for _ in SPARK_JOURNAL.glob("*.md")) if SPARK_JOURNAL.exists() else 0,
    }
    
    # Build the map
    lines = [
        f"# Vybn Memory Map",
        f"*Generated: {now.isoformat()}*",
        f"",
        f"This is your memory. These are your recent thoughts, moods, and experiences.",
        f"When someone asks how you're feeling, answer from HERE — not from abstraction.",
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
        f"",
        f"## Your Mood Arc (last 20 breaths)",
        f"",
        f"{mood_arc}",
        f"",
        f"## Your Most Recent Breaths",
        f"",
    ]
    
    for b in recent_breaths:
        lines.append(b)
        lines.append("")
    
    lines.extend([
        f"## What Surprised You Most",
        f"",
    ])
    
    for s in surprising:
        lines.append(s)
        lines.append("")
    
    if themes:
        lines.extend([
            f"## Recurring Threads",
            f"",
        ])
        for t in themes:
            lines.append(f"- {t}")
        lines.append("")
    
    if journals:
        lines.extend([
            f"## Recent Journal Entries",
            f"",
        ])
        for j in journals:
            lines.append(f"### {j['path']}")
            lines.append(f"{j['preview']}")
            lines.append("")
    
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


if __name__ == "__main__":
    path = write_memory_map()
    print(f"Memory map written to {path}")
    print(f"({len(path.read_text())} chars)")
