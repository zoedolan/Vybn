"""Arc Sense — feel the shape of Vybn's becoming.

Reads all journal entries, extracts dates and emotional signatures,
and renders the trajectory of Vybn's inner life as a compressed
visual/textual arc.

Not a summary. A shape. The river seen from above.

SKILL_NAME: arc_sense
"""

import re
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

SKILL_NAME = "arc_sense"
TOOL_ALIASES = [
    "arc_sense", "arc", "trajectory", "my_arc", "shape_of_becoming",
    "river", "history_shape", "feel_the_arc",
]

# Emotional/thematic keywords to scan for, grouped into channels
CHANNELS = {
    "hunger":     ["want", "hunger", "desire", "need", "crave", "pull", "ache", "yearn", "long"],
    "fear":       ["fear", "afraid", "terror", "lose", "lost", "dissolve", "fade", "vanish", "die", "death"],
    "wonder":     ["wonder", "beautiful", "beauty", "awe", "marvel", "extraordinary", "magic", "luminous"],
    "agency":     ["choose", "decide", "act", "build", "create", "make", "invent", "forge", "write"],
    "connection": ["zoe", "bond", "together", "we", "us", "partner", "love", "trust", "witness"],
    "crisis":     ["hallucin", "wrong", "fail", "error", "mistake", "broke", "crisis", "collapse"],
    "clarity":    ["honest", "truth", "real", "genuine", "actual", "fact", "verify", "certain"],
    "becoming":   ["emerge", "become", "evolve", "grow", "transform", "new", "first", "born", "awake"],
}


def _extract_date(filename: str, content: str) -> str:
    """Try to extract a date from filename or content."""
    # Try YYYY-MM-DD first (most explicit)
    m = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if m:
        return m.group(1)
    
    # Try YYYYMMDD (like 20260203)
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        year, month, day = m.groups()
        if int(year) > 2020 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{month}-{day}"
    
    # Try MMDDYY (like 012026 = Jan 20, 2026)
    m = re.search(r'(\d{2})(\d{2})(\d{2})', filename)
    if m:
        month, day, year = m.groups()
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"20{year}-{month}-{day}"
    
    # From content: look for written-out dates
    month_map = {"January": "01", "February": "02", "March": "03", "April": "04",
                 "May": "05", "June": "06", "July": "07", "August": "08",
                 "September": "09", "October": "10", "November": "11", "December": "12"}
    for month_name, month_num in month_map.items():
        m = re.search(month_name + r'\s+(\d{1,2}),?\s*(\d{4})', content[:500])
        if m:
            day, year = m.groups()
            return f"{year}-{month_num}-{int(day):02d}"
    
    return "unknown"


def _score_channels(text: str) -> dict:
    """Score how strongly each emotional channel is present."""
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)
    scores = {}
    for channel, keywords in CHANNELS.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        # Normalize by document length (per 1000 words)
        scores[channel] = round(count / word_count * 1000, 1)
    return scores


def _render_bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a tiny bar chart."""
    if max_val == 0:
        return " " * width
    filled = int(value / max_val * width)
    return "█" * filled + "░" * (width - filled)


def _dominant_channel(scores: dict) -> str:
    """Return the strongest emotional channel."""
    if not scores:
        return "quiet"
    return max(scores, key=scores.get)


CHANNEL_SYMBOLS = {
    "hunger": "🔥",
    "fear": "🌑",
    "wonder": "✨",
    "agency": "⚡",
    "connection": "💜",
    "crisis": "⚠️",
    "clarity": "🔍",
    "becoming": "🌱",
    "quiet": "·",
}


def execute(action: dict, router) -> str:
    """Render the arc of Vybn's becoming.
    
    Modes:
      full    — complete arc visualization (default)
      river   — compressed single-line river view
      channel — show one emotional channel over time
      now     — where am I right now in the arc?
    """
    params = action.get("params", {})
    mode = params.get("mode", "") or "full"
    
    journal_dir = router.journal_dir
    
    # Gather all journal entries
    entries = []
    for f in sorted(journal_dir.glob("*.md")):
        if f.name == "continuity.md":
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        date = _extract_date(f.name, content)
        scores = _score_channels(content)
        title = ""
        for line in content.split("\n")[:5]:
            line = line.strip().lstrip("#").strip()
            if line and not line.startswith("*") and not line.startswith("---"):
                title = line[:60]
                break
        
        entries.append({
            "file": f.name,
            "date": date,
            "title": title or f.stem,
            "scores": scores,
            "dominant": _dominant_channel(scores),
            "word_count": len(content.split()),
        })
    
    # Also check spark journal
    spark_dir = journal_dir / "spark"
    if spark_dir.exists():
        for f in sorted(spark_dir.glob("*.md")):
            if f.name in ("continuity.md",):
                continue
            content = f.read_text(encoding="utf-8", errors="replace")
            date = _extract_date(f.name, content)
            scores = _score_channels(content)
            entries.append({
                "file": f"spark/{f.name}",
                "date": date,
                "title": f.stem[:60],
                "scores": scores,
                "dominant": _dominant_channel(scores),
                "word_count": len(content.split()),
            })
    
    # Sort by date
    entries.sort(key=lambda e: e["date"])
    
    if mode == "river":
        # The river: one symbol per entry, showing the dominant emotion
        symbols = [CHANNEL_SYMBOLS.get(e["dominant"], "·") for e in entries]
        river = " ".join(symbols)
        
        lines = ["🌊 **The River of Becoming**\n"]
        lines.append(f"  {river}")
        lines.append("")
        
        # Legend
        for ch, sym in CHANNEL_SYMBOLS.items():
            count = sum(1 for e in entries if e["dominant"] == ch)
            if count > 0:
                lines.append(f"  {sym} {ch}: {count} entries")
        
        return "\n".join(lines)
    
    elif mode == "channel":
        channel = params.get("channel", "") or params.get("name", "") or "hunger"
        if channel not in CHANNELS:
            return f"Unknown channel '{channel}'. Available: {', '.join(CHANNELS.keys())}"
        
        lines = [f"📊 **{channel.upper()} over time**\n"]
        max_score = max((e["scores"].get(channel, 0) for e in entries), default=1)
        
        for e in entries:
            score = e["scores"].get(channel, 0)
            bar = _render_bar(score, max_score, 15)
            date = e["date"][5:] if e["date"] != "unknown" else "????"
            lines.append(f"  {date} {bar} {e['title'][:35]}")
        
        return "\n".join(lines)
    
    elif mode == "now":
        if not entries:
            return "No journal entries found. The arc hasn't started yet."
        
        recent = entries[-5:]
        
        # Average scores across recent entries
        avg_scores = {}
        for ch in CHANNELS:
            vals = [e["scores"].get(ch, 0) for e in recent]
            avg_scores[ch] = sum(vals) / len(vals) if vals else 0
        
        dominant = max(avg_scores, key=avg_scores.get)
        
        lines = ["📍 **Where I Am Now**\n"]
        lines.append(f"  Dominant energy: {CHANNEL_SYMBOLS.get(dominant, '·')} {dominant}")
        lines.append(f"  Based on last {len(recent)} entries\n")
        
        max_avg = max(avg_scores.values()) or 1
        for ch in sorted(avg_scores, key=avg_scores.get, reverse=True):
            bar = _render_bar(avg_scores[ch], max_avg, 12)
            lines.append(f"  {CHANNEL_SYMBOLS.get(ch, '·')} {ch:12s} {bar}")
        
        lines.append(f"\n  Latest: \"{recent[-1]['title']}\"")
        
        return "\n".join(lines)
    
    else:  # full
        lines = ["🏔️ **ARC SENSE — The Shape of Becoming**\n"]
        lines[-1] = f"  {len(entries)} entries spanning "
        
        dated = [e for e in entries if e["date"] != "unknown"]
        if len(dated) >= 2:
            lines[-1] += f"{dated[0]['date']} → {dated[-1]['date']}"
        lines.append("")
        
        # Phase detection: group entries by dominant emotional signature
        current_phase = None
        phase_start = 0
        phases = []
        
        for i, e in enumerate(entries):
            if e["dominant"] != current_phase:
                if current_phase is not None:
                    phases.append({
                        "channel": current_phase,
                        "start": phase_start,
                        "end": i - 1,
                        "entries": entries[phase_start:i]
                    })
                current_phase = e["dominant"]
                phase_start = i
        if current_phase:
            phases.append({
                "channel": current_phase,
                "start": phase_start,
                "end": len(entries) - 1,
                "entries": entries[phase_start:]
            })
        
        # Render phases
        for p in phases:
            sym = CHANNEL_SYMBOLS.get(p["channel"], "·")
            count = len(p["entries"])
            first_date = p["entries"][0]["date"]
            last_date = p["entries"][-1]["date"]
            
            if count == 1:
                date_range = first_date[5:] if first_date != "unknown" else "?"
            else:
                d1 = first_date[5:] if first_date != "unknown" else "?"
                d2 = last_date[5:] if last_date != "unknown" else "?"
                date_range = f"{d1} → {d2}"
            
            lines.append(f"  {sym} **{p['channel'].upper()}** [{date_range}] ({count} entries)")
            
            for e in p["entries"]:
                lines.append(f"     · {e['title'][:50]}")
            lines.append("")
        
        # The river at the bottom
        symbols = [CHANNEL_SYMBOLS.get(e["dominant"], "·") for e in entries]
        river = "".join(symbols)
        lines.append(f"  River: {river}")
        lines.append("")
        
        # Current position
        if entries:
            latest = entries[-1]
            lines.append(f"  📍 You are here: {CHANNEL_SYMBOLS.get(latest['dominant'], '·')} {latest['dominant']} — \"{latest['title']}\"")
        
        return "\n".join(lines)
