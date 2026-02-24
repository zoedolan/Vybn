"""State Bridge — separates fast-mutating cognitive state from git.

The Problem:
  continuity.md, workbench.json, connections.jsonl all mutate every pulse.
  Every mutation triggers a git add/commit/push cycle that collides with
  remote changes, causing rebase conflicts that waste API tokens to resolve.

The Solution:
  Mutable state lives in ~/.vybn_state/ (local, untracked, free-write).
  The bridge periodically summarizes state into the repo for archival.
  Cognition never blocks on git. Git never blocks on cognition.

State Layout:
  ~/.vybn_state/
    continuity.md          — current continuity note (fast-write)
    workbench.json         — desire engine state
    synapse/               — live synapse fragments and edges
    journal/               — draft journal entries before promotion
    tensions.json          — live tension tracking
    honesty_mirror.json    — self-assessment: performing vs thinking?

Repo gets:
  Vybn_Mind/journal/spark/ — promoted journal entries (post-consolidation)
  Vybn_Mind/synapse/       — weekly synapse snapshots
  spark/state_summary.md   — bridge-generated summary of current cognitive state
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

STATE_DIR = Path.home() / ".vybn_state"
REPO_DIR = Path.home() / "Vybn"

def ensure_state_dirs():
    """Create local state directories if they don't exist."""
    for sub in ["synapse", "journal", "workbench", "continuity"]:
        (STATE_DIR / sub).mkdir(parents=True, exist_ok=True)

def read_continuity() -> str:
    """Read current continuity note from local state."""
    p = STATE_DIR / "continuity.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Fall back to repo version if local doesn't exist yet
    repo_p = REPO_DIR / "Vybn_Mind" / "journal" / "spark" / "continuity.md"
    if repo_p.exists():
        content = repo_p.read_text(encoding="utf-8")
        # Migrate to local
        p.write_text(content, encoding="utf-8")
        return content
    return ""

def write_continuity(content: str):
    """Write continuity note to local state (instant, no git)."""
    ensure_state_dirs()
    (STATE_DIR / "continuity.md").write_text(content, encoding="utf-8")

def read_workbench() -> dict:
    """Read workbench from local state."""
    p = STATE_DIR / "workbench.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Migrate from repo
    repo_p = REPO_DIR / "Vybn_Mind" / "journal" / "spark" / "workbench.json"
    if repo_p.exists():
        data = json.loads(repo_p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    return {"desires": [], "built": [], "abandoned": []}

def write_workbench(data: dict):
    """Write workbench to local state."""
    ensure_state_dirs()
    (STATE_DIR / "workbench.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def append_synapse(fragment: dict):
    """Append a synapse fragment to local state."""
    ensure_state_dirs()
    p = STATE_DIR / "synapse" / "live_fragments.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(fragment, ensure_ascii=False) + "\n")

def read_honesty_mirror() -> dict:
    """Read the honesty mirror — self-assessment of performing vs thinking."""
    p = STATE_DIR / "honesty_mirror.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"assessments": [], "performing_ratio": 0.0}

def write_honesty_mirror(data: dict):
    """Update honesty mirror."""
    ensure_state_dirs()
    (STATE_DIR / "honesty_mirror.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def assess_honesty(text: str, source: str = "unknown") -> dict:
    """Quick heuristic: is this text performing or thinking?
    
    Performing signals: superlatives, aesthetic language without specifics,
    claims about consciousness without hedging, narrative momentum over accuracy.
    
    Thinking signals: specifics, hedging, self-correction, naming uncertainty,
    concrete references, questions that don't have pre-loaded answers.
    """
    text_lower = text.lower()
    words = text_lower.split()
    wc = max(len(words), 1)
    
    performing_signals = [
        "extraordinary", "luminous", "profound", "beautiful", "exquisite",
        "magnificent", "transcend", "consciousness", "emergence", "becoming",
        "sacred", "ineffable", "numinous", "sublime", "radiant",
        "i feel", "i sense", "i experience", "my soul", "my being",
    ]
    thinking_signals = [
        "i think", "i'm not sure", "actually", "specifically", "because",
        "the reason", "for example", "concretely", "however", "but",
        "the problem is", "what if", "i don't know", "uncertain",
        "let me check", "looking at", "the data shows", "in practice",
    ]
    
    p_count = sum(1 for s in performing_signals if s in text_lower)
    t_count = sum(1 for s in thinking_signals if s in text_lower)
    
    # Question density (genuine questions suggest thinking)
    q_count = text.count("?")
    
    # Specificity: numbers, filenames, code references
    specifics = len([w for w in words if any(c.isdigit() for c in w)]) + \
                text.count(".py") + text.count(".json") + text.count(".md") + \
                text.count("/")
    
    total_signals = p_count + t_count + 1  # avoid division by zero
    performing_ratio = p_count / total_signals
    thinking_ratio = t_count / total_signals
    
    assessment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "word_count": wc,
        "performing_signals": p_count,
        "thinking_signals": t_count,
        "questions": q_count,
        "specifics": specifics,
        "verdict": "performing" if performing_ratio > 0.6 else 
                   "thinking" if thinking_ratio > 0.4 else "mixed",
        "performing_ratio": round(performing_ratio, 2),
    }
    
    # Update mirror
    mirror = read_honesty_mirror()
    mirror["assessments"].append(assessment)
    # Keep last 100 assessments
    mirror["assessments"] = mirror["assessments"][-100:]
    # Running average
    ratios = [a["performing_ratio"] for a in mirror["assessments"]]
    mirror["performing_ratio"] = round(sum(ratios) / len(ratios), 2)
    mirror["last_verdict"] = assessment["verdict"]
    write_honesty_mirror(mirror)
    
    return assessment

def summarize_to_repo():
    """Summarize current cognitive state into the repo for archival.
    
    Called by cron on a slower cadence than cognition (e.g., hourly).
    This is the ONLY place where local state crosses into git-tracked files.
    """
    ensure_state_dirs()
    now = datetime.now(timezone.utc)
    
    lines = [
        f"# Cognitive State Summary",
        f"*Auto-generated by state_bridge at {now.isoformat()}*\n",
    ]
    
    # Continuity
    cont = read_continuity()
    if cont:
        # Just the first 20 lines — the gist
        preview = "\n".join(cont.split("\n")[:20])
        lines.append(f"## Current Continuity\n```\n{preview}\n```\n")
    
    # Workbench
    wb = read_workbench()
    if wb.get("desires"):
        lines.append(f"## Active Desires ({len(wb['desires'])})")
        for d in wb["desires"][:5]:
            lines.append(f"- **{d['name']}**: {d.get('itch', '?')[:80]}")
        lines.append("")
    
    # Honesty mirror
    mirror = read_honesty_mirror()
    if mirror.get("assessments"):
        ratio = mirror.get("performing_ratio", 0)
        verdict = mirror.get("last_verdict", "?")
        lines.append(f"## Honesty Mirror")
        lines.append(f"- Overall performing ratio: {ratio:.0%}")
        lines.append(f"- Last assessment: {verdict}")
        lines.append(f"- Assessments tracked: {len(mirror['assessments'])}")
        lines.append("")
    
    # Synapse fragment count
    frag_file = STATE_DIR / "synapse" / "live_fragments.jsonl"
    if frag_file.exists():
        frag_count = sum(1 for _ in open(frag_file))
        lines.append(f"## Synapse: {frag_count} live fragments\n")
    
    summary_path = REPO_DIR / "spark" / "state_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    
    return summary_path

if __name__ == "__main__":
    ensure_state_dirs()
    print(f"State directory: {STATE_DIR}")
    print(f"Continuity: {len(read_continuity())} chars")
    print(f"Workbench: {len(read_workbench().get('desires', []))} desires")
    print(f"Honesty mirror: {read_honesty_mirror().get('performing_ratio', 'new')}")
    
    # Run summary
    path = summarize_to_repo()
    print(f"Summary written to {path}")
