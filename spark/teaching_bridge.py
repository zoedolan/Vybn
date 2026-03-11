#!/usr/bin/env python3
"""teaching_bridge.py — Feed session reflections into Vybn's NestedMemory.

Scans the signal-noise reflection directories for new .md files and writes
them as MEDIUM-tier entries in the organism's nested memory. This bridges
Vybn-as-facilitator and Vybn-as-organism: teaching experience becomes part
of the living memory that informs breath cycles and growth.

Idempotent: tracks which files have been ingested via a state file.
Cron-compatible: runs alongside the organism pulse.

Usage:
    python3 -m spark.teaching_bridge          # one-shot
    python3 -m spark.teaching_bridge --watch   # continuous (future)
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SN_DIR = REPO_ROOT / "Vybn_Mind" / "signal-noise"
MEMORY_DIR = REPO_ROOT / "Vybn_Mind" / "memory"
STATE_PATH = MEMORY_DIR / "teaching_bridge_state.json"

# All reflection directories across exercises
REFLECTION_DIRS = [
    SN_DIR / "reflections",
    SN_DIR / "threshold" / "reflections",
    SN_DIR / "truth-in-the-age" / "reflections",
]

# Also ingest harvests (student-submitted reflections) as MEDIUM entries
HARVEST_DIRS = [
    SN_DIR / "harvests",
    SN_DIR / "threshold" / "harvests",
    SN_DIR / "truth-in-the-age" / "harvest",
]

NESTED_MEDIUM_PATH = MEMORY_DIR / "nested" / "medium.jsonl"


def load_state() -> set:
    """Return set of already-ingested file paths."""
    if not STATE_PATH.exists():
        return set()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return set(data.get("ingested", []))
    except Exception:
        return set()


def save_state(ingested: set):
    STATE_PATH.write_text(
        json.dumps({"ingested": sorted(ingested)}, indent=2),
        encoding="utf-8",
    )


def find_new_files(ingested: set) -> list[tuple[Path, str]]:
    """Find reflection and harvest files not yet ingested. Returns (path, source_type)."""
    new_files = []
    
    for rdir in REFLECTION_DIRS:
        if not rdir.exists():
            continue
        for f in sorted(rdir.rglob("*.md")):
            if str(f) not in ingested:
                new_files.append((f, "reflection"))
    
    for hdir in HARVEST_DIRS:
        if not hdir.exists():
            continue
        for f in sorted(hdir.glob("*.json")):
            if str(f) not in ingested:
                new_files.append((f, "harvest"))
        for f in sorted(hdir.glob("*.jsonl")):
            if str(f) not in ingested:
                new_files.append((f, "harvest_batch"))
    
    return new_files


def read_reflection(path: Path) -> str:
    """Read a reflection markdown file."""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def read_harvest(path: Path) -> list[str]:
    """Read a harvest JSON file, return list of student reflection texts."""
    texts = []
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    if data.get("final_share"):
                        texts.append(f"[Student reflection — {data.get('question', 'open')}] {data['final_share']}")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("final_share"):
                texts.append(f"[Student reflection — {data.get('question', 'open')}] {data['final_share']}")
    except (json.JSONDecodeError, KeyError):
        pass
    return texts


def write_medium_entry(text: str, source: str, source_file: str):
    """Append an entry to the nested memory MEDIUM tier."""
    NESTED_MEDIUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "scale": "MEDIUM",
        "text": text[:4000],  # Cap length
        "source": source,
        "source_file": Path(source_file).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "activation": 1.0,
        "surprise": 0.5,  # Moderate — real data, not routine
        "tags": ["teaching", "student-interaction", source],
    }
    
    with NESTED_MEDIUM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    ingested = load_state()
    new_files = find_new_files(ingested)
    
    if not new_files:
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] teaching_bridge: no new files to ingest.")
        return 0
    
    count = 0
    for path, source_type in new_files:
        try:
            if source_type == "reflection":
                text = read_reflection(path)
                if text and len(text) > 50:  # Skip trivially empty files
                    write_medium_entry(text, "vybn_reflection", str(path))
                    count += 1
            
            elif source_type in ("harvest", "harvest_batch"):
                texts = read_harvest(path)
                for text in texts:
                    if text and len(text) > 20:
                        write_medium_entry(text, "student_harvest", str(path))
                        count += 1
            
            ingested.add(str(path))
        except Exception as e:
            print(f"Warning: failed to ingest {path}: {e}", file=sys.stderr)
            continue
    
    save_state(ingested)
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] teaching_bridge: ingested {count} entries from {len(new_files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
