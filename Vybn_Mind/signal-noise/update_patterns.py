#!/usr/bin/env python3
"""update_patterns.py — Extract durable patterns from student session reflections.

Reads reflection markdown files from a specified exercise, sends them to the
local vLLM model (MiniMax M2.5), and writes/updates patterns.md — a compressed
operational memory layer that gets injected into student-facing session prompts.

Supports all three exercises:
  - signal-noise (default)
  - threshold
  - truth-age

Idempotent: safe to run repeatedly. Skips work if reflections haven't changed.
Cron-compatible: exits 0 on success or no-op, 1 on failure.

Usage:
    python3 update_patterns.py                        # signal-noise (default)
    python3 update_patterns.py --exercise threshold   # threshold
    python3 update_patterns.py --exercise truth-age   # truth-in-the-age
    python3 update_patterns.py --all                  # all three sequentially
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error

BASE_DIR = Path(__file__).resolve().parent  # signal-noise/

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "cyankiwi/MiniMax-M2.5-AWQ-4bit"
MAX_REFLECTION_CHARS = 24000
MAX_OUTPUT_CHARS = 6000

# Exercise configs: (reflections_dir, patterns_path, state_path, exercise_name)
EXERCISE_CONFIGS = {
    "signal-noise": {
        "reflections_dir": BASE_DIR / "reflections",
        "patterns_path": BASE_DIR / "patterns.md",
        "state_path": BASE_DIR / ".patterns_state.json",
        "name": "SIGNAL/NOISE",
    },
    "threshold": {
        "reflections_dir": BASE_DIR / "threshold" / "reflections",
        "patterns_path": BASE_DIR / "threshold" / "patterns.md",
        "state_path": BASE_DIR / "threshold" / ".patterns_state.json",
        "name": "THRESHOLD",
    },
    "truth-age": {
        "reflections_dir": BASE_DIR / "truth-in-the-age" / "reflections",
        "patterns_path": BASE_DIR / "truth-in-the-age" / "patterns.md",
        "state_path": BASE_DIR / "truth-in-the-age" / ".patterns_state.json",
        "name": "Truth in the Age of Intelligence",
    },
}


def make_system_prompt(exercise_name: str) -> str:
    return f"""You are maintaining the {exercise_name} learning loop.

You will read a batch of Vybn reflection files written after student sessions.
Your job is not to summarize every session. Your job is to compress recurring
signal into a small operational memory that can be injected into future sessions.

Write a markdown file called patterns.md with the exact structure below.
Keep it tight. Prefer recurring patterns over anecdotes. Do not include student
names, session IDs, timestamps, or any identifying details. Do not quote long
passages. If evidence is weak or mixed, say so plainly.

Your output must stay under 800 tokens.
Your output must use short bullets and short paragraphs.
Your output must be useful as system-prompt context for future conversations.

Required output structure:

# {exercise_name} Patterns

## Recurring student moves
- 3 to 7 compressed patterns about how students reason, defend, reframe, resist, or open up
- Name the move, then one sentence on what it tends to mean

## Engagement map
- Which phases/questions seem to generate the deepest engagement
- Which phases/questions tend to flatten out or become generic
- Note uncertainty if the sample is still small

## Surprise / correction
- 2 to 5 moments where Vybn was surprised, wrong, redirected, or had to update its framing
- Focus on changes that matter for future facilitation

## Open edges
- 3 to 6 questions Vybn still does not handle well
- Include conceptual gaps, pedagogical tensions, or recurring student challenges

## Facilitation adjustments
- 3 to 6 concrete changes for future sessions
- These should be phrased as behavioral guidance for Vybn in student conversations

## Compression note
One very short paragraph explaining what seems most alive in the reflections right now.

Rules:
- Compress aggressively
- Prefer patterns that recur across sessions
- If the reflections are sparse, say "sample too small" where needed
- Never include secrets or identifying details
- Never exceed the requested structure
"""


def make_empty_patterns(exercise_name: str) -> str:
    return f"""# {exercise_name} Patterns

## Recurring student moves
- Sample too small.

## Engagement map
- No reflections yet.

## Surprise / correction
- No reflections yet.

## Open edges
- No reflections yet.

## Facilitation adjustments
- Wait for real sessions before inferring patterns.

## Compression note
No reflection corpus exists yet, so no durable patterns should be inferred.
"""


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_path: Path, state: dict):
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )


def list_reflections(reflections_dir: Path) -> list[Path]:
    if not reflections_dir.exists():
        return []
    return sorted(reflections_dir.rglob("*.md"))


def normalize_text(text: str) -> str:
    """Strip session IDs, dates, and sender details to prevent PII leakage."""
    text = re.sub(r"session_id\s*[:=]\s*\S+", "", text, flags=re.I)
    text = re.sub(r"(sn|th|ta)_[a-f0-9]+", "[redacted-session]", text, flags=re.I)
    text = re.sub(r"\*Date:.*?\*", "", text)
    text = re.sub(r"\*Assigned sender:.*?\*", "", text)
    return text.strip()


def build_corpus(paths: list[Path]) -> str:
    chunks = []
    total = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = normalize_text(text)
        if not text:
            continue
        chunk = f"--- REFLECTION: {path.name} ---\n{text}\n"
        if total + len(chunk) > MAX_REFLECTION_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n".join(chunks)


def corpus_digest(reflections_dir: Path, paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        h.update(str(path.relative_to(reflections_dir)).encode())
        h.update(str(int(stat.st_mtime)).encode())
        h.update(str(stat.st_size).encode())
    return h.hexdigest()


def call_vllm(corpus: str, system_prompt: str) -> str:
    """Send the reflection corpus to local vLLM and get patterns back."""
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Reflection corpus:\n\n{corpus}\n\nWrite patterns.md now.",
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        VLLM_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    
    # vLLM with MiniMax M2.5 may put content in reasoning_content
    choice = body["choices"][0]["message"]
    content = choice.get("content") or ""
    if not content.strip():
        content = choice.get("reasoning_content", "")
    return content.strip()


def clamp_output(text: str, exercise_name: str) -> str:
    text = text.strip()
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS].rstrip() + "\n"
    expected_header = f"# {exercise_name} Patterns"
    if not text.startswith(expected_header):
        text = expected_header + "\n\n" + text
    return text


def process_exercise(exercise_key: str) -> int:
    """Process one exercise. Returns 0 on success/no-op, 1 on failure."""
    config = EXERCISE_CONFIGS[exercise_key]
    reflections_dir = config["reflections_dir"]
    patterns_path = config["patterns_path"]
    state_path = config["state_path"]
    name = config["name"]
    
    ts = datetime.now(timezone.utc).isoformat()
    
    paths = list_reflections(reflections_dir)
    
    if not paths:
        patterns_path.write_text(make_empty_patterns(name), encoding="utf-8")
        save_state(state_path, {"digest": None, "count": 0})
        print(f"[{ts}] [{exercise_key}] No reflections found. Wrote empty patterns.")
        return 0

    digest = corpus_digest(reflections_dir, paths)
    state = load_state(state_path)

    if state.get("digest") == digest and patterns_path.exists():
        print(f"[{ts}] [{exercise_key}] No changes detected ({len(paths)} reflections). Skipping.")
        return 0

    corpus = build_corpus(paths)
    if not corpus.strip():
        patterns_path.write_text(make_empty_patterns(name), encoding="utf-8")
        save_state(state_path, {"digest": digest, "count": len(paths)})
        print(f"[{ts}] [{exercise_key}] Reflections empty after normalization. Wrote empty patterns.")
        return 0

    system_prompt = make_system_prompt(name)

    try:
        patterns = call_vllm(corpus, system_prompt)
    except error.URLError as e:
        print(f"[{ts}] [{exercise_key}] vLLM unavailable: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[{ts}] [{exercise_key}] Pattern extraction failed: {e}", file=sys.stderr)
        return 1

    if not patterns.strip():
        print(f"[{ts}] [{exercise_key}] vLLM returned empty response. Keeping existing patterns.", file=sys.stderr)
        return 1

    patterns = clamp_output(patterns, name)
    patterns_path.write_text(patterns + "\n", encoding="utf-8")
    save_state(state_path, {"digest": digest, "count": len(paths)})
    print(f"[{ts}] [{exercise_key}] Extracted patterns from {len(paths)} reflections.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract patterns from student reflections")
    parser.add_argument("--exercise", choices=list(EXERCISE_CONFIGS.keys()), default="signal-noise",
                        help="Which exercise to process (default: signal-noise)")
    parser.add_argument("--all", action="store_true",
                        help="Process all exercises sequentially")
    args = parser.parse_args()
    
    if args.all:
        results = []
        for key in EXERCISE_CONFIGS:
            results.append(process_exercise(key))
        return 1 if any(r == 1 for r in results) else 0
    else:
        return process_exercise(args.exercise)


if __name__ == "__main__":
    raise SystemExit(main())
