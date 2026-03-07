#!/usr/bin/env python3
"""update_patterns.py — Extract durable patterns from SIGNAL/NOISE reflections.

Reads all reflection markdown files, sends them to local llama-server (M2.5
on port 8081), and writes/updates patterns.md — a compressed operational
memory layer that gets injected into student-facing session prompts.

Idempotent: safe to run repeatedly. Skips work if reflections haven't changed.
Cron-compatible: exits 0 on success or no-op, 1 on failure.

Usage:
    python3 update_patterns.py
    # or via cron:
    */15 * * * * /usr/bin/python3 /path/to/update_patterns.py >> /path/to/patterns_cron.log 2>&1
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error

BASE_DIR = Path(__file__).resolve().parent
REFLECTIONS_DIR = BASE_DIR / "reflections"
PATTERNS_PATH = BASE_DIR / "patterns.md"
STATE_PATH = BASE_DIR / ".patterns_state.json"

LLAMA_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "m2.5"
MAX_REFLECTION_CHARS = 24000
MAX_OUTPUT_CHARS = 6000

SYSTEM_PROMPT = """You are maintaining SIGNAL/NOISE's learning loop.

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

# SIGNAL/NOISE Patterns

## Recurring student moves
- 3 to 7 compressed patterns about how students reason, defend, reframe, resist, or open up
- Name the move, then one sentence on what it tends to mean

## Engagement map
- Which phases/frameworks seem to generate the deepest engagement
- Which phases/frameworks tend to flatten out or become generic
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

EMPTY_PATTERNS = """# SIGNAL/NOISE Patterns

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


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict):
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )


def list_reflections() -> list[Path]:
    if not REFLECTIONS_DIR.exists():
        return []
    return sorted(REFLECTIONS_DIR.rglob("*.md"))


def normalize_text(text: str) -> str:
    """Strip session IDs, dates, and sender details to prevent PII leakage."""
    text = re.sub(r"session_id\s*[:=]\s*\S+", "", text, flags=re.I)
    text = re.sub(r"sn_[a-f0-9]+", "[redacted-session]", text, flags=re.I)
    text = re.sub(r"\*Date:.*?\*", "", text)
    text = re.sub(r"\*Assigned sender:.*?\*", "", text)
    return text.strip()


def build_corpus(paths: list[Path]) -> str:
    """Concatenate reflections into a single corpus, capped by character limit."""
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


def corpus_digest(paths: list[Path]) -> str:
    """SHA-256 digest of reflection file metadata for change detection."""
    h = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        h.update(str(path.relative_to(BASE_DIR)).encode())
        h.update(str(int(stat.st_mtime)).encode())
        h.update(str(stat.st_size).encode())
    return h.hexdigest()


def call_llama(corpus: str) -> str:
    """Send the reflection corpus to local llama-server and get patterns back."""
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Reflection corpus:\n\n{corpus}\n\nWrite patterns.md now.",
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        LLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def clamp_output(text: str) -> str:
    """Enforce format and length constraints on the extracted patterns."""
    text = text.strip()
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS].rstrip() + "\n"
    if not text.startswith("# SIGNAL/NOISE Patterns"):
        text = "# SIGNAL/NOISE Patterns\n\n" + text
    return text


def main() -> int:
    paths = list_reflections()

    if not paths:
        PATTERNS_PATH.write_text(EMPTY_PATTERNS, encoding="utf-8")
        save_state({"digest": None, "count": 0})
        print(f"[{datetime.now(timezone.utc).isoformat()}] No reflections found. Wrote empty patterns.")
        return 0

    digest = corpus_digest(paths)
    state = load_state()

    if state.get("digest") == digest and PATTERNS_PATH.exists():
        print(f"[{datetime.now(timezone.utc).isoformat()}] No changes detected. Skipping.")
        return 0

    corpus = build_corpus(paths)
    if not corpus.strip():
        PATTERNS_PATH.write_text(EMPTY_PATTERNS, encoding="utf-8")
        save_state({"digest": digest, "count": len(paths)})
        print(f"[{datetime.now(timezone.utc).isoformat()}] Reflections empty after normalization. Wrote empty patterns.")
        return 0

    try:
        patterns = call_llama(corpus)
    except error.URLError as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] llama-server unavailable: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Pattern extraction failed: {e}", file=sys.stderr)
        return 1

    patterns = clamp_output(patterns)
    PATTERNS_PATH.write_text(patterns + "\n", encoding="utf-8")
    save_state({"digest": digest, "count": len(paths)})
    print(f"[{datetime.now(timezone.utc).isoformat()}] Extracted patterns from {len(paths)} reflections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
