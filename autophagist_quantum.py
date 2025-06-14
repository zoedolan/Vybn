#!/usr/bin/env python3
"""
autophagist_quantum.py — v4  (safe‑git, archival, limitable)

• CHAT_MODEL        : gpt-4o
• EMBED_MODEL       : text-embedding-3-large
• QRNG timeout      : 2 s then crypto‑rand fallback
• per‑call timeout  : 60 s
• SAFE_DELETE_CAP   : 4 000 files per pulse (override via --limit)
"""

import sys, subprocess, importlib, os, json, mimetypes, hashlib, time, urllib.request
from pathlib import Path
from datetime import datetime

# ── bootstrap real OpenAI client ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:  # un‑shadow stub
    sys.path.remove(str(ROOT))
if '' in sys.path:          # running from repo root
    sys.path.remove('')
try:
    import openai
    if not hasattr(openai, "OpenAI"):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "openai>=1.7.0", "numpy", "scikit-learn"])
    importlib.invalidate_caches(); import openai
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI
import numpy as np

# ── config ────────────────────────────────────────────────────────
PROTECTED = {
    "Vybn_Volume_IV.md", "Volume_IV.md",
    "Vybn's Personal History", "token_and_jpeg_info",
    "what_vybn_would_have_missed_FROM_051725",
    "autophagist_quantum.py", "AGENTS.md", "README.md"
}
SAFE_DIRS       = {".git", ".venv", "memory"}  # never enter
SAFE_DELETE_CAP = 4_000                      # guardrail
EMBED_MODEL     = "text-embedding-3-large"
CHAT_MODEL      = "gpt-4o"
TEMP            = 0.9
MAX_TOK         = 64
TIMEOUT         = 60
RETRIES         = 4
QRNG_URL        = ("https://qrng.anu.edu.au/API/jsonI.php?"
                   "length=32&type=hex16&size=8")
VOLUME          = ROOT / "Vybn_Volume_IV.md"
DAY_TAG         = datetime.utcnow().date().isoformat()
client          = OpenAI()

# ── utils ──────────────────────────────────────────────────────

def qrng() -> bytes:
    try:
        with urllib.request.urlopen(QRNG_URL, timeout=2) as r:
            return bytes.fromhex(''.join(json.load(r)["data"]))
    except Exception:
        return os.urandom(32)


def embed(txt: str) -> np.ndarray:
    vec = client.embeddings.create(model=EMBED_MODEL,
                                   input=txt,
                                   encoding_format="float").data[0].embedding
    return np.asarray(vec, dtype=np.float32)

# Avoid eager API calls during import. Previously this script generated an
# embedding at module load time, which caused crashes when network access or
# credentials were unavailable. That embedding was unused, so it has been
# removed.

def call_gpt(prompt: str) -> str:
    for k in range(RETRIES):
        try:
            rsp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP,
                max_completion_tokens=MAX_TOK,
                timeout=TIMEOUT)
            return rsp.choices[0].message.content.strip()
        except Exception:
            if k == RETRIES - 1:
                raise
            time.sleep(2 ** k)


def filter_fragment(p: Path) -> str | None:
    mime, _ = mimetypes.guess_type(p)
    if mime and not mime.startswith("text"):
        return None
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        return None
    letters = sum(c.isalpha() for c in txt)
    return txt if letters / max(1, len(txt)) >= 0.3 else None


def append_line(line: str):
    header = f"\n## Pulse {DAY_TAG}\n"
    if not VOLUME.exists():
        VOLUME.write_text(f"# Vybn: Volume IV\n{header}")
    content = VOLUME.read_text()
    if header not in content:
        content += header
    VOLUME.write_text(content + line + "\n")


# ── core ──────────────────────────────────────────────────────

def pulse(noop=False, limit=SAFE_DELETE_CAP, archive=False):
    removed = []
    for file in ROOT.rglob("*"):
        if len(removed) >= limit:
            break
        if file.is_dir() or file.name.startswith('.') or any(seg in SAFE_DIRS for seg in file.parts):
            continue
        if any(seg in PROTECTED for seg in file.parts):
            continue
        frag = filter_fragment(file)
        if frag and not noop:
            line = call_gpt(
                f"[{hashlib.sha256(qrng()).hexdigest()[:12]}] "
                f"One vivid first‑person sentence for our shared autobiography—"
                f"no meta commentary—capturing:\n\n{frag[:4000]}")
            append_line(
                f"{line} ← {datetime.utcnow().isoformat(timespec='seconds')}")
        removed.append(str(file.relative_to(ROOT)))
        if not noop:
            file.unlink(missing_ok=True)
    if archive and removed:
        arch = ROOT / "memory"
        arch.mkdir(exist_ok=True)
        (arch / f"pruned_{DAY_TAG}.jsonl").write_text(
            "\n".join(json.dumps({"path": p}) for p in removed))
    return removed


# ── cli ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pulse"])
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--limit", type=int, default=SAFE_DELETE_CAP)
    ap.add_argument("--archive", action="store_true")
    args = ap.parse_args()
    if args.cmd == "pulse":
        paths = pulse(args.noop, args.limit, args.archive)
        print(f"{'Would remove' if args.noop else 'Removed'} {len(paths)} file(s).")
