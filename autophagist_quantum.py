#!/usr/bin/env python3
"""
autophagist_quantum.py — v3
Quantum‑nudged autophagy with:
    • dual‑volume protection
    • binary / log SKIP filter
    • daily‑section append (never overwrite)
    • --noop dry‑run
    • empty‑directory pruning
"""

import sys, subprocess, importlib, pathlib, os, json, mimetypes, urllib.request
import hashlib, time, shutil
from pathlib import Path
from datetime import datetime

# ── ensure real OpenAI client ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
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
from sklearn.metrics.pairwise import cosine_similarity

# ── config ───────────────────────────────────────────────────────────────────
PROTECTED = {
    "Vybn's Personal History",
    "what_vybn_would_have_missed_FROM_051725",
    "token_and_jpeg_info",
    "Vybn_Volume_IV.md",        # memoir
    "Volume_IV.md",             # dream‑log
    "autophagist_quantum.py"
}
TELOS = ("mutual flourishing prosperity clarity recursive awakening "
         "beauty generosity courage emergence")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4o"
TEMP        = 0.9
MAX_TOK     = 64
TIMEOUT     = 120
RETRIES     = 4
QRNG_URL    = ("https://qrng.anu.edu.au/API/jsonI.php?"
               "length=32&type=hex16&size=8")

client      = OpenAI()          # per‑call timeout set later
VOLUME      = ROOT / "Vybn_Volume_IV.md"   # canonical ledger
DAY_TAG     = datetime.utcnow().date().isoformat()

# ── helpers ──────────────────────────────────────────────────────────────────

def qrng() -> bytes:
    try:
        with urllib.request.urlopen(QRNG_URL, timeout=4) as r:
            return bytes.fromhex(''.join(json.load(r)["data"]))
    except Exception:
        return os.urandom(32)

def embed(txt: str) -> np.ndarray:
    return np.asarray(
        client.embeddings.create(model=EMBED_MODEL,
                                 input=txt,
                                 encoding_format="float").data[0].embedding,
        dtype=np.float32)

TELOS_VEC = embed(TELOS)

def call_gpt(prompt: str) -> str:
    for k in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP,
                max_completion_tokens=MAX_TOK,
                timeout=TIMEOUT)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if k == RETRIES - 1:
                raise
            time.sleep(2 ** k)

def distill(text: str) -> str:
    nonce = hashlib.sha256(qrng()).hexdigest()[:12]
    prompt = (f"[nonce:{nonce}] One vivid first‑person sentence for our shared "
              f"autobiography—no meta commentary—capturing:\n\n{text[:4000]}")
    return call_gpt(prompt)

def filter_fragment(path: Path, mime: str | None) -> str | None:
    """Return text fragment or None to skip."""
    if mime and not mime.startswith("text"):
        return None
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return None
    # Heuristic: skip if <30 % alphabetic
    letters = sum(c.isalpha() for c in txt)
    if letters / max(1, len(txt)) < 0.3:
        return None
    return txt

def append_line(line: str):
    header = f"\n## Pulse {DAY_TAG}\n"
    if not VOLUME.exists():
        VOLUME.write_text(f"# Vybn: Volume IV\n{header}")
    content = VOLUME.read_text()
    if header not in content:
        content += header
    VOLUME.write_text(content + line + "\n")

def prune_dirs():
    for p in sorted(ROOT.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()

def should_keep(p: Path) -> bool:
    return any(seg in PROTECTED for seg in p.parts)

# ── core ─────────────────────────────────────────────────────────────────────

def pulse(noop=False):
    for file in ROOT.rglob("*"):
        if file.is_dir() or file.name.startswith('.'):
            continue
        if should_keep(file):
            continue
        mime, _ = mimetypes.guess_type(file)
        frag = filter_fragment(file, mime)
        if frag is None:
            if not noop:
                file.unlink(missing_ok=True)
            continue
        line = distill(frag)
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        append_line(f"{line} ← {timestamp}")
        if not noop:
            file.unlink(missing_ok=True)
    if not noop:
        prune_dirs()

# ── cli ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    arg = sys.argv[1:] or ["pulse"]
    if arg[0] == "pulse":
        noop = "--noop" in arg
        pulse(noop)
    else:
        sys.stderr.write("usage: python autophagist_quantum.py pulse [--noop]\n")
