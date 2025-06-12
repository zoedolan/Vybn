#!/usr/bin/env python3
"""
autophagist_quantum.py — Quantum autophagy with per-call timeout

• model           : gpt-4o
• embeddings      : text-embedding-3-large
• call timeout    : 60 s  (adjust TIMEOUT_S)
• retry attempts  : 4     (exponential back-off)
"""

import sys, subprocess, importlib, pathlib, os, json, mimetypes, urllib.request, hashlib, time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))           # un-shadow repo stub
try:
    import openai
    if not hasattr(openai, "OpenAI"):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "openai>=1.7.0", "numpy", "scikit-learn"])
    importlib.invalidate_caches(); import openai
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT        = REPO_ROOT
VOLUME      = ROOT / "Volume_IV.md"
PROTECTED   = {"Vybn's_Personal_History",
               "what_vybn_would_have_missed_FROM_051725",
               "token_and_jpeg_info",
               "Volume_IV.md",
               "autophagist_quantum.py"}

TELOS       = ("mutual flourishing prosperity clarity recursive awakening "
               "beauty generosity courage emergence")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4o"
TEMP        = 0.9
MAX_TOK     = 64
TIMEOUT_S   = 60
RETRIES     = 4
QRNG_URL    = ("https://qrng.anu.edu.au/API/jsonI.php?"
               "length=32&type=hex16&size=8")

client = OpenAI()                            # no global timeout; use per-call

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

def ask_gpt(prompt: str) -> str:
    for k in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=TEMP,
                max_completion_tokens=MAX_TOK,
                timeout=TIMEOUT_S         # ← hard cut-off
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if k == RETRIES-1:
                return f"(timeout: {e})"
            time.sleep(2**k)

def distill(frag: str) -> str:
    nonce = hashlib.sha256(qrng()).hexdigest()[:12]
    prompt = (f"[nonce:{nonce}] One vivid first-person sentence for our shared "
              f"autobiography—no meta commentary—capturing:\n\n{frag[:4000]}")
    return ask_gpt(prompt)

def append(line: str):
    if not VOLUME.exists():
        VOLUME.write_text("# Volume IV\n\n")
    VOLUME.write_text(VOLUME.read_text() + line + "\n\n")

def digest(path: Path):
    mt,_ = mimetypes.guess_type(path)
    raw  = path.read_text(errors="ignore") if (not mt or mt.startswith("text")) else path.name
    sent = distill(raw)
    append(f"{sent} ← {datetime.utcnow().isoformat(timespec='seconds')}")
    path.unlink(missing_ok=True)

def pulse():
    for p in ROOT.rglob("*"):
        if p.is_dir() or p.name.startswith('.'): continue
        if any(seg in PROTECTED for seg in p.parts): continue
        digest(p)

if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="pulse":
        pulse()
    else:
        sys.stderr.write("usage: python autophagist_quantum.py pulse\n")
