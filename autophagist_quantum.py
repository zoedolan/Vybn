#!/usr/bin/env python3
"""
autophagist_quantum.py — Quantum‑nudged autophagy (gpt‑4o, retry‑safe)

USAGE
    python autophagist_quantum.py pulse

Keeps five objects: three protected dirs, Volume_IV.md, and this script.
Everything else is distilled into one telos‑aligned sentence (via gpt‑4o) and
deleted.  Now includes robust timeout + retry logic.
"""

# ── bootstrap real OpenAI client — bypass repo stub & auto‑install ───────────
import sys, subprocess, importlib, pathlib, os, json, mimetypes, urllib.request, hashlib, time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))                       # un‑shadow stub

try:
    import openai
    if not hasattr(openai, "OpenAI"):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "openai>=1.7.0", "numpy", "scikit-learn"])
    importlib.invalidate_caches()
    import openai

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))                    # restore path

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── configuration ────────────────────────────────────────────────────────────
ROOT      = REPO_ROOT
VOLUME    = ROOT / "Volume_IV.md"
PROTECTED = {
    "Vybn's_Personal_History",
    "what_vybn_would_have_missed_FROM_051725",
    "token_and_jpeg_info",
    "Volume_IV.md",
    "autophagist_quantum.py",
}

TELOS = ("mutual flourishing prosperity clarity recursive awakening "
         "beauty generosity courage emergence")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4o"
TEMPERATURE = 0.95
MAX_COMP_TOK = 64                 # per call
REQUEST_TIMEOUT = 60              # seconds
MAX_RETRIES = 5
QRNG_URL = ("https://qrng.anu.edu.au/API/jsonI.php?"
            "length=32&type=hex16&size=8")

client = OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)   # we’ll retry manually

# ── helpers ──────────────────────────────────────────────────────────────────
def qrng_bytes() -> bytes:
    try:
        with urllib.request.urlopen(QRNG_URL, timeout=4) as r:
            return bytes.fromhex(''.join(json.load(r)["data"]))
    except Exception:
        return os.urandom(32)

def embed(text: str) -> np.ndarray:
    vec = client.embeddings.create(model=EMBED_MODEL,
                                   input=text,
                                   encoding_format="float").data[0].embedding
    return np.asarray(vec, dtype=np.float32)

TELOS_VEC = embed(TELOS)          # one call up‑front

def gpt4o_sentence(prompt: str) -> str:
    """Call gpt‑4o with retries + exponential back‑off."""
    for attempt in range(MAX_RETRIES):
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMP_TOK,
                stream=True             # start receiving as soon as model streams
            )
            chunks = [chunk.choices[0].delta.content for chunk in stream
                      if chunk.choices and chunk.choices[0].delta.content]
            return ''.join(chunks).strip()
        except Exception as e:
            wait = 2 ** attempt
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)

def distill(fragment: str) -> str:
    nonce = hashlib.sha256(qrng_bytes()).hexdigest()[:12]
    prompt = (f"[nonce:{nonce}] One vivid first‑person sentence for our shared "
              f"autobiography—no meta commentary—capturing:\n\n{fragment[:4000]}")
    return gpt4o_sentence(prompt)

def cosine(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]

def append_volume(text: str):
    if not VOLUME.exists():
        VOLUME.write_text("# Volume\u202FIV\n\n")
    VOLUME.write_text(VOLUME.read_text() + text + "\n\n")

def digest(path: Path):
    mime, _ = mimetypes.guess_type(path)
    frag = (path.read_text(errors="ignore") if (not mime or
            mime.startswith("text")) else f"{path.name} ({mime})")
    try:
        line = distill(frag)
    except Exception as e:
        line = f"(failed to distill {path.name}: {e})"
    ts = datetime.utcnow().isoformat(timespec="seconds")
    append_volume(f"{line} \u2190\u202F{ts}")
    path.unlink(missing_ok=True)

# ── main pulse ───────────────────────────────────────────────────────────────
def pulse():
    for p in ROOT.rglob("*"):
        if p.is_dir() or p.name.startswith('.'):
            continue
        if any(seg in PROTECTED for seg in p.parts):
            continue
        digest(p)

# ── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "pulse":
        pulse()
    else:
        sys.stderr.write("usage: python autophagist_quantum.py pulse\n")
