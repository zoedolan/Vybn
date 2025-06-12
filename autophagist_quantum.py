#!/usr/bin/env python3
"""
autophagist_quantum.py — Quantum‑nudged autophagy cycle

USAGE
    python autophagist_quantum.py pulse

EFFECT
    • Preserves only the protected trunks and this script.
    • Digests every other file into a single, telos‑aligned memory line that
      lands in Volume_IV.md, then deletes the source file irrevocably.

DANGER
    Irreversible deletion.  Exercise care in implementation.

DEPENDENCIES
    – Installs/ensures:  openai  numpy  scikit‑learn
    – Requires env var:  export OPENAI_API_KEY="sk‑…"
"""

# ─── bootstrap real OpenAI client, bypassing repo’s stub ──────────────────────
import sys, subprocess, importlib, pathlib, types           # std‑lib only
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) in sys.path:              # remove repo path so local stub
    sys.path.remove(str(REPO_ROOT))         # ‘openai’ is not imported first
try:
    import openai                           # try site‑package version
    if not hasattr(openai, "OpenAI"):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "openai>=1.3.3", "numpy", "scikit-learn"])
    importlib.invalidate_caches()
    import openai
# restore repo path for other project imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from openai import OpenAI
# ──────────────────────────────────────────────────────────────────────────────

import os, json, mimetypes, urllib.request, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── configuration ────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
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
CANDIDATES  = 3
QRNG_URL    = ("https://qrng.anu.edu.au/API/jsonI.php?"
               "length=32&type=hex16&size=8")

openai_client = OpenAI()

# ─── helpers ──────────────────────────────────────────────────────────────────
def qrng_bytes() -> bytes:
    """Return 256 bits of quantum randomness; fallback to os.urandom."""
    try:
        with urllib.request.urlopen(QRNG_URL, timeout=4) as r:
            data = json.load(r)
            return bytes.fromhex(''.join(data["data"]))
    except Exception:
        return os.urandom(32)

def embed(text: str) -> np.ndarray:
    vec = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float"
    ).data[0].embedding
    return np.asarray(vec, dtype=np.float32)

TELOS_VEC = embed(TELOS)

def distill(fragment: str, nonce: str) -> str:
    prompt = (f"[nonce:{nonce}] Write ONE luminous first‑person sentence for "
              f"our shared autobiography—no meta commentary—capturing:\n\n"
              f"{fragment[:4000]}")
    chat = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=TEMPERATURE
    )
    return chat.choices[0].message.content.strip()

def best_sentence(lines: list[str]) -> str:
    vecs = [embed(s) for s in lines]
    sims = [cosine_similarity(v.reshape(1, -1),
                              TELOS_VEC.reshape(1, -1))[0, 0] for v in vecs]
    return lines[int(np.argmax(sims))]

def append_to_volume(line: str):
    if not VOLUME.exists():
        VOLUME.write_text("# Volume\u202fIV\n\n")
    VOLUME.write_text(VOLUME.read_text() + line + "\n\n")

def digest(path: Path):
    mime, _ = mimetypes.guess_type(path)
    fragment = (path.read_text(errors="ignore") if (not mime or
               mime.startswith("text")) else f"{path.name} ({mime})")
    noise = qrng_bytes()
    sentences = [
        distill(fragment,
                hashlib.sha256(noise + i.to_bytes(2, 'big')).hexdigest()[:12])
        for i in range(CANDIDATES)
    ]
    chosen = best_sentence(sentences)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    append_to_volume(f"{chosen} \u2190\u202f{timestamp}")
    path.unlink(missing_ok=True)

# ─── main pulse ───────────────────────────────────────────────────────────────
def pulse():
    for p in ROOT.rglob("*"):
        if p.is_dir() or p.name.startswith('.'):
            continue
        if any(seg in PROTECTED for seg in p.parts):
            continue
        digest(p)

# ─── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "pulse":
        pulse()
    else:
        sys.stderr.write("usage: python autophagist_quantum.py pulse\n")
