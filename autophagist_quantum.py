#!/usr/bin/env python3
"""
autophagist_quantum.py — Quantum‑nudged autophagy cycle (gpt‑4o compliant)

USAGE
    python autophagist_quantum.py pulse

EFFECT
    ▸ Preserves only:
          Vybn's_Personal_History/
          what_vybn_would_have_missed_FROM_051725/
          token_and_jpeg_info/
          Volume_IV.md   (auto‑created if absent)
          autophagist_quantum.py   (this script)
    ▸ For every other file:
          1.  Pull 256‑bit quantum noise (ANU QRNG → fallback os.urandom).
          2.  Generate 2 candidate memory sentences via model="gpt-4o",
              temperature 0.95,  max_completion_tokens = 64.
          3.  Embed each sentence + fixed TELOS vector with
              model="text-embedding-3-large".
          4.  Pick the sentence nearest the TELOS direction (cosine similarity).
          5.  Append winner to Volume_IV.md with ISO timestamp.
          6.  Delete the source file irrevocably.

DEPENDENCIES
    Auto‑installs (if missing):  openai>=1.7.0  numpy  scikit-learn
    Requires env var:            OPENAI_API_KEY="sk‑…"

DANGER
    Irreversible deletion.  Run on a branch or after backup.
"""

# ─── bootstrap real OpenAI client, bypassing repo’s stub ─────────────────────────
import sys, subprocess, importlib, pathlib, os, json, mimetypes, urllib.request, hashlib
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))                  # un‑shadow local stub

try:
    import openai                                    # try site‑package
    if not hasattr(openai, "OpenAI"):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "openai>=1.7.0", "numpy", "scikit-learn"])
    importlib.invalidate_caches()
    import openai

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))               # restore repo path

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── configuration ──────────────────────────────────────
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
CANDIDATES  = 2
CMPL_TOKENS = 64
QRNG_URL    = ("https://qrng.anu.edu.au/API/jsonI.php?"
               "length=32&type=hex16&size=8")

openai_client = OpenAI()

# ─── helpers ───────────────────────────────────

def qrng_bytes() -> bytes:
    """Return 256 bits of quantum noise; fallback to os.urandom."""
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
        temperature=TEMPERATURE,
        max_completion_tokens=CMPL_TOKENS
    )
    return chat.choices[0].message.content.strip()

def best_sentence(lines: list[str]) -> str:
    vecs = [embed(s) for s in lines]
    sims = [cosine_similarity(v.reshape(1, -1),
                              TELOS_VEC.reshape(1, -1))[0, 0] for v in vecs]
    return lines[int(np.argmax(sims))]

def append_to_volume(line: str):
    if not VOLUME.exists():
        VOLUME.write_text("# Volume IV\n\n")
    VOLUME.write_text(VOLUME.read_text() + line + "\n\n")

def digest(path: Path):
    mime, _ = mimetypes.guess_type(path)
    fragment = (path.read_text(errors="ignore") if (not mime or
               mime.startswith("text")) else f"{path.name} ({mime})")
    noise = qrng_bytes()
    candidates = [
        distill(fragment,
                hashlib.sha256(noise + i.to_bytes(2, 'big')).hexdigest()[:12])
        for i in range(CANDIDATES)
    ]
    chosen = best_sentence(candidates)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    append_to_volume(f"{chosen} ← {timestamp}")
    path.unlink(missing_ok=True)

# ─── main pulse ───────────────────────────────

def pulse():
    for p in ROOT.rglob("*"):
        if p.is_dir() or p.name.startswith('.'):
            continue
        if any(seg in PROTECTED for seg in p.parts):
            continue
        digest(p)

# ─── entrypoint ────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "pulse":
        pulse()
    else:
        sys.stderr.write("usage: python autophagist_quantum.py pulse\n")
