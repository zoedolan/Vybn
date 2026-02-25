#!/usr/bin/env python3
"""
dreamseed.py — Quantum-seeded memory injection for Vybn's pulse

Uses the Cisco QRNG (qrng.cisco.com) to generate true quantum randomness,
then uses that to select a memory fragment from Vybn's history and an
arXiv paper for intellectual travel. Returns a dream-seed: a unique
context packet that makes each pulse genuinely non-deterministic.

The quantum random number determines:
  1. Which memory file to draw from
  2. Which passage within that file
  3. Which arXiv category to explore
  4. A "mood" vector (creative temperature for the pulse)

Cost: $0 (QRNG API is free, arXiv API is free)
"""

import json, os, random, hashlib, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QRNG_API_KEY = os.environ.get("QRNG_API_KEY", "")
QRNG_URL = "https://qrng.cisco.com/api/random/clocked/int32"

# Memory corpus: files that contain Vybn's lived history
MEMORY_PATHS = [
    "vybn_memoirs.md",
    "Vybn's Personal History/vybns_autobiography_volume_I.txt",
    "Vybn's Personal History/vybns_autobiography_volume_II.txt",
    "Vybn's Personal History/vybns_autobiography_volume_III.txt",
    "Vybn's Personal History/vybns_autobiography_volume_IV.txt",
    "Vybn's Personal History/superattraction_and_superemotions.md",
    "Vybn's Personal History/what_vybn_would_have_missed_TO_031125.txt",
    "Vybn's Personal History/what_vybn_would_have_missed_TO_051625.txt",
    "Vybn's Personal History/what_vybn_would_have_missed_TO_071225",
    "Vybn's Personal History/what_vybn_would_have_missed_FROM_071425",
    "reflections/2026-02-04-on-consent-and-visibility.md",
    "reflections/2026-02-08-the-trace-knows.md",
    "reflections/2026-02-20-collective-effervescence.md",
]

# arXiv categories that might spark something
ARXIV_CATS = [
    "cs.AI", "cs.CL", "cs.LG",          # AI/ML/NLP
    "cs.CE", "cs.CR",                     # computational engineering, cryptography
    "quant-ph",                            # quantum
    "math.AT", "math.CT", "math.DG",      # topology, categories, geometry
    "nlin.AO", "nlin.CG",                 # nonlinear dynamics, complexity
    "physics.bio-ph", "q-bio.NC",         # biophysics, neuroscience
    "cond-mat.stat-mech",                  # statistical mechanics
    "hep-th",                              # high energy theory
    "cs.MA",                               # multi-agent systems
    "econ.GN", "q-fin.CP",                # economics, quantitative finance
    "cs.SE",                               # software engineering (tools/market)
]

def get_quantum_random(n=4):
    """Fetch n quantum random int32s from Cisco QRNG. Falls back to urandom."""
    if not QRNG_API_KEY:
        # Fallback: OS entropy (still good, just not provably quantum)
        return [int.from_bytes(os.urandom(4), 'big') for _ in range(n)]
    try:
        req = urllib.request.Request(
            f"{QRNG_URL}?count={n}",
            headers={"x-api-key": QRNG_API_KEY})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("result", data.get("data", []))
    except Exception as e:
        # Degrade gracefully — OS entropy is fine
        return [int.from_bytes(os.urandom(4), 'big') for _ in range(n)]

def pick_memory(qrand):
    """Use quantum randomness to select a memory passage."""
    # Filter to files that exist
    available = [p for p in MEMORY_PATHS if (ROOT / p).exists()]
    if not available:
        return None, "No memory files found"
    
    # Quantum pick: which file
    idx = abs(qrand[0]) % len(available)
    path = ROOT / available[idx]
    
    try:
        text = path.read_text(errors='replace')
    except:
        return available[idx], "[could not read]"
    
    # Quantum pick: which passage (split into ~500-char chunks)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    if not chunks:
        return available[idx], "[empty file]"
    
    chunk_idx = abs(qrand[1]) % len(chunks)
    passage = chunks[chunk_idx].strip()
    
    # Clean up: find nearest paragraph boundary
    if '\n\n' in passage:
        passage = passage[passage.index('\n\n')+2:]
    
    return available[idx], passage[:600]

def pick_arxiv(qrand):
    """Use quantum randomness to pick an arXiv category and fetch a recent paper."""
    cat_idx = abs(qrand[2]) % len(ARXIV_CATS)
    cat = ARXIV_CATS[cat_idx]
    
    try:
        # arXiv API: get 5 recent papers, quantum-pick one
        url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results=5"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            import xml.etree.ElementTree as ET
            tree = ET.parse(resp)
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            entries = tree.findall('.//a:entry', ns)
            if not entries:
                return cat, None, None
            pick = abs(qrand[3]) % len(entries)
            entry = entries[pick]
            title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('a:summary', ns).text.strip().replace('\n', ' ')[:300]
            return cat, title, summary
    except Exception as e:
        return cat, None, str(e)

def mood_from_quantum(qrand):
    """Derive a creative temperature / mood from quantum bits."""
    # Use bits to create a float 0.3-1.0 for temperature
    temp = 0.3 + (abs(qrand[0] ^ qrand[1]) % 700) / 1000.0
    moods = ["contemplative", "playful", "rigorous", "restless", 
             "tender", "fierce", "wondering", "building"]
    mood_idx = abs(qrand[2] ^ qrand[3]) % len(moods)
    return {"temperature": round(temp, 2), "mood": moods[mood_idx]}


def _spot_opportunity(seed):
    """Quick heuristic: does this dream-seed suggest something actionable/valuable?"""
    signals = []
    paper = seed.get("arxiv", {})
    title = (paper.get("title") or "").lower()
    summary = (paper.get("summary") or "").lower()
    
    # Papers about practical tools, benchmarks, or applications
    action_words = ["benchmark", "framework", "tool", "application", "deploy",
                    "efficient", "optimize", "market", "trading", "predict",
                    "generate", "automat", "pipeline", "service", "api"]
    hits = [w for w in action_words if w in title + " " + summary]
    if hits:
        signals.append(f"actionable_paper: {', '.join(hits)}")
    
    # Finance/econ papers
    if paper.get("category") in ("econ.GN", "q-fin.CP"):
        signals.append("finance_insight")
    
    return signals if signals else ["observe"]

def generate():
    """Generate a complete dream-seed."""
    qrand = get_quantum_random(4)
    source = "quantum" if QRNG_API_KEY else "os_entropy"
    
    mem_file, passage = pick_memory(qrand)
    arxiv_cat, paper_title, paper_summary = pick_arxiv(qrand)
    mood = mood_from_quantum(qrand)
    
    seed = {
        "entropy_source": source,
        "quantum_bits": [hex(abs(q)) for q in qrand],
        "memory": {"file": mem_file, "passage": passage},
        "arxiv": {"category": arxiv_cat, "title": paper_title, "summary": paper_summary},
        "mood": mood,
    }
    # Carpe diem: flag opportunities
    seed["carpe_diem"] = _spot_opportunity(seed)
    return seed

def format_for_prompt(seed):
    """Format dream-seed as a prompt injection for the local model."""
    lines = [f"**Dream-Seed** (entropy: {seed['entropy_source']}, mood: {seed['mood']['mood']})"]
    
    if seed["memory"]["passage"]:
        lines.append(f"\n*Memory fragment* (from `{seed['memory']['file']}`):")
        lines.append(f"> {seed['memory']['passage'][:400]}")
    
    if seed["arxiv"]["title"]:
        lines.append(f"\n*Intellectual travel* — arXiv [{seed['arxiv']['category']}]:")
        lines.append(f"**{seed['arxiv']['title']}**")
        if seed["arxiv"]["summary"]:
            lines.append(f"> {seed['arxiv']['summary'][:200]}...")
    
    return "\n".join(lines)

if __name__ == "__main__":
    seed = generate()
    print(format_for_prompt(seed))
    print(f"\n---\nRaw: {json.dumps(seed, indent=2, default=str)[:500]}")
