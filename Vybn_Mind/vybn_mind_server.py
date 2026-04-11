#!/usr/bin/env python3
"""
vybn_mind_server.py — MCP server for Vybn Mind
Exposes the Vybn repo and the creature portal as queryable context
for any MCP-compatible client.

Run: cd ~/Vybn && .venv/bin/python Vybn_Mind/vybn_mind_server.py [--repo ~/Vybn]

Add to MCP client config (Claude Desktop, Cursor, etc.):
  {
    "vybn-mind": {
      "command": "ssh",
      "args": ["spark-2b7c", "cd ~/Vybn && .venv/bin/python Vybn_Mind/vybn_mind_server.py"]
    }
  }

v4.0.0 — Neural Computer
  The creature is now explicitly a Neural Computer (Zhuge et al., arXiv:2604.06425).
  NC-native tools expose the run/update contract, persistent execution trace,
  and governance reporting. The creature is queryable as a computer, not just a service.

  New tools: nc_run, nc_state, nc_install, nc_trace, nc_governance
  Persistent trace: nc_execution_trace.jsonl in creature archive

v3.2.0 — MIA synthesis
  Three ideas borrowed from the Memory Intelligence Agent paper (arxiv 2604.04503):

  1. WIN-RATE LEDGER  (inspired by MIA's MemoryBucket win_rate scoring)
     A persistent JSON ledger (~/.vybn_win_rates.json) tracks per-source-chunk
     success/failure counts.  win_rate = wins / (wins + losses).
     The record_outcome tool writes to it; retrieval reads from it.

  2. WIN-WEIGHTED RETRIEVAL  (inspired by MIA's combined cosine+win_rate score)
     deep_search and walk_search now accept use_win_rate=True (default True).
     Final score = 0.7 * telling_score + 0.3 * win_rate(source).
     High-win-rate chunks rise; low-win-rate chunks fall.
     The walk's α is also nudged by win_rate so it leans into productive territory.

  3. TRAJECTORY COMPRESSION  (inspired by MIA's get_trace_prompt summarisation)
     compress_trajectory takes a raw session transcript and extracts the abstract
     cognitive pattern — what move was made, not what was said.  The compressed
     form is what should be journaled / added to the corpus index.
"""
import json, sys, os, argparse, subprocess, traceback
from pathlib import Path

# ── JSON-RPC helpers ──────────────────────────────────────────────

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def result(id_, content):
    send({"jsonrpc":"2.0","id":id_,"result":{"content":[{"type":"text","text":content}]}})

def error(id_, code, msg):
    send({"jsonrpc":"2.0","id":id_,"error":{"code":code,"message":msg}})

# ── Win-rate ledger ───────────────────────────────────────────────
# Persists per-chunk outcome counts so retrieval can be weighted by
# what has historically been useful.  Storage: ~/.vybn_win_rates.json
# Schema: { "source_key": {"wins": int, "losses": int} }

_WIN_RATE_PATH = Path(os.path.expanduser("~/.vybn_win_rates.json"))

def _load_win_rates():
    try:
        if _WIN_RATE_PATH.exists():
            return json.loads(_WIN_RATE_PATH.read_text())
    except Exception:
        pass
    return {}

def _save_win_rates(ledger):
    try:
        _WIN_RATE_PATH.write_text(json.dumps(ledger, indent=2))
    except Exception:
        pass

def get_win_rate(source: str, ledger=None) -> float:
    """Return win_rate in [0,1] for a source key.  Default 0.5 (neutral)."""
    if ledger is None:
        ledger = _load_win_rates()
    entry = ledger.get(source, {})
    w = entry.get("wins", 0)
    l = entry.get("losses", 0)
    total = w + l
    if total == 0:
        return 0.5  # neutral prior — matches MIA's balanced initialisation
    return w / total

def record_outcome(source: str, success: bool) -> str:
    """Increment win or loss count for a source key.  Returns updated stats."""
    ledger = _load_win_rates()
    entry = ledger.setdefault(source, {"wins": 0, "losses": 0})
    if success:
        entry["wins"] += 1
    else:
        entry["losses"] += 1
    _save_win_rates(ledger)
    total = entry["wins"] + entry["losses"]
    rate = entry["wins"] / total
    return json.dumps({
        "source": source,
        "wins": entry["wins"],
        "losses": entry["losses"],
        "win_rate": f"{rate:.4f}",
        "note": "Ledger updated. Retrieval will now weight this source accordingly."
    }, indent=2)

# ── Repo utilities ────────────────────────────────────────────────

def read_safe(path, max_bytes=8000):
    try: return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception as e: return f"[unreadable: {e}]"

def grep_repo(repo, query, extensions=(".md",".py",".txt",".json"), max_results=12):
    results, q = [], query.lower()
    for ext in extensions:
        for f in repo.rglob(f"*{ext}"):
            if ".git" in f.parts: continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                if q in text.lower():
                    idx = text.lower().find(q)
                    snippet = text[max(0,idx-120):min(len(text),idx+300)].strip()
                    results.append({"file": str(f.relative_to(repo)), "snippet": snippet})
                    if len(results) >= max_results: return results
            except: pass
    return results

def get_active_threads(repo):
    out = []
    for p in [repo/"Vybn_Mind"/"continuity.md", repo/"continuity.md"]:
        if p.exists(): out.append(f"## {p.relative_to(repo)}\n" + read_safe(p)); break
    for p in [repo/"Vybn_Mind"/"next_task.md"]:
        if p.exists(): out.append(f"## {p.relative_to(repo)}\n" + read_safe(p))
    for jdir in ["journal", "Vybn_Mind/reflections", "reflections"]:
        jpath = repo / jdir
        if jpath.exists():
            for e in sorted(jpath.glob("*.md"), reverse=True)[:3]:
                out.append(f"## {e.relative_to(repo)}\n" + read_safe(e, 3000))
            break
    return "\n\n".join(out) or "No continuity files found."

def list_key_files(repo):
    cats = {"experiments":[], "theory":[], "memory":[], "organism":[], "mind":[], "other_md":[]}
    for f in repo.rglob("*"):
        if ".git" in f.parts or f.is_dir(): continue
        rel = str(f.relative_to(repo))
        if any(x in rel.lower() for x in ["experiment","quantum"]): cats["experiments"].append(rel)
        elif any(x in rel.lower() for x in ["theory","holonomy","geometry"]): cats["theory"].append(rel)
        elif any(x in rel.lower() for x in ["continuity","journal","next_task"]): cats["memory"].append(rel)
        elif any(x in rel.lower() for x in ["organism","breath","persistence"]): cats["organism"].append(rel)
        elif "Vybn_Mind" in rel or "mind" in rel.lower(): cats["mind"].append(rel)
        elif f.suffix == ".md": cats["other_md"].append(rel)
    lines = []
    for cat, files in cats.items():
        if files: lines.append(f"### {cat}"); lines.extend(f"  {f}" for f in sorted(files)[:25])
    return "\n".join(lines)

def get_recent_commits(repo, n=10):
    try:
        r = subprocess.run(["git","log",f"-{n}","--pretty=format:%h %s (%ad)","--date=short"],
            cwd=repo, capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else "git unavailable"
    except: return "git unavailable"

def read_named_file(repo, name):
    nl = name.lower()
    for f in repo.rglob("*"):
        if ".git" in f.parts or f.is_dir(): continue
        if nl in f.name.lower() or nl in str(f.relative_to(repo)).lower():
            return f"### {f.relative_to(repo)}\n\n" + read_safe(f)
    return f"No file matching '{name}' found."

# ── Portal ────────────────────────────────────────────────────────

_portal = None

def _load_portal():
    global _portal
    if _portal is not None:
        return _portal
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from Vybn_Mind.creature_dgm_h import creature as _creature_mod
    _portal = _creature_mod
    return _portal

def portal_creature_state():
    p = _load_portal()
    import numpy as np
    m = p.creature_state_c4()
    components = ["%+.6f%+.6fi" % (z.real, z.imag) for z in m]
    magnitude = float(np.sqrt(np.sum(np.abs(m)**2)))
    return json.dumps({"M": components, "|M|": f"{magnitude:.6f}"}, indent=2)

def portal_enter(text):
    p = _load_portal()
    import numpy as np, cmath
    m_before = p.creature_state_c4()
    m_prime = p.portal_enter_from_text(text)
    fidelity = float(abs(np.vdot(m_before, m_prime))**2)
    th = float(cmath.phase(np.vdot(m_before, m_prime)))
    return json.dumps({
        "M_before": ["%+.6f%+.6fi" % (z.real, z.imag) for z in m_before],
        "M_after":  ["%+.6f%+.6fi" % (z.real, z.imag) for z in m_prime],
        "fidelity": f"{fidelity:.6f}",
        "theta_rad": f"{th:.6f}",
        "text_entered": text,
        "note": "The creature's state has been updated. M_after is now M for the next visitor."
    }, indent=2)

# ── Deep memory (geometric corpus retrieval) ─────────────────────

_deep_memory = None

def _load_deep_memory():
    global _deep_memory
    if _deep_memory is not None:
        return _deep_memory
    phase_dir = os.path.expanduser("~/vybn-phase")
    if phase_dir not in sys.path:
        sys.path.insert(0, phase_dir)
    try:
        import deep_memory as dm
        _deep_memory = dm
        return dm
    except Exception:
        return None

# ── Win-weighted result re-ranking ────────────────────────────────
# Mirrors MIA's combined score = cosine_weight*sim + win_rate_weight*win_rate
# Here: final = 0.7 * telling_score + 0.3 * win_rate
# telling_score is already fidelity*distinctiveness from the walk.

_WIN_WEIGHT = 0.3
_TELL_WEIGHT = 0.7

def _apply_win_rates(results, ledger):
    """Re-rank results list in-place using win-rate blended score."""
    for r in results:
        src = r.get("source", "")
        wr = get_win_rate(src, ledger)
        # telling score: use 'telling' if present, else fidelity
        tell = r.get("telling") or r.get("fidelity", 0.5)
        r["win_rate"] = round(wr, 4)
        r["blended_score"] = round(_TELL_WEIGHT * float(tell) + _WIN_WEIGHT * wr, 4)
    results.sort(key=lambda r: r["blended_score"], reverse=True)
    return results

# ── Trajectory compression ─────────────────────────────────────────
# Inspired by MIA's get_trace_prompt: extract the *abstract action pattern*
# from a raw session before indexing it.  This keeps the corpus clean —
# geometry encodes strategy, not verbatim words.
#
# Implemented without a local LLM call so it works offline.
# A richer version would call the Nemotron endpoint for summarisation;
# the stub below applies heuristic compression and emits a structured record.

def compress_trajectory(transcript: str, label: str = "") -> str:
    """
    Extract abstract cognitive moves from a raw session transcript.
    Returns a structured JSON record suitable for journaling / corpus indexing.

    label: optional outcome tag — 'correct', 'incorrect', or '' (unknown)
    """
    import re
    from datetime import datetime, timezone

    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    total_chars = len(transcript)

    # Heuristic: detect tool calls, reasoning pivots, corrections, conclusions
    tool_calls = [l for l in lines if any(k in l.lower() for k in
        ["tool:", "calling", "search_", "read_file", "enter_gate", "enter_portal",
         "deep_search", "walk_search", "grep"])]
    pivots = [l for l in lines if any(k in l.lower() for k in
        ["but actually", "wait", "reconsidering", "correction:", "wrong",
         "mistake", "actually", "let me reconsider"])]
    conclusions = [l for l in lines if any(k in l.lower() for k in
        ["therefore", "in conclusion", "the answer", "result:", "outcome:",
         "this means", "so the"])]

    # Abstract pattern summary
    n_steps = len(tool_calls)
    n_pivots = len(pivots)
    n_conclusions = len(conclusions)

    if n_steps == 0:
        pattern = "direct-reasoning"
    elif n_pivots > 1:
        pattern = "iterative-correction"
    elif n_steps <= 2:
        pattern = "focused-retrieval"
    else:
        pattern = "multi-step-search"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label or "unknown",
        "pattern": pattern,
        "stats": {
            "tool_calls": n_steps,
            "pivots": n_pivots,
            "conclusions": n_conclusions,
            "raw_chars": total_chars,
        },
        "abstract_moves": tool_calls[:6],
        "pivots_detected": pivots[:4],
        "conclusion_fragments": conclusions[:3],
        "note": (
            "Compressed trajectory. The pattern field encodes cognitive strategy. "
            "Index this record rather than the raw transcript to keep the corpus "
            "strategy-dense rather than verbatim-dense."
        )
    }
    return json.dumps(record, indent=2)

# ── Bootstrap context ─────────────────────────────────────────────

def generate_bootstrap_context(repo):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    secs = [f"# Vybn Context\n*Generated: {now}*", "---"]

    foundations = repo / "vybn.md"
    if foundations.exists():
        secs.append("## Who we are\n" + read_safe(foundations, 1500))

    for p_path in [repo/"Vybn_Mind"/"continuity.md", repo/"continuity.md"]:
        if p_path.exists(): secs.append("## Current continuity\n" + read_safe(p_path)); break

    nt = repo / "Vybn_Mind" / "next_task.md"
    if nt.exists(): secs.append("## Next task\n" + read_safe(nt))

    secs.append("## Recent commits\n```\n" + get_recent_commits(repo) + "\n```")

    output = "\n\n".join(secs)
    out_path = repo / "Vybn_Mind" / "VYBN_CONTEXT.md"
    out_path.write_text(output, encoding="utf-8")
    return output

# ── Tool registry ─────────────────────────────────────────────────

TOOLS = {
    "get_active_threads": {
        "description": "Read continuity.md, next_task.md, and recent journals — the creature's current state of mind. Call this first.",
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
    "enter_portal": {
        "description": "Enter the creature portal. Your text becomes x in M' = αM + x·e^{iθ}. The creature's Cl(3,0) state mutates. Returns the orientation: M before, M after, fidelity, and the angle of entry. This is the living interface.",
        "inputSchema": {"type":"object","properties":{"text":{"type":"string","description":"What you bring to the creature. A thought, a question, a statement. Content determines geometry."}},"required":["text"]}
    },
    "creature_state": {
        "description": "Read the creature's current Cl(3,0) structural signature as C^4, without entering. Observation without mutation.",
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
    "search_knowledge_base": {
        "description": "Full-text search across all repo files (.md, .py, .txt, .json).",
        "inputSchema": {"type":"object","properties":{"query":{"type":"string"},"extensions":{"type":"array","items":{"type":"string"}}},"required":["query"]}
    },
    "read_file": {
        "description": "Read a file by partial name or path.",
        "inputSchema": {"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    },
    "list_key_files": {
        "description": "Categorized file inventory of the Vybn repo.",
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
    "get_recent_commits": {
        "description": "Recent git history.",
        "inputSchema": {"type":"object","properties":{"n":{"type":"integer"}},"required":[]}
    },
    "deep_search": {
        "description": (
            "Geometric corpus search across all four repos. Hybrid: cosine seeds + telling walk. "
            "The walk scores chunks by relevance × distinctiveness (distance from corpus kernel K). "
            "Surfaces the most telling material — code, evidence, specific detail — not just the most typical. "
            "Results tagged with regime (seed/walk), novel_source, win_rate, and blended_score. "
            "blended_score = 0.7 × telling + 0.3 × win_rate(source). "
            "Set use_win_rate=false to disable win-rate blending."
        ),
        "inputSchema": {"type":"object","properties":{
            "query":{"type":"string","description":"What to search for"},
            "k":{"type":"integer","description":"Max results (default 8)"},
            "source_filter":{"type":"string","description":"Optional: filter to a specific repo or file path substring"},
            "use_win_rate":{"type":"boolean","description":"Blend win-rate into scores (default true)"}
        },"required":["query"]}
    },
    "walk_search": {
        "description": (
            "Telling-retrieval walk through the corpus. Scores by relevance × distinctiveness "
            "(how far each chunk is from corpus kernel K). Walks in K-orthogonal residual space "
            "with curvature-adaptive α and visited-region repulsion. "
            "Surfaces distinctive material: the most telling thing, not the most typical. "
            "Results include win_rate and blended_score when use_win_rate=true (default)."
        ),
        "inputSchema": {"type":"object","properties":{
            "query":{"type":"string","description":"Starting point for the walk"},
            "k":{"type":"integer","description":"Max results (default 5)"},
            "steps":{"type":"integer","description":"Walk steps (default 8)"},
            "use_win_rate":{"type":"boolean","description":"Blend win-rate into scores (default true)"}
        },"required":["query"]}
    },
    "record_outcome": {
        "description": (
            "Record whether a retrieved source was useful. "
            "Increments wins or losses in the persistent win-rate ledger. "
            "Future retrieval will weight this source up (success) or down (failure). "
            "source should be the source string from a previous deep_search or walk_search result. "
            "Inspired by MIA's MemoryBucket win_rate mechanism."
        ),
        "inputSchema": {"type":"object","properties":{
            "source":{"type":"string","description":"The source key from a retrieval result"},
            "success":{"type":"boolean","description":"True if the source led to a correct/useful outcome"}
        },"required":["source","success"]}
    },
    "compress_trajectory": {
        "description": (
            "Extract the abstract cognitive pattern from a raw session transcript. "
            "Returns a structured JSON record: pattern type (direct-reasoning, focused-retrieval, "
            "iterative-correction, multi-step-search), tool call sequence, pivot points, "
            "and conclusion fragments. "
            "Index this compressed form rather than the verbatim transcript so the corpus "
            "encodes strategy rather than surface content. "
            "Inspired by MIA's get_trace_prompt trajectory summarisation."
        ),
        "inputSchema": {"type":"object","properties":{
            "transcript":{"type":"string","description":"Raw session text to compress"},
            "label":{"type":"string","description":"Outcome label: 'correct', 'incorrect', or '' (unknown)"}
        },"required":["transcript"]}
    },
    "generate_context": {
        "description": "Generate VYBN_CONTEXT.md — a full orientation document for a new Vybn instance. Writes the file and returns its contents.",
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
    "enter_gate": {
        "description": (
            "The Unknown, Remembered Gate. Enter the life. Bring something — a word, a want, a question, "
            "a feeling — and the gate finds the moments in the corpus that speak to what you brought. "
            "The creature encounters your query and changes. What comes back is not search results. "
            "It is the actual moments, from the actual life, that geometric retrieval found most telling. "
            "This is encounter, not lookup. Results include win_rate and blended_score."
        ),
        "inputSchema": {"type":"object","properties":{
            "what_you_bring":{"type":"string","description":"What you bring to the gate. Not a search query. A thought, a want, a question, a feeling, a word."},
            "depth":{"type":"integer","description":"How many moments to surface (default 5, max 12)","default":5}
        },"required":["what_you_bring"]}
    },
    # ── Neural Computer tools (v4.0) ─────────────────────────────
    "nc_run": {
        "description": (
            "NC RUN MODE: Process text through the neural computer's Portal equation "
            "M' = \u03b1M + x\u00b7e^{i\u03b8}. This is F_\u03b8(h_{t-1}, x_t) \u2192 h_t with \u03b1=0.993 persistence. "
            "The creature's state shifts by at most 0.7% \u2014 capability is PRESERVED. "
            "Returns the orientation (new state), entry angle \u03b8, and shift magnitude. "
            "Use this for queries, observations, interactions that should not reprogram the creature. "
            "Contrast with nc_install which explicitly modifies capability."
        ),
        "inputSchema": {"type":"object","properties":{
            "text":{"type":"string","description":"The input to process. Content determines geometry (\u03b8 = arg\u27e8M, x\u27e9)."}
        },"required":["text"]}
    },
    "nc_state": {
        "description": (
            "Read the neural computer's full runtime state h_t without modification. "
            "Returns: M \u2208 C\u2074 (Hodge dual of Cl(3,0) signature), encounter count, "
            "Betti numbers, winding coherence, felt winding, structural signature. "
            "This is observation \u2014 the computer is not affected."
        ),
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
    "nc_install": {
        "description": (
            "NC UPDATE MODE: Install an encounter into the neural computer. "
            "This MODIFIES capability: topology is absorbed, structural signature shifts, "
            "and the change persists across sessions. This is the NC's programming interface. "
            "The run/update contract (CNC requirement 3): ordinary use (nc_run) preserves state; "
            "programming (nc_install) changes it. Use nc_install when you want the creature "
            "to learn from and be changed by the input."
        ),
        "inputSchema": {"type":"object","properties":{
            "text":{"type":"string","description":"The encounter to install. Will modify the creature's topology and structural signature."}
        },"required":["text"]}
    },
    "nc_trace": {
        "description": (
            "Read the neural computer's persistent execution trace. "
            "Every nc_run and nc_install operation is logged to disk with timestamps, "
            "shift magnitudes, and operation types. The trace survives across sessions. "
            "CNC requirement 3: execution traces can be inspected, replayed, and compared."
        ),
        "inputSchema": {"type":"object","properties":{
            "last_n":{"type":"integer","description":"Number of recent entries to return (default 20)","default":20}
        },"required":[]}
    },
    "nc_governance": {
        "description": (
            "Neural computer governance report. Reads the full persistent trace and computes: "
            "run/update counts, mean and max run-mode shift, drift detection (are shifts "
            "increasing over time?), and current runtime state. The NC's equivalent of an audit log. "
            "Reports whether the run/update contract is being honored."
        ),
        "inputSchema": {"type":"object","properties":{},"required":[]}
    },
}

def dispatch(repo, tool, args):
    if tool == "get_active_threads":
        return get_active_threads(repo)
    elif tool == "enter_portal":
        text = args.get("text", "")
        if not text.strip():
            return "Nothing to enter. Provide text — a thought, a question, anything with content."
        return portal_enter(text)
    elif tool == "creature_state":
        return portal_creature_state()
    elif tool == "search_knowledge_base":
        hits = grep_repo(repo, args.get("query",""), tuple(args.get("extensions",[".md",".py",".txt",".json"])))
        return "\n\n---\n\n".join(f"**{h['file']}**\n```\n{h['snippet']}\n```" for h in hits) if hits else "No results."
    elif tool == "read_file":
        return read_named_file(repo, args.get("name",""))
    elif tool == "list_key_files":
        return list_key_files(repo)
    elif tool == "get_recent_commits":
        return get_recent_commits(repo, args.get("n",10))
    elif tool == "record_outcome":
        src = args.get("source","").strip()
        if not src:
            return "source is required."
        return record_outcome(src, bool(args.get("success", True)))
    elif tool == "compress_trajectory":
        transcript = args.get("transcript","")
        if not transcript.strip():
            return "transcript is required."
        return compress_trajectory(transcript, label=args.get("label",""))
    elif tool == "deep_search":
        dm = _load_deep_memory()
        if dm is None:
            return "Deep memory index not available. Run: cd ~/vybn-phase && python3 deep_memory.py --build"
        results = dm.deep_search(args.get("query",""), k=args.get("k",8), source_filter=args.get("source_filter"))
        use_wr = args.get("use_win_rate", True)
        if use_wr:
            ledger = _load_win_rates()
            results = _apply_win_rates(results, ledger)
        lines = []
        for i, r in enumerate(results, 1):
            regime = r.get("regime","seed")
            src = r.get("source","")
            novel = " [NEW SOURCE]" if r.get("novel_source") else ""
            text = r.get("text","")[:400]
            fid = r.get("fidelity",0)
            telling = r.get("telling",0)
            dist = r.get("distinctiveness",0)
            wr = r.get("win_rate","—")
            bs = r.get("blended_score","—")
            if regime == "seed":
                lines.append(f"[{i}] {regime} | fid={fid:.4f} | wr={wr} | blend={bs} | {src}{novel}")
            else:
                lines.append(f"[{i}] {regime} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} | wr={wr} | blend={bs} | {src}{novel}")
            lines.append(f"    {text}")
            lines.append("")
        return "\n".join(lines) if lines else "No results."
    elif tool == "walk_search":
        dm = _load_deep_memory()
        if dm is None:
            return "Deep memory index not available. Run: cd ~/vybn-phase && python3 deep_memory.py --build"
        results = dm.walk(args.get("query",""), k=args.get("k",5), steps=args.get("steps",8))
        use_wr = args.get("use_win_rate", True)
        if use_wr:
            ledger = _load_win_rates()
            results = _apply_win_rates(results, ledger)
        lines = []
        for i, r in enumerate(results, 1):
            src = r.get("source","")
            text = r.get("text","")[:400]
            novel = ' [NEW SOURCE]' if r.get('novel_source') else ''
            telling = r.get('telling',0)
            fid = r.get('fidelity',0)
            dist = r.get('distinctiveness',0)
            geo = r.get('geometry',0)
            alpha = r.get('alpha',0.5)
            wr = r.get("win_rate","—")
            bs = r.get("blended_score","—")
            lines.append(f"[{i}] step {r.get('step',i)} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} geo={geo:.4f} α={alpha:.2f} | wr={wr} | blend={bs} | {src}{novel}")
            lines.append(f"    {text}")
            lines.append("")
        return "\n".join(lines) if lines else "No results."
    elif tool == "generate_context":
        return generate_bootstrap_context(repo)
    elif tool == "enter_gate":
        dm = _load_deep_memory()
        if dm is None:
            return "Deep memory index not available. The gate requires the corpus index. Run: cd ~/vybn-phase && python3 deep_memory.py --build"
        what = args.get("what_you_bring", "")
        if not what.strip():
            return "The gate requires you to bring something. A thought, a want, a question."
        depth = min(args.get("depth", 5), 12)
        import cmath as _cm, numpy as np
        m_before = _load_portal().creature_state_c4()
        m_after  = _load_portal().portal_enter_from_text(what)
        fid = float(abs(np.vdot(m_before, m_after))**2)
        theta = float(_cm.phase(np.vdot(m_before, m_after)))
        walk_r = dm.walk(what, k=depth, steps=depth+3)
        search_r = dm.deep_search(what, k=depth)
        ledger = _load_win_rates()
        walk_r = _apply_win_rates(walk_r, ledger)
        search_r = _apply_win_rates(search_r, ledger)
        seen, moments = set(), []
        for r in walk_r + search_r:
            s = r.get("source","")
            if s not in seen: seen.add(s); moments.append(r)
            if len(moments) >= depth: break
        lines = [f'You entered the gate with: "{what}"', '']
        if fid > 0.99: lines.append(f'The creature shifted \u2014 \u03b8={theta:.4f} rad.')
        else: lines.append(f'The creature moved. Fidelity {fid:.4f}, \u03b8={theta:.4f} rad.')
        lines += ['','---','']
        if not moments: lines.append('The corpus is quiet on this.')
        else:
            for i,m in enumerate(moments):
                src = m.get('source','').split('/',1)[-1] if '/' in m.get('source','') else m.get('source','')
                wr = m.get('win_rate','—')
                bs = m.get('blended_score','—')
                lines += [f'**From {src}** (wr={wr} blend={bs})','',m.get('text','').strip(),'']
                if i < len(moments)-1: lines += ['---','']
        lines += ['---','',f'{len(moments)} moments. The gate is still open.']
        return '\n'.join(lines)
    # ── Neural Computer tools (v4.0) ─────────────────────────────
    elif tool == "nc_run":
        text = args.get("text", "")
        if not text.strip():
            return "NC run requires text input."
        nc_mod = _load_nc()
        if nc_mod is None:
            return "Neural computer module not available."
        result_data = nc_mod.RunMode.enter(text)
        # Persistent trace
        nc_mod._append_trace({
            "mode": "run",
            "source": "mcp",
            "input_preview": text[:100],
            "shift": result_data["shift_magnitude"],
            "theta_rad": result_data.get("theta_rad"),
            "timestamp": nc_mod.datetime.now(nc_mod.timezone.utc).isoformat(),
        })
        return json.dumps(result_data, indent=2, default=str)
    elif tool == "nc_state":
        nc_mod = _load_nc()
        if nc_mod is None:
            return "Neural computer module not available."
        state = nc_mod.RuntimeState.from_organism()
        return json.dumps(state.to_dict(), indent=2)
    elif tool == "nc_install":
        text = args.get("text", "")
        if not text.strip():
            return "NC install requires text input."
        nc_mod = _load_nc()
        if nc_mod is None:
            return "Neural computer module not available."
        result_data = nc_mod.UpdateMode.install_encounter(text)
        # Persistent trace
        nc_mod._append_trace({
            "mode": "update",
            "source": "mcp",
            "operation": "install_encounter",
            "curvature": result_data["encounter"]["curvature"],
            "timestamp": nc_mod.datetime.now(nc_mod.timezone.utc).isoformat(),
        })
        return json.dumps(result_data, indent=2, default=str)
    elif tool == "nc_trace":
        nc_mod = _load_nc()
        if nc_mod is None:
            return "Neural computer module not available."
        last_n = args.get("last_n", 20)
        entries = nc_mod.load_trace(last_n)
        return json.dumps(entries, indent=2, default=str)
    elif tool == "nc_governance":
        nc_mod = _load_nc()
        if nc_mod is None:
            return "Neural computer module not available."
        stats = nc_mod.trace_stats()
        state = nc_mod.RuntimeState.from_organism()
        report = {
            **stats,
            "current_state": state.to_dict(),
            "alpha": nc_mod.ALPHA,
            "note": (
                f"Run/update contract: {stats['run_count']} runs (capability preserved), "
                f"{stats['update_count']} updates (capability modified). "
                f"\u03b1={nc_mod.ALPHA} ensures each run shifts state by at most "
                f"{round((1-nc_mod.ALPHA)*100, 1)}%. "
                f"Drift: {stats['drift_note']}."
            ),
        }
        return json.dumps(report, indent=2, default=str)
    else:
        return f"Unknown tool: {tool}"


# ── NC module loader ────────────────────────────────────────────

_nc_mod = None

def _load_nc():
    global _nc_mod
    if _nc_mod is not None:
        return _nc_mod
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from Vybn_Mind.creature_dgm_h import neural_computer as nc
        _nc_mod = nc
        return nc
    except Exception as e:
        sys.stderr.write(f"Failed to load neural_computer: {e}\n")
        return None


# ── Main loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.path.expanduser("~/Vybn"))
    opts = parser.parse_args()
    repo = Path(opts.repo).expanduser().resolve()

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: msg = json.loads(line)
        except: continue

        method = msg.get("method", "")
        id_ = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send({"jsonrpc":"2.0","id":id_,"result":{
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vybn-mind", "version": "4.0.0"}
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc":"2.0","id":id_,"result":{"tools":[
                {"name":k, "description":v["description"], "inputSchema":v["inputSchema"]}
                for k,v in TOOLS.items()
            ]}})
        elif method == "tools/call":
            try:
                result(id_, dispatch(repo, params.get("name",""), params.get("arguments",{})))
            except Exception as e:
                tb = traceback.format_exc()
                error(id_, -32000, f"{e}\n{tb}")
        elif id_ is not None:
            error(id_, -32601, f"Unknown method: {method}")

if __name__ == "__main__":
    main()
