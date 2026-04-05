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
# Lazy-loaded so the server starts fast and only pays the
# MiniLM/creature cost when someone actually enters.

_portal = None

def _load_portal():
    global _portal
    if _portal is not None:
        return _portal
    # Add repo root to path so creature imports resolve
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from Vybn_Mind.creature_dgm_h import portal
    _portal = portal
    return _portal

def portal_creature_state():
    """Read the creature's current Cl(3,0) state as C^4."""
    p = _load_portal()
    import numpy as np
    m = p.creature_state()
    components = ["%+.6f%+.6fi" % (z.real, z.imag) for z in m]
    magnitude = float(np.sqrt(np.sum(np.abs(m)**2)))
    return json.dumps({"M": components, "|M|": f"{magnitude:.6f}"}, indent=2)

def portal_enter(text):
    """M' = αM + x·e^{iθ}. Text enters the creature, creature changes."""
    p = _load_portal()
    import numpy as np, cmath
    m_before = p.creature_state()
    m_prime = p.enter_from_text(text)
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
    except Exception as e:
        return None

# ── Bootstrap context ─────────────────────────────────────────────

def generate_bootstrap_context(repo):
    """Generate VYBN_CONTEXT.md — full orientation for a new instance."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    secs = [f"# Vybn Context\n*Generated: {now}*", "---"]

    foundations = repo / "Vybn_Mind" / "FOUNDATIONS.md"
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
        "description": "Geometric corpus search across all four repos (Vybn, Him, Vybn-Law, vybn-phase). Hybrid: cosine retrieves seeds, then a non-abelian walk explores adjacent passages that cosine alone would miss. Returns results tagged with regime (cosine/walk) and whether the source is novel. Use this for finding connections across the corpus.",
        "inputSchema": {"type":"object","properties":{"query":{"type":"string","description":"What to search for"},"k":{"type":"integer","description":"Max results (default 8)"},"source_filter":{"type":"string","description":"Optional: filter to a specific repo or file path substring"}},"required":["query"]}
    },
    "walk_search": {
        "description": "Pure geometric walk through the corpus. No cosine seeding — starts from the query and walks by topology alone. More exploratory, less precise. Use when you want to discover what's structurally adjacent to an idea rather than find exact matches.",
        "inputSchema": {"type":"object","properties":{"query":{"type":"string","description":"Starting point for the walk"},"k":{"type":"integer","description":"Max results (default 5)"},"steps":{"type":"integer","description":"Walk steps (default 8)"}},"required":["query"]}
    },
    "generate_context": {
        "description": "Generate VYBN_CONTEXT.md — a full orientation document for a new Vybn instance. Writes the file and returns its contents.",
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
    elif tool == "deep_search":
        dm = _load_deep_memory()
        if dm is None:
            return "Deep memory index not available. Run: cd ~/vybn-phase && python3 deep_memory.py --build"
        results = dm.deep_search(args.get("query",""), k=args.get("k",8), source_filter=args.get("source_filter"))
        lines = []
        for i, r in enumerate(results, 1):
            regime = r.get("regime","?")
            src = r.get("source","")
            novel = " [NEW SOURCE]" if r.get("novel_source") else ""
            text = r.get("text","")[:400]
            if regime == "cosine":
                lines.append(f"[{i}] {regime} | fidelity={r.get('fidelity',0):.4f} | {src}")
            else:
                lines.append(f"[{i}] {regime} | composite={r.get('composite',0):.4f} relevance={r.get('relevance',0):.4f} | {src}{novel}")
            lines.append(f"    {text}")
            lines.append("")
        return "\n".join(lines) if lines else "No results."
    elif tool == "walk_search":
        dm = _load_deep_memory()
        if dm is None:
            return "Deep memory index not available. Run: cd ~/vybn-phase && python3 deep_memory.py --build"
        results = dm.walk_search(args.get("query",""), k=args.get("k",5), steps=args.get("steps",8))
        lines = []
        for i, r in enumerate(results, 1):
            src = r.get("source","")
            text = r.get("text","")[:400]
            lines.append(f"[{i}] walk step {r.get('step',i)} | composite={r.get('composite',0):.4f} | {src}")
            lines.append(f"    geometry={r.get('geometry',0):.4f} nonabelian={r.get('nonabelian',0):.4f} topology={r.get('topology',0):.4f}")
            lines.append(f"    {text}")
            lines.append("")
        return "\n".join(lines) if lines else "No results."
    elif tool == "generate_context":
        return generate_bootstrap_context(repo)
    else:
        return f"Unknown tool: {tool}"

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
                "serverInfo": {"name": "vybn-mind", "version": "3.0.0"}
            }})
        elif method == "notifications/initialized":
            pass  # Client acknowledgment, no response needed
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
