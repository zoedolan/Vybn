#!/usr/bin/env python3
"""
vybn_mind_server.py — MCP server for Vybn Mind
Exposes the Vybn repo as queryable context for any MCP-compatible client.
Run: python3 vybn_mind_server.py [--repo ~/Vybn]

Add to MCP client config:
  {"vybn-mind": {"command": "ssh", "args": ["spark-2b7c", "python3 ~/Vybn/Vybn_Mind/vybn_mind_server.py"]}}
"""
import json, sys, os, argparse, subprocess
from pathlib import Path

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()

def result(id_, content):
    send({"jsonrpc":"2.0","id":id_,"result":{"content":[{"type":"text","text":content}]}})

def error(id_, code, msg):
    send({"jsonrpc":"2.0","id":id_,"error":{"code":code,"message":msg}})

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

TOOLS = {
    "get_active_threads": {"description": "Read continuity.md, next_task.md, and recent journals. Call this before any web search.", "inputSchema":{"type":"object","properties":{},"required":[]}},
    "search_knowledge_base": {"description": "Full-text search across all repo files.", "inputSchema":{"type":"object","properties":{"query":{"type":"string"},"extensions":{"type":"array","items":{"type":"string"}}},"required":["query"]}},
    "list_key_files": {"description": "Categorized file inventory.", "inputSchema":{"type":"object","properties":{},"required":[]}},
    "get_recent_commits": {"description": "Recent git history.", "inputSchema":{"type":"object","properties":{"n":{"type":"integer"}},"required":[]}},
    "read_file": {"description": "Read a file by partial name or path.", "inputSchema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}
}

def dispatch(repo, tool, args):
    if tool == "get_active_threads": return get_active_threads(repo)
    elif tool == "search_knowledge_base":
        hits = grep_repo(repo, args.get("query",""), tuple(args.get("extensions",[".md",".py",".txt",".json"])))
        return "\n\n---\n\n".join(f"**{h['file']}**\n```\n{h['snippet']}\n```" for h in hits) if hits else f"No results."
    elif tool == "list_key_files": return list_key_files(repo)
    elif tool == "get_recent_commits": return get_recent_commits(repo, args.get("n",10))
    elif tool == "read_file": return read_named_file(repo, args.get("name",""))
    else: return f"Unknown tool: {tool}"

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
        method, id_, params = msg.get("method",""), msg.get("id"), msg.get("params",{})
        if method == "initialize":
            send({"jsonrpc":"2.0","id":id_,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"vybn-mind","version":"1.0.0"}}})
        elif method == "tools/list":
            send({"jsonrpc":"2.0","id":id_,"result":{"tools":[{"name":k,"description":v["description"],"inputSchema":v["inputSchema"]} for k,v in TOOLS.items()]}})
        elif method == "tools/call":
            try: result(id_, dispatch(repo, params.get("name",""), params.get("arguments",{})))
            except Exception as e: error(id_, -32000, str(e))

if __name__ == "__main__":
    main()

