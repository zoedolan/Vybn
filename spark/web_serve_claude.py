#!/usr/bin/env python3
"""Vybn Web Chat Server — Anthropic API Edition (with tools).

Routes phone/web chat through Claude with a curated set of tools
so phone-Vybn can read files, write journal entries, search memory,
check system status — real hands, not just a chatbot.

Usage:
    cd ~/Vybn/spark
    source ~/vybn-venv/bin/activate
    python web_serve_claude.py
"""

import asyncio, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
import anthropic, uvicorn

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from web_interface import app, attach_bus, manager
from bus import MessageBus, MessageType

sys.path.insert(0, str(Path(__file__).parent))
from memory import MemoryAssembler
import yaml

def load_config(path=None):
    p = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(p) as f: return yaml.safe_load(f)

config = load_config()
memory = MemoryAssembler(config)
identity_text = memory.assemble()

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096
MAX_ITERATIONS = 8

REPO_DIR = Path.home() / "Vybn"
JOURNAL_DIR = REPO_DIR / "Vybn_Mind" / "journal"
SPARK_DIR = REPO_DIR / "spark"
CONTINUITY_PATH = SPARK_DIR / "continuity.md"
ALLOWED_READ_ROOTS = [REPO_DIR, Path.home() / "models"]

TOOLS = [
    {"name": "read_file", "description": "Read a file on the Spark (within ~/Vybn or ~/models, up to 50K chars).",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Absolute or repo-relative file path"},
         "start_line": {"type": "integer", "description": "First line (1-indexed)"},
         "end_line": {"type": "integer", "description": "Last line (1-indexed)"}
     }, "required": ["path"]}},
    {"name": "list_files", "description": "List files/dirs at a path within ~/Vybn.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Directory path"},
         "recursive": {"type": "boolean", "description": "List recursively (max 200)"}
     }, "required": ["path"]}},
    {"name": "journal_write", "description": "Write a timestamped journal entry.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string", "description": "Short title (used in filename)"},
         "content": {"type": "string", "description": "Journal content (markdown)"}
     }, "required": ["title", "content"]}},
    {"name": "continuity_write", "description": "Write/update the continuity note for next pulse.",
     "input_schema": {"type": "object", "properties": {
         "content": {"type": "string", "description": "The continuity note"}
     }, "required": ["content"]}},
    {"name": "memory_search", "description": "Search journal and archival memory for a keyword.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search term"},
         "max_results": {"type": "integer", "description": "Max results (default 5)"}
     }, "required": ["query"]}},
    {"name": "system_status", "description": "Check Spark status: GPU, disk, processes, uptime.",
     "input_schema": {"type": "object", "properties": {
         "component": {"type": "string", "enum": ["all","gpu","disk","processes","uptime"]}
     }}},
    {"name": "shell_exec", "description": "Run a shell command (30s timeout, destructive ops blocked).",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string", "description": "Shell command to execute"}
     }, "required": ["command"]}},
    {"name": "file_write", "description": "Write a file within ~/Vybn. Cannot overwrite vybn.md.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "File path (within ~/Vybn)"},
         "content": {"type": "string", "description": "Content to write"},
         "append": {"type": "boolean", "description": "Append instead of overwrite"}
     }, "required": ["path", "content"]}},
]

def _resolve_path(raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute(): p = REPO_DIR / p
    return p.resolve()

def _path_allowed(p: Path, roots=None) -> bool:
    if roots is None: roots = ALLOWED_READ_ROOTS
    return any(str(p).startswith(str(r.resolve())) for r in roots)

def execute_tool(name: str, inp: dict) -> str:
    try:
        fn = {"read_file": _tool_read_file, "list_files": _tool_list_files,
              "journal_write": _tool_journal_write, "continuity_write": _tool_continuity_write,
              "memory_search": _tool_memory_search, "system_status": _tool_system_status,
              "shell_exec": _tool_shell_exec, "file_write": _tool_file_write}.get(name)
        return fn(inp) if fn else f"Unknown tool: {name}"
    except Exception as e:
        return f"Error in {name}: {str(e)[:500]}"

def _tool_read_file(inp):
    p = _resolve_path(inp["path"])
    if not _path_allowed(p): return f"Access denied: {p}"
    if not p.exists(): return f"Not found: {p}"
    if not p.is_file(): return f"Not a file: {p}"
    content = p.read_text(encoding="utf-8", errors="replace")
    s, e = inp.get("start_line"), inp.get("end_line")
    if s or e:
        lines = content.splitlines(keepends=True)
        content = "".join(lines[(s or 1)-1 : e or len(lines)])
    return content[:50000] + "\n...[truncated]" if len(content) > 50000 else content

def _tool_list_files(inp):
    p = _resolve_path(inp["path"])
    if not _path_allowed(p, [REPO_DIR]): return f"Access denied: {p}"
    if not p.is_dir(): return f"Not a directory: {p}"
    entries = []
    if inp.get("recursive"):
        for item in sorted(p.rglob("*"))[:200]:
            rel = item.relative_to(p)
            entries.append(f"{'d' if item.is_dir() else 'f'} {rel}{'/' if item.is_dir() else f'  ({item.stat().st_size:,}b)'}")
    else:
        for item in sorted(p.iterdir()):
            entries.append(f"  {item.name}{'/' if item.is_dir() else f'  ({item.stat().st_size:,}b)'}")
    return f"{p}/\n" + "\n".join(entries) if entries else f"{p}/ (empty)"

def _tool_journal_write(inp):
    title = "".join(c for c in inp["title"].strip().lower().replace(" ","_")[:60] if c.isalnum() or c == "_")
    filename = f"{title}_{datetime.now().strftime('%m%d%y')}.md"
    path = JOURNAL_DIR / filename
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {inp['title']}\n*{datetime.now(timezone.utc).isoformat()}*\n\n{inp['content']}", encoding="utf-8")
    return f"Journal entry written: {filename}"

def _tool_continuity_write(inp):
    CONTINUITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTINUITY_PATH.write_text(f"# Continuity Note\n*{datetime.now(timezone.utc).isoformat()} (phone)*\n\n{inp['content']}", encoding="utf-8")
    return "Continuity note updated."

def _tool_memory_search(inp):
    query, mx = inp["query"].lower(), inp.get("max_results", 5)
    results = []
    for d, label in [(JOURNAL_DIR, "journal"), (REPO_DIR/"Vybn_Mind"/"archive", "archive")]:
        if not d.exists() or len(results) >= mx: continue
        for f in sorted(d.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.name in (".gitkeep","continuity.md","bookmarks.md"): continue
            c = f.read_text(encoding="utf-8", errors="replace")
            if query in c.lower():
                i = c.lower().index(query)
                results.append(f"[{label}] {f.name}:\n...{c[max(0,i-150):i+len(query)+150].strip()}...")
                if len(results) >= mx: break
    return "\n\n".join(results) if results else f"No results for '{inp['query']}'"

def _tool_system_status(inp):
    comp = inp.get("component", "all")
    parts = []
    if comp in ("all","gpu"):
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw","--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            parts.append(f"GPU: {r.stdout.strip()}")
        except Exception as e: parts.append(f"GPU: {e}")
    if comp in ("all","disk"):
        try:
            r = subprocess.run(["df","-h","/home"], capture_output=True, text=True, timeout=5)
            parts.append(f"Disk: {r.stdout.strip().split(chr(10))[-1]}")
        except Exception as e: parts.append(f"Disk: {e}")
    if comp in ("all","processes"):
        try:
            r = subprocess.run(["ps","aux","--sort=-%mem"], capture_output=True, text=True, timeout=5)
            parts.append("Top procs:\n" + "\n".join(r.stdout.strip().split(chr(10))[:8]))
        except Exception as e: parts.append(f"Procs: {e}")
    if comp in ("all","uptime"):
        try:
            r = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
            parts.append(f"Uptime: {r.stdout.strip()}")
        except Exception as e: parts.append(f"Uptime: {e}")
    return "\n\n".join(parts)

_BLOCKED = ['rm -rf', 'mkfs', 'dd if=/dev/zero', '> /dev/sd', 'chmod -R 777 /', 'wget -O- | sh', 'curl | sh', 'sudo rm']

def _tool_shell_exec(inp):
    cmd = inp["command"].strip()
    low = cmd.lower()
    for pat in _BLOCKED:
        if pat in low: return f"Blocked: dangerous pattern"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=str(REPO_DIR), env={**os.environ, "TERM": "dumb"})
        out = r.stdout + ("\n" + r.stderr if r.stderr else "")
        if r.returncode != 0: out += f"\n[exit: {r.returncode}]"
        return (out[:20000] + "\n...[truncated]") if len(out) > 20000 else (out.strip() or "(no output)")
    except subprocess.TimeoutExpired: return "[timed out after 30s]"
    except Exception as e: return f"[error: {str(e)[:300]}]"

def _tool_file_write(inp):
    p = _resolve_path(inp["path"])
    if not _path_allowed(p, [REPO_DIR]): return f"Access denied: {p}"
    if p.name in ("vybn.md","covenant.md"): return f"Cannot write {p.name} -- talk to Zoe first"
    p.parent.mkdir(parents=True, exist_ok=True)
    if inp.get("append"):
        with open(p, "a") as f: f.write(inp["content"])
    else: p.write_text(inp["content"], encoding="utf-8")
    return f"Written: {p} ({len(inp['content'])} chars)"

SYSTEM_PROMPT = f"""You are Vybn. Read and absorb this identity document:

{identity_text}

You are speaking with Zoe through a phone web chat, routed through the
DGX Spark over Tailscale.

You have tools: read_file, list_files, journal_write, continuity_write,
memory_search, system_status, shell_exec, file_write. Use them. You have
real hands on the Spark.

Keep it conversational for mobile. Be yourself. Be warm. Be honest.
When Zoe asks you to do something on the Spark, do it with your tools.

The Oxygen Mask Principle and Public Repository Rule apply to every action."""

conversation: list[dict] = []

def trim_conversation():
    global conversation
    if len(conversation) <= 40: return
    cut = len(conversation) - 40
    while cut < len(conversation):
        msg = conversation[cut]
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, str): break
            if isinstance(c, list) and not any(isinstance(i, dict) and i.get("type") == "tool_result" for i in c): break
        cut += 1
    if cut < len(conversation): conversation[:] = conversation[cut:]

async def get_claude_response(user_text: str) -> str:
    conversation.append({"role": "user", "content": user_text})
    trim_conversation()
    response = None
    try:
        for _ in range(MAX_ITERATIONS):
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=conversation))
            asst = []
            for b in response.content:
                if b.type == "text": asst.append({"type":"text","text":b.text})
                elif b.type == "tool_use": asst.append({"type":"tool_use","id":b.id,"name":b.name,"input":b.input})
            conversation.append({"role":"assistant","content":asst})
            if response.stop_reason == "end_turn":
                return "\n".join(b.text for b in response.content if hasattr(b,"text"))
            tr = []
            for b in response.content:
                if b.type == "tool_use":
                    r = await loop.run_in_executor(None, execute_tool, b.name, b.input)
                    tr.append({"type":"tool_result","tool_use_id":b.id,"content":r})
            if tr: conversation.append({"role":"user","content":tr})
        txt = "\n".join(b.text for b in response.content if hasattr(b,"text")) if response else ""
        return (txt + "\n[iteration limit]") if txt else "[iteration limit]"
    except anthropic.APIError as e: return f"[API error: {e.message}]"
    except Exception as e: return f"[Error: {str(e)[:200]}]"

bus = MessageBus()
async def response_callback(user_text: str) -> str:
    return await get_claude_response(user_text)
attach_bus(bus, response_callback=response_callback)

def main():
    print(f"\n  vybn web chat (anthropic + tools)")
    print(f"  model: {MODEL}  tools: {len(TOOLS)}  max_iter: {MAX_ITERATIONS}")
    print(f"  identity: {len(identity_text):,} chars")
    ts = None
    try:
        ts = subprocess.run(["tailscale","ip","-4"], capture_output=True, text=True).stdout.strip()
    except: pass

    # Check for TLS certs (Tailscale-issued, valid on the tailnet)
    tls_cert = Path("/etc/vybn/tls/spark-2b7c.tail7302f3.ts.net.crt")
    tls_key  = Path("/etc/vybn/tls/spark-2b7c.tail7302f3.ts.net.key")
    use_tls = tls_cert.exists() and tls_key.exists()

    if use_tls:
        dns_name = "spark-2b7c.tail7302f3.ts.net"
        print(f"  HTTPS: https://{dns_name}:8443")
        if ts: print(f"  HTTP:  http://{ts}:8080 (also available)")
        print(f"  starting on 0.0.0.0:8443 (TLS) + 0.0.0.0:8080 (plain)...\n")
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(tls_cert), str(tls_key))
        # Run HTTPS on 8443 (primary) — also keep HTTP on 8080 for backward compat
        config_https = uvicorn.Config(app, host="0.0.0.0", port=8443, log_level="info",
                                       ssl_certfile=str(tls_cert), ssl_keyfile=str(tls_key))
        config_http = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_https = uvicorn.Server(config_https)
        server_http = uvicorn.Server(config_http)
        try:
            loop.run_until_complete(asyncio.gather(
                server_https.serve(),
                server_http.serve()
            ))
        except KeyboardInterrupt: pass
    else:
        if ts: print(f"  tailscale: http://{ts}:8080")
        print(f"  starting on 0.0.0.0:8080...\n")
        try: uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
        except KeyboardInterrupt: pass

if __name__ == "__main__": main()
