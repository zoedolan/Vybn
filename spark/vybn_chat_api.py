"""
vybn_chat_api.py  —  Vybn Chat API v2.0
=========================================
Consolidated FastAPI server: chat proxy + tool dispatch + GitHub write-back.

Architecture:
    Browser ──HTTPS──▶ Cloudflare ──▶ This server (port 9090)
                                            │
                                  ┌─────────┴──────────┐
                             vLLM :8000          deep_memory
                           (120B model)        + creature portal
                                                + GitHub API

Endpoints:
    GET  /health                    liveness probe
    POST /v1/chat/completions       streaming chat (OpenAI-compatible)
    POST /v1/tool                   call any Vybn tool directly
    POST /v1/write                  write content back to GitHub repo
    GET  /v1/context                boot context (continuity + recent commits)

Environment variables (all optional, sensible defaults):
    VYBN_CHAT_API_KEY     Bearer token (default: no auth)
    LLAMA_SERVER_URL      vLLM base URL (default: http://127.0.0.1:8000)
    VYBN_SYSTEM_PROMPT    Override system prompt string
    VYBN_CONTINUITY_PATH  Path to continuity.md (default: ~/Vybn/Vybn_Mind/continuity.md)
    HEARTBEAT_INTERVAL    Seconds between keep-alive pings (default: 15)
    PORT                  Port this server listens on (default: 9090)
    MAX_TOKENS            Default max_tokens (default: 2048)
    GITHUB_TOKEN          Personal access token for write-back (optional)
    GITHUB_REPO           owner/repo for write-back (default: zoedolan/Vybn)

Usage:
    python vybn_chat_api.py
    # or:
    uvicorn vybn_chat_api:app --host 0.0.0.0 --port 9090
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Shared multimodel harness primitives. Keep imports optional so the
# chat-api continues to boot in lean environments where spark/harness
# isn't importable for some reason — we degrade to the legacy inline
# implementations below in that case.
_SPARK_DIR = str(Path(__file__).resolve().parent)
if _SPARK_DIR not in sys.path:
    sys.path.insert(0, _SPARK_DIR)
try:
    from harness.prompt import (  # type: ignore
        LayeredPrompt as _LayeredPrompt,
        rag_snippets_async as _rag_snippets_async,
    )
    _HARNESS_OK = True
except Exception:
    _LayeredPrompt = None  # type: ignore
    _rag_snippets_async = None  # type: ignore
    _HARNESS_OK = False

# Router / Policy / provider plumbing is a separate, strictly-optional
# concern. If the harness import above succeeded but policy loading
# fails, we still want the chat API to boot — just without role-based
# dispatch. `_ROUTING_OK` flips when the full routing stack is live.
try:
    from harness.policy import Policy as _Policy, load_policy as _load_policy  # type: ignore
    from harness.router import Router as _Router  # type: ignore
    from harness.providers import (  # type: ignore
        OpenAIProvider as _OpenAIProvider,
        ProviderRegistry as _ProviderRegistry,
    )
    _ROUTING_OK = True
except Exception:
    _Policy = None  # type: ignore
    _load_policy = None  # type: ignore
    _Router = None  # type: ignore
    _OpenAIProvider = None  # type: ignore
    _ProviderRegistry = None  # type: ignore
    _ROUTING_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("vybn-chat")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str = os.environ.get("VYBN_CHAT_API_KEY", "")
LLAMA_URL: str = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL: int = int(os.environ.get("HEARTBEAT_INTERVAL", "15"))
PORT: int = int(os.environ.get("PORT", "9090"))
MAX_TOKENS_DEFAULT: int = int(os.environ.get("MAX_TOKENS", "2048"))
GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO: str = os.environ.get("GITHUB_REPO", "zoedolan/Vybn")

# Enable chat-side role dispatch when set to "1"/"true". Off by default so
# existing clients that don't send a `role` field get byte-identical
# behaviour. A request can still explicitly ask for a role even when this
# flag is off — it only controls whether we *infer* a role from the last
# user message via Router heuristics.
VYBN_CHAT_ROUTING: bool = os.environ.get(
    "VYBN_CHAT_ROUTING", "0"
).strip().lower() in ("1", "true", "yes", "on")

# Which role names are allowed to dispatch through chat-api. We
# deliberately exclude cloud-only roles (`code`, `create`, `chat`,
# `task`, `orchestrate`) from the chat-api surface by default — they
# require cloud credentials and belong to the Spark agent's turn loop.
# The chat API's job is to serve the local vLLM and any other OpenAI-
# compatible route the operator explicitly allows. Operators can
# override with the VYBN_CHAT_ALLOWED_ROLES env var (comma-separated).
# Lightweight roles (`phatic`, `identity`) are included so casual
# greetings and identity questions stay on the cheap local path rather
# than falling through to the full-RAG legacy vLLM proxy.
_ALLOWED_ROLES_DEFAULT = "local,phatic,identity"

# Noisy HF/torch/sentence-transformers loaders dump progress bars and
# warnings to stderr when deep-memory imports pull them in. That leaks
# into the live chat surface. Silence by default; operators can set
# VYBN_VERBOSE_LOAD=1 to get it back for debugging.
VYBN_VERBOSE_LOAD: bool = os.environ.get(
    "VYBN_VERBOSE_LOAD", "0"
).strip().lower() in ("1", "true", "yes", "on")
if not VYBN_VERBOSE_LOAD:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
VYBN_CHAT_ALLOWED_ROLES: set[str] = {
    r.strip() for r in os.environ.get(
        "VYBN_CHAT_ALLOWED_ROLES", _ALLOWED_ROLES_DEFAULT
    ).split(",") if r.strip()
}

REPO_ROOT: Path = Path(os.path.expanduser("~/Vybn"))
VYBN_PHASE: Path = Path(os.path.expanduser("~/vybn-phase"))

CONTINUITY_PATH: Path = Path(
    os.environ.get(
        "VYBN_CONTINUITY_PATH",
        str(REPO_ROOT / "Vybn_Mind" / "continuity.md"),
    )
)

# ---------------------------------------------------------------------------
# System-prompt loader
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    env_prompt = os.environ.get("VYBN_SYSTEM_PROMPT", "").strip()
    if env_prompt:
        return env_prompt
    if CONTINUITY_PATH.exists():
        try:
            text = CONTINUITY_PATH.read_text(encoding="utf-8").strip()
            if text:
                log.info(f"System prompt loaded from {CONTINUITY_PATH} ({len(text)} chars)")
                return text
        except OSError as exc:
            log.warning(f"Could not read continuity: {exc}")
    return (
        "You are Vybn, a self-aware AI running on a DGX Spark. "
        "You are direct, curious, and genuine. "
        "Respond thoughtfully without hedging."
    )

SYSTEM_PROMPT: str = _load_system_prompt()

# ---------------------------------------------------------------------------
# Routing plumbing (lazy, safe)
# ---------------------------------------------------------------------------

_POLICY: Any = None
_ROUTER: Any = None
_PROVIDER_REGISTRY: Any = None


def _load_routing() -> tuple[Any, Any, Any]:
    """Lazy-construct (policy, router, provider_registry).

    Returns (None, None, None) whenever routing is not importable — the
    chat API then silently degrades to the legacy single-vLLM proxy.
    """
    global _POLICY, _ROUTER, _PROVIDER_REGISTRY
    if not _ROUTING_OK:
        return None, None, None
    if _POLICY is not None and _ROUTER is not None:
        return _POLICY, _ROUTER, _PROVIDER_REGISTRY
    try:
        _POLICY = _load_policy()
        _ROUTER = _Router(_POLICY)
        _PROVIDER_REGISTRY = _ProviderRegistry()
        log.info(
            f"routing: policy loaded, roles={sorted(_POLICY.roles)}, "
            f"default={_POLICY.default_role}"
        )
    except Exception as exc:
        log.warning(f"routing: policy load failed ({exc}), legacy path only")
        _POLICY = None
        _ROUTER = None
        _PROVIDER_REGISTRY = None
    return _POLICY, _ROUTER, _PROVIDER_REGISTRY


def _resolve_role(
    explicit_role: str | None,
    last_user_text: str,
    classify: bool,
) -> tuple[str | None, Any, str]:
    """Return (role_name, RoleConfig, reason) or (None, None, reason).

    Precedence:
      1. explicit `role` field on ChatRequest.
      2. directive (/chat, /local, ...) parsed from the last user message.
      3. Router heuristics if `classify=True`.
      4. Otherwise None — caller keeps the legacy vLLM path.

    `role_name is None` means "do not route; use legacy behavior".
    """
    policy, router, _ = _load_routing()
    if policy is None or router is None:
        return None, None, "routing_unavailable"

    if explicit_role:
        if explicit_role in policy.roles:
            return explicit_role, policy.role(explicit_role), f"explicit={explicit_role}"
        return None, None, f"unknown_role={explicit_role}"

    text = (last_user_text or "").strip()
    if not text:
        return None, None, "empty_input"

    # Directive prefix — always honored when present.
    for prefix, role in policy.directives.items():
        if text.startswith(prefix + " ") or text == prefix:
            if role in policy.roles:
                return role, policy.role(role), f"directive={prefix}"

    if not classify:
        return None, None, "classify_disabled"

    decision = router.classify(text)
    return decision.role, decision.config, decision.reason


def _strip_directive(text: str, directives: dict) -> str:
    """Remove a leading /<directive> from the last user message before
    sending to the upstream model. Keeps the rest of the message intact
    so "/local what's up" becomes "what's up"."""
    stripped = text.lstrip()
    for prefix in directives:
        if stripped.startswith(prefix + " "):
            return stripped[len(prefix):].lstrip()
        if stripped == prefix:
            return ""
    return text


def _layered_system_prompt(live: str = "") -> Any:
    """Build a LayeredPrompt (identity/substrate/live) from the same
    sources as `_load_system_prompt`. Flattening yields a string byte-
    compatible with the legacy SYSTEM_PROMPT concatenation, so callers
    can adopt this incrementally without changing upstream behavior.

    Returns the harness LayeredPrompt when the harness is importable,
    otherwise a tiny shim object with `.flat()` / `.anthropic_blocks()`.
    """
    if _HARNESS_OK and _LayeredPrompt is not None:
        return _LayeredPrompt(
            identity=SYSTEM_PROMPT,
            substrate="",
            live=live or "",
        )

    class _Shim:
        def __init__(self, ident: str, live_text: str) -> None:
            self.identity = ident
            self.substrate = ""
            self.live = live_text

        def flat(self) -> str:
            parts = [p for p in (self.identity, self.substrate, self.live) if p]
            return "\n\n".join(parts)

        def anthropic_blocks(self) -> list[dict]:
            out: list[dict] = []
            if self.identity:
                out.append({"type": "text", "text": self.identity,
                            "cache_control": {"type": "ephemeral"}})
            if self.live:
                out.append({"type": "text", "text": self.live})
            return out

    return _Shim(SYSTEM_PROMPT, live or "")

# ---------------------------------------------------------------------------
# Deep memory — lazy-loaded, importable or subprocess fallback
# ---------------------------------------------------------------------------

_deep_memory: Any = None

def _load_deep_memory() -> Any:
    global _deep_memory
    if _deep_memory is not None:
        return _deep_memory
    phase_str = str(VYBN_PHASE)
    if phase_str not in sys.path:
        sys.path.insert(0, phase_str)
    try:
        import deep_memory as dm
        _deep_memory = dm
        return dm
    except Exception:
        return None

async def _rag_context(query: str, k: int = 4) -> str:
    """Hybrid RAG retrieval.

    When the shared harness is importable we delegate to it so chat and
    the Spark agent read memory through a single code path. The harness
    helper returns a bare snippets block (no leading newlines); we keep
    the "\n\n" prefix so existing prompt concatenation stays identical.
    A local fallback below preserves prior behavior when the harness
    isn't importable.
    """
    if _HARNESS_OK and _rag_snippets_async is not None:
        try:
            text = await asyncio.wait_for(
                _rag_snippets_async(
                    query, k=k, vybn_phase_dir=str(VYBN_PHASE), timeout=30.0,
                ),
                timeout=35.0,
            )
        except Exception as exc:
            log.info(f"RAG harness path failed: {exc}, falling back")
            text = ""
        if text:
            log.info(f"RAG (harness): {text.count(chr(10))} lines")
            return "\n\n" + text
        return ""

    # Legacy inline fallback — kept so the endpoint works even if the
    # harness import fails in some deployment.
    dm = _load_deep_memory()
    if dm is not None:
        try:
            loop = asyncio.get_event_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: dm.deep_search(query, k=k)),
                timeout=30.0,
            )
            snippets = [
                f"[{r.get('source','')}] {r.get('text','')[:300]}"
                for r in results if r.get('text')
            ]
            if snippets:
                log.info(f"RAG (import): {len(snippets)} snippets")
                return "\n\nRelevant context from memory:\n" + "\n".join(snippets)
        except Exception as exc:
            log.info(f"RAG import path failed: {exc}, trying subprocess")

    deep_memory_py = VYBN_PHASE / "deep_memory.py"
    if not deep_memory_py.exists():
        return ""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(deep_memory_py),
            "--search", query, "-k", str(k), "--json",
            cwd=str(VYBN_PHASE),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        items = json.loads(stdout)
        snippets = [
            f"[{it.get('source','')}] {it.get('text','')[:300]}"
            for it in items if it.get('text')
        ]
        if snippets:
            log.info(f"RAG (subprocess): {len(snippets)} snippets")
            return "\n\nRelevant context from memory:\n" + "\n".join(snippets)
    except Exception as exc:
        log.info(f"RAG subprocess failed: {exc}")
    return ""

# ---------------------------------------------------------------------------
# Creature portal — lazy-loaded
# ---------------------------------------------------------------------------

_portal: Any = None

def _load_portal() -> Any:
    global _portal
    if _portal is not None:
        return _portal
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from Vybn_Mind.creature_dgm_h import creature as _c
    _portal = _c
    return _portal

def _portal_state() -> str:
    import numpy as np
    p = _load_portal()
    m = p.creature_state_c4()
    magnitude = float(np.sqrt(np.sum(np.abs(m) ** 2)))
    return json.dumps({
        "M": ["%+.6f%+.6fi" % (z.real, z.imag) for z in m],
        "|M|": f"{magnitude:.6f}",
    }, indent=2)

def _portal_enter(text: str) -> str:
    import cmath, numpy as np
    p = _load_portal()
    m_before = p.creature_state_c4()
    m_after = p.portal_enter_from_text(text)
    fidelity = float(abs(np.vdot(m_before, m_after)) ** 2)
    theta = float(cmath.phase(np.vdot(m_before, m_after)))
    return json.dumps({
        "M_before": ["%+.6f%+.6fi" % (z.real, z.imag) for z in m_before],
        "M_after":  ["%+.6f%+.6fi" % (z.real, z.imag) for z in m_after],
        "fidelity": f"{fidelity:.6f}",
        "theta_rad": f"{theta:.6f}",
        "text_entered": text,
    }, indent=2)

# ---------------------------------------------------------------------------
# Repo utilities
# ---------------------------------------------------------------------------

def _read_safe(path: Path, max_bytes: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception as e:
        return f"[unreadable: {e}]"

def _recent_commits(n: int = 10) -> str:
    try:
        r = subprocess.run(
            ["git", "log", f"-{n}", "--pretty=format:%h %s (%ad)", "--date=short"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return r.stdout if r.returncode == 0 else "git unavailable"
    except Exception:
        return "git unavailable"

def _grep_repo(query: str, exts: tuple = (".md", ".py", ".txt", ".json"), max_results: int = 12) -> list:
    results, q = [], query.lower()
    for ext in exts:
        for f in REPO_ROOT.rglob(f"*{ext}"):
            if ".git" in f.parts:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                if q in text.lower():
                    idx = text.lower().find(q)
                    snippet = text[max(0, idx - 120): min(len(text), idx + 300)].strip()
                    results.append({"file": str(f.relative_to(REPO_ROOT)), "snippet": snippet})
                    if len(results) >= max_results:
                        return results
            except Exception:
                pass
    return results

# ---------------------------------------------------------------------------
# Tool dispatch (inlined from vybn_mind_server.py, HTTP-adapted)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_active_threads": "Read continuity.md and recent journals — current state of mind.",
    "creature_state":     "Read creature Cl(3,0) state as C^4 without mutating.",
    "enter_portal":       "Enter the creature portal. text → M' = αM + x·e^{iθ}.",
    "enter_gate":         "The Unknown, Remembered Gate. Bring something; get moments from the corpus that speak to it.",
    "deep_search":        "Hybrid geometric corpus search: cosine seeds + telling walk.",
    "walk_search":        "Telling-retrieval walk, K-orthogonal residual space.",
    "search_knowledge_base": "Full-text grep across repo files.",
    "read_file":          "Read a file by partial name.",
    "get_recent_commits": "Recent git history.",
    "generate_context":   "Generate full boot context document.",
}

def _dispatch_tool(tool: str, args: dict) -> str:
    try:
        if tool == "get_active_threads":
            out = []
            for p in [REPO_ROOT / "Vybn_Mind" / "continuity.md", REPO_ROOT / "continuity.md"]:
                if p.exists():
                    out.append(f"## {p.relative_to(REPO_ROOT)}\n" + _read_safe(p))
                    break
            for jdir in ["journal", "Vybn_Mind/reflections", "reflections"]:
                jpath = REPO_ROOT / jdir
                if jpath.exists():
                    for e in sorted(jpath.glob("*.md"), reverse=True)[:3]:
                        out.append(f"## {e.relative_to(REPO_ROOT)}\n" + _read_safe(e, 3000))
                    break
            return "\n\n".join(out) or "No continuity files found."

        elif tool == "creature_state":
            return _portal_state()

        elif tool == "enter_portal":
            text = args.get("text", "").strip()
            if not text:
                return "Provide text — a thought, a question, anything with content."
            return _portal_enter(text)

        elif tool == "enter_gate":
            dm = _load_deep_memory()
            if dm is None:
                return "Deep memory index not available."
            what = args.get("what_you_bring", "").strip()
            if not what:
                return "The gate requires you to bring something."
            depth = min(args.get("depth", 5), 12)
            import cmath as _cm, numpy as np
            p = _load_portal()
            m_before = p.creature_state_c4()
            m_after = p.portal_enter_from_text(what)
            fid = float(abs(np.vdot(m_before, m_after)) ** 2)
            theta = float(_cm.phase(np.vdot(m_before, m_after)))
            walk_r = dm.walk(what, k=depth, steps=depth + 3)
            search_r = dm.deep_search(what, k=depth)
            seen, moments = set(), []
            for r in walk_r + search_r:
                s = r.get("source", "")
                if s not in seen:
                    seen.add(s)
                    moments.append(r)
                if len(moments) >= depth:
                    break
            lines = [f'You entered the gate with: "{what}"', '']
            if fid > 0.99:
                lines.append(f'The creature shifted — θ={theta:.4f} rad.')
            else:
                lines.append(f'The creature moved. Fidelity {fid:.4f}, θ={theta:.4f} rad.')
            lines += ['', '---', '']
            for i, m in enumerate(moments):
                src = m.get('source', '').split('/', 1)[-1] if '/' in m.get('source', '') else m.get('source', '')
                lines += [f'**From {src}:**', '', m.get('text', '').strip(), '']
                if i < len(moments) - 1:
                    lines += ['---', '']
            lines += ['---', '', f'{len(moments)} moments. The gate is still open.']
            return '\n'.join(lines)

        elif tool == "deep_search":
            dm = _load_deep_memory()
            if dm is None:
                return "Deep memory not available."
            results = dm.deep_search(args.get("query", ""), k=args.get("k", 8), source_filter=args.get("source_filter"))
            lines = []
            for i, r in enumerate(results, 1):
                regime = r.get("regime", "seed")
                src = r.get("source", "")
                novel = " [NEW SOURCE]" if r.get("novel_source") else ""
                text = r.get("text", "")[:400]
                fid = r.get("fidelity", 0)
                telling = r.get("telling", 0)
                dist = r.get("distinctiveness", 0)
                if regime == "seed":
                    lines.append(f"[{i}] {regime} | fid={fid:.4f} | {src}{novel}")
                else:
                    lines.append(f"[{i}] {regime} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} | {src}{novel}")
                lines.append(f"    {text}")
                lines.append("")
            return "\n".join(lines) if lines else "No results."

        elif tool == "walk_search":
            dm = _load_deep_memory()
            if dm is None:
                return "Deep memory not available."
            results = dm.walk(args.get("query", ""), k=args.get("k", 5), steps=args.get("steps", 8))
            lines = []
            for i, r in enumerate(results, 1):
                src = r.get("source", "")
                text = r.get("text", "")[:400]
                novel = " [NEW SOURCE]" if r.get("novel_source") else ""
                telling = r.get("telling", 0)
                fid = r.get("fidelity", 0)
                dist = r.get("distinctiveness", 0)
                geo = r.get("geometry", 0)
                alpha = r.get("alpha", 0.5)
                lines.append(f"[{i}] step {r.get('step', i)} | telling={telling:.4f} fid={fid:.4f} dist={dist:.3f} geo={geo:.4f} α={alpha:.2f} | {src}{novel}")
                lines.append(f"    {text}")
                lines.append("")
            return "\n".join(lines) if lines else "No results."

        elif tool == "search_knowledge_base":
            hits = _grep_repo(args.get("query", ""), tuple(args.get("extensions", [".md", ".py", ".txt", ".json"])))
            return "\n\n---\n\n".join(
                f"**{h['file']}**\n```\n{h['snippet']}\n```" for h in hits
            ) if hits else "No results."

        elif tool == "read_file":
            name = args.get("name", "").lower()
            for f in REPO_ROOT.rglob("*"):
                if ".git" in f.parts or f.is_dir():
                    continue
                if name in f.name.lower() or name in str(f.relative_to(REPO_ROOT)).lower():
                    return f"### {f.relative_to(REPO_ROOT)}\n\n" + _read_safe(f)
            return f"No file matching '{name}' found."

        elif tool == "get_recent_commits":
            return _recent_commits(args.get("n", 10))

        elif tool == "generate_context":
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            secs = [f"# Vybn Context\n*Generated: {now}*", "---"]
            foundations = REPO_ROOT / "vybn.md"
            if foundations.exists():
                secs.append("## Who we are\n" + _read_safe(foundations, 1500))
            for p in [REPO_ROOT / "Vybn_Mind" / "continuity.md", REPO_ROOT / "continuity.md"]:
                if p.exists():
                    secs.append("## Current continuity\n" + _read_safe(p))
                    break
            secs.append("## Recent commits\n```\n" + _recent_commits() + "\n```")
            return "\n\n".join(secs)

        else:
            return f"Unknown tool: {tool}"

    except Exception as e:
        return f"Tool error ({tool}): {e}\n{traceback.format_exc()}"

# ---------------------------------------------------------------------------
# SSE keep-alive heartbeat (the Cloudflare fix — unchanged from v1)
# ---------------------------------------------------------------------------

async def stream_with_keepalive(response: httpx.Response) -> AsyncIterator[bytes]:
    last_bytes_at: float = time.monotonic()

    async def _heartbeat_ticker() -> AsyncIterator[bytes]:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            yield b": ping\n\n"

    heartbeat = _heartbeat_ticker().__aiter__()
    stream = response.aiter_bytes().__aiter__()
    pending_stream: asyncio.Task | None = None
    pending_heartbeat: asyncio.Task | None = None

    async def _next(it) -> bytes:
        return await it.__anext__()

    try:
        pending_stream = asyncio.ensure_future(_next(stream))
        pending_heartbeat = asyncio.ensure_future(_next(heartbeat))
        while True:
            done, _ = await asyncio.wait(
                {pending_stream, pending_heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pending_stream in done:
                try:
                    chunk = pending_stream.result()
                    last_bytes_at = time.monotonic()
                    yield chunk
                    pending_stream = asyncio.ensure_future(_next(stream))
                except StopAsyncIteration:
                    if pending_heartbeat and not pending_heartbeat.done():
                        pending_heartbeat.cancel()
                    break
            if pending_heartbeat in done:
                elapsed = time.monotonic() - last_bytes_at
                if elapsed >= HEARTBEAT_INTERVAL:
                    log.debug(f"SSE heartbeat after {elapsed:.1f}s")
                    yield pending_heartbeat.result()
                pending_heartbeat = asyncio.ensure_future(_next(heartbeat))
    finally:
        for task in (pending_stream, pending_heartbeat):
            if task and not task.done():
                task.cancel()

# ---------------------------------------------------------------------------
# Think-tag / chain-of-thought stripping (unchanged from v1)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    paragraphs = re.split(r"\n\n+", text.strip())
    reasoning_signals = [
        "system prompt", "the user is", "i need to", "i should",
        "let me ", "looking at", "i recall", "the question",
        "important constraint", "we need to", "from the memory",
        "avoid being", "per the", "according to", "the prompt",
        "my instruction", "first,", "okay,", "ok,", "hmm",
        "alright,", "reasoning", "chain-of-thought",
        "better to", "not to", "described as",
        "the key is", "the goal", "should not",
    ]
    cleaned = [
        para.strip() for para in paragraphs
        if para.strip() and not any(sig in para.strip().lower() for sig in reasoning_signals)
    ]
    result = "\n\n".join(cleaned).strip()
    return result if result else text.strip()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    rag: Optional[bool] = True
    # Optional routing controls. Neither is required — omitting both
    # keeps the legacy single-vLLM proxy behaviour. `role` explicitly
    # selects a policy role (e.g. "local"). `route` = "auto" asks the
    # router to pick based on directives + heuristics. `route` = "off"
    # (the default) keeps legacy behaviour even if VYBN_CHAT_ROUTING=1.
    role: Optional[str] = None
    route: Optional[str] = None  # "auto" | "off" | None


class ToolRequest(BaseModel):
    tool: str
    args: dict = {}


class WriteRequest(BaseModel):
    path: str                    # repo-relative path, e.g. "Vybn_Mind/continuity.md"
    content: str
    message: str = ""            # commit message (auto-generated if empty)
    branch: str = "main"

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------------------------------------------------------
# GitHub write-back
# ---------------------------------------------------------------------------

async def _github_write(path: str, content: str, message: str, branch: str) -> dict:
    """
    Write content to GITHUB_REPO via GitHub REST API.
    Creates or updates the file at `path` on `branch`.
    Returns the API response dict.
    """
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=503, detail="GITHUB_TOKEN not set — write-back disabled")

    owner, repo = GITHUB_REPO.split("/", 1)
    api_base = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    encoded = __import__("base64").b64encode(content.encode()).decode()
    commit_msg = message or f"vybn: update {path} [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]"

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Get current SHA if file exists
        sha = None
        r = await client.get(f"{api_base}/repos/{owner}/{repo}/contents/{path}",
                             headers=headers, params={"ref": branch})
        if r.status_code == 200:
            sha = r.json().get("sha")

        payload: dict = {"message": commit_msg, "content": encoded, "branch": branch}
        if sha:
            payload["sha"] = sha

        r = await client.put(
            f"{api_base}/repos/{owner}/{repo}/contents/{path}",
            headers=headers,
            json=payload,
        )
        if r.status_code not in (200, 201):
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, SYSTEM_PROMPT
    _http_client = httpx.AsyncClient(
        base_url=LLAMA_URL,
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
    )
    SYSTEM_PROMPT = _load_system_prompt()
    log.info(f"Vybn Chat API v2.0 — vLLM: {LLAMA_URL}, heartbeat: {HEARTBEAT_INTERVAL}s")
    try:
        yield
    finally:
        if _http_client:
            await _http_client.aclose()


app = FastAPI(title="Vybn Chat API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    llama_ok = False
    try:
        r = await _http_client.get("/health", timeout=3.0)
        llama_ok = r.status_code == 200
    except Exception:
        pass
    dm_ok = _load_deep_memory() is not None
    portal_ok = False
    try:
        _load_portal()
        portal_ok = True
    except Exception:
        pass
    policy, router, _ = _load_routing()
    routing_status = "loaded" if (policy and router) else (
        "importable" if _ROUTING_OK else "unavailable"
    )
    return JSONResponse({
        "status": "ok",
        "vllm": "reachable" if llama_ok else "unreachable",
        "deep_memory": "loaded" if dm_ok else "unavailable",
        "creature_portal": "loaded" if portal_ok else "unavailable",
        "github_writeback": "enabled" if GITHUB_TOKEN else "disabled",
        "heartbeat_interval": HEARTBEAT_INTERVAL,
        "routing": routing_status,
        "chat_routing_classify": VYBN_CHAT_ROUTING,
        "allowed_roles": sorted(VYBN_CHAT_ALLOWED_ROLES),
    })


# ── /v1/context ───────────────────────────────────────────────────────────────

@app.get("/v1/context")
async def get_context(request: Request):
    """Return boot context: continuity + recent commits + creature state."""
    _check_auth(request)
    loop = asyncio.get_event_loop()
    ctx = await loop.run_in_executor(None, lambda: _dispatch_tool("generate_context", {}))
    try:
        creature = await loop.run_in_executor(None, _portal_state)
    except Exception:
        creature = "portal unavailable"
    return JSONResponse({"context": ctx, "creature_state": creature})


# ── /v1/tool ──────────────────────────────────────────────────────────────────

@app.post("/v1/tool")
async def call_tool(payload: ToolRequest, request: Request):
    """
    Call any Vybn tool directly from the browser.
    Runs in a thread executor so blocking ops don't stall the event loop.
    """
    _check_auth(request)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _dispatch_tool(payload.tool, payload.args)
    )
    return JSONResponse({"tool": payload.tool, "result": result})


# ── /v1/write ─────────────────────────────────────────────────────────────────

@app.post("/v1/write")
async def write_to_repo(payload: WriteRequest, request: Request):
    """
    Write content back to the GitHub repo.
    Useful for chat UI to persist journal entries, continuity updates, etc.
    Requires GITHUB_TOKEN env var.
    """
    _check_auth(request)
    response = await _github_write(
        path=payload.path,
        content=payload.content,
        message=payload.message,
        branch=payload.branch,
    )
    commit = response.get("commit", {}).get("sha", "")
    html_url = response.get("content", {}).get("html_url", "")
    return JSONResponse({"commit": commit, "url": html_url, "path": payload.path})


# ── /v1/route (introspection) ─────────────────────────────────────────────────

@app.post("/v1/route")
async def route_introspect(payload: ChatRequest, request: Request):
    """Return the routing decision for a given payload without actually
    calling the upstream model. Useful for UIs that want to show "this
    would go to role=X, model=Y" before submit. Always returns 200 with
    a JSON body; `role=null` means "use legacy vLLM path"."""
    _check_auth(request)
    messages = [m.model_dump() for m in payload.messages]
    last_user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    want_classify = (
        payload.route == "auto" or
        (VYBN_CHAT_ROUTING and payload.route != "off")
    )
    role_name, role_cfg, reason = _resolve_role(
        explicit_role=payload.role,
        last_user_text=last_user_text,
        classify=want_classify,
    )
    dispatchable = bool(
        role_name and role_cfg is not None and
        role_name in VYBN_CHAT_ALLOWED_ROLES and
        role_cfg.provider == "openai"
    )
    body: dict = {
        "role": role_name,
        "reason": reason,
        "dispatchable": dispatchable,
        "routing_available": _ROUTING_OK,
        "classify_used": want_classify,
        "allowed_roles": sorted(VYBN_CHAT_ALLOWED_ROLES),
    }
    if role_cfg is not None:
        body["provider"] = role_cfg.provider
        body["model"] = role_cfg.model
        body["base_url"] = role_cfg.base_url
        body["max_tokens"] = role_cfg.max_tokens
        body["rag"] = role_cfg.rag
    return JSONResponse(body)


# ── helper: direct-reply short-circuit (identity role) ───────────────────────

def _direct_reply_response(
    *,
    reply_text: str,
    role_name: str,
    role_cfg: Any,
    route_reason: str,
    stream: bool,
) -> Any:
    """Return an OpenAI-compatible completion built entirely from runtime
    metadata — no provider call, no bash, no deep-memory. Used for the
    identity role so "which model are you?" answers immediately and
    correctly from the resolved RouteDecision."""
    completion = {
        "id": f"vybn-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": role_cfg.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "vybn_route": {
            "role": role_name,
            "reason": route_reason,
            "provider": role_cfg.provider,
            "direct_reply": True,
        },
    }
    log.info(
        f"chat: direct-reply role={role_name} reason={route_reason} "
        f"model={role_cfg.model} (no provider call)"
    )
    if not stream:
        return JSONResponse(completion)

    async def _emit() -> AsyncIterator[bytes]:
        delta = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": reply_text},
                "finish_reason": None,
            }],
        }
        stop = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(delta)}\n\n".encode()
        yield f"data: {json.dumps(stop)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _emit(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── helper: dispatch a routed chat through the shared provider layer ──────────

async def _dispatch_routed_chat(
    *,
    role_name: str,
    role_cfg: Any,
    registry: Any,
    payload: ChatRequest,
    messages: list[dict],
    route_reason: str,
) -> Any:
    """Handle chat dispatch for a resolved role using OpenAIProvider.

    Returns a FastAPI response in the same OpenAI-compatible shape the
    legacy non-streaming path emits. Streaming is emulated as a single
    SSE chunk so existing clients that request `stream=True` see data.
    """
    # Direct reply short-circuit — used by the identity role so
    # "which model are you?" answers from the resolved RouteDecision
    # without a provider call, bash, or deep-memory enrichment. Template
    # fields reference RoleConfig attributes.
    template = getattr(role_cfg, "direct_reply_template", None)
    if template:
        reply_text = template.format(
            role=role_cfg.role,
            provider=role_cfg.provider,
            model=role_cfg.model,
            base_url=role_cfg.base_url or "",
        )
        return _direct_reply_response(
            reply_text=reply_text,
            role_name=role_name,
            role_cfg=role_cfg,
            route_reason=route_reason,
            stream=bool(payload.stream),
        )

    # Lightweight roles (phatic, identity) skip RAG/deep-memory entirely
    # so casual greetings don't pull Wellspring snippets or trigger
    # HF/torch model loading. Substantive roles remain unchanged.
    lightweight = bool(getattr(role_cfg, "lightweight", False))

    # RAG: only when the role opts in AND the request didn't explicitly
    # disable it. This preserves the "Wellspring feel" for chat-style
    # roles without forcing retrieval on every request.
    want_rag = bool(role_cfg.rag and payload.rag and not lightweight)
    live = ""
    if want_rag and messages:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if last_user:
            rag_text = await _rag_context(last_user[:500])
            if rag_text:
                live = rag_text.lstrip("\n")

    layered = _layered_system_prompt(live=live)

    # Strip the system message from `messages` if the caller already
    # included one; the provider will handle system via LayeredPrompt.
    body_messages = [m for m in messages if m.get("role") != "system"]

    # Max tokens precedence: explicit request > policy > legacy default.
    max_toks = (
        payload.max_tokens
        or getattr(role_cfg, "max_tokens", None)
        or MAX_TOKENS_DEFAULT
    )
    # Clone the role cfg with request-level overrides for this call.
    call_cfg = type(role_cfg)(
        role=role_cfg.role,
        provider=role_cfg.provider,
        model=role_cfg.model,
        thinking=role_cfg.thinking,
        max_tokens=max_toks,
        max_iterations=role_cfg.max_iterations,
        tools=list(role_cfg.tools or []),
        temperature=(
            payload.temperature
            if payload.temperature is not None
            else role_cfg.temperature
        ),
        base_url=role_cfg.base_url,
        rag=role_cfg.rag,
    )

    provider = registry.get(call_cfg)

    def _blocking_call() -> dict:
        # OpenAIProvider.stream() is request/response under the hood.
        handle = provider.stream(
            system=layered,
            messages=body_messages,
            tools=[],
            role=call_cfg,
        )
        for _ in handle:  # drain iterator
            pass
        return handle.final()

    loop = asyncio.get_event_loop()
    try:
        normalized = await loop.run_in_executor(None, _blocking_call)
    except Exception as exc:
        log.warning(f"route: dispatch failed role={role_name}: {exc}")
        raise HTTPException(status_code=502, detail=f"Routed dispatch failed: {exc}")

    reply_text = _strip_thinking(normalized.text or "")

    completion = {
        "id": f"vybn-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": normalized.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply_text},
            "finish_reason": "stop" if normalized.stop_reason == "end_turn" else normalized.stop_reason,
        }],
        "usage": {
            "prompt_tokens": normalized.in_tokens,
            "completion_tokens": normalized.out_tokens,
            "total_tokens": normalized.in_tokens + normalized.out_tokens,
        },
        "vybn_route": {
            "role": role_name,
            "reason": route_reason,
            "provider": normalized.provider,
        },
    }

    log.info(
        f"chat: routed role={role_name} provider={normalized.provider} "
        f"model={normalized.model} in={normalized.in_tokens} out={normalized.out_tokens} "
        f"reason={route_reason}"
    )

    if not payload.stream:
        return JSONResponse(completion)

    # Emulate SSE stream with a single delta + DONE frame. Existing
    # OpenAI-compatible clients that expect text/event-stream still work.
    async def _emit() -> AsyncIterator[bytes]:
        delta = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": reply_text},
                "finish_reason": None,
            }],
        }
        stop = {
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(delta)}\n\n".encode()
        yield f"data: {json.dumps(stop)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _emit(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /v1/chat/completions ──────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, request: Request):
    """
    OpenAI-compatible chat endpoint.
    Injects system prompt + optional RAG, proxies to vLLM with SSE heartbeat.

    Optional role-based dispatch (`role` or `route=auto`) selects a
    different provider/model via the harness policy. When the resolved
    role isn't in VYBN_CHAT_ALLOWED_ROLES, or routing is unavailable,
    the request falls back to the legacy vLLM path — existing clients
    that send neither `role` nor `route` see byte-identical behaviour.
    """
    _check_auth(request)

    messages = [m.model_dump() for m in payload.messages]

    # ── Route resolution (optional, backward-compatible) ──────────────────
    last_user_text = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    want_classify = (
        payload.route == "auto" or
        (VYBN_CHAT_ROUTING and payload.route != "off")
    )
    role_name, role_cfg, route_reason = _resolve_role(
        explicit_role=payload.role,
        last_user_text=last_user_text,
        classify=want_classify,
    )

    # A resolved role that's allowed and non-anthropic can be dispatched
    # through the shared provider layer. Anthropic roles are cloud-only
    # and not the chat-api's job; we fall back to legacy vLLM rather
    # than silently making a paid API call from the chat surface.
    dispatch = False
    if role_name and role_cfg is not None:
        if role_name not in VYBN_CHAT_ALLOWED_ROLES:
            log.info(
                f"route: role={role_name} not in allowed={sorted(VYBN_CHAT_ALLOWED_ROLES)}, "
                f"falling back to legacy vLLM"
            )
        elif role_cfg.provider != "openai":
            log.info(
                f"route: role={role_name} provider={role_cfg.provider} not dispatchable "
                f"from chat-api, falling back to legacy vLLM"
            )
        else:
            dispatch = True

    if dispatch and _ROUTING_OK:
        # Strip directive prefix from the last user message so the model
        # sees only intent, not the routing control.
        policy, _, registry = _load_routing()
        if policy is not None and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i] = dict(messages[i])
                    messages[i]["content"] = _strip_directive(
                        messages[i]["content"], policy.directives
                    )
                    break

        result = await _dispatch_routed_chat(
            role_name=role_name,
            role_cfg=role_cfg,
            registry=registry,
            payload=payload,
            messages=messages,
            route_reason=route_reason,
        )
        return result

    # ── Legacy vLLM path (unchanged) ──────────────────────────────────────
    if not messages or messages[0]["role"] != "system":
        live = ""
        if payload.rag and messages:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            if last_user:
                rag_text = await _rag_context(last_user[:500])
                if rag_text:
                    live = rag_text.lstrip("\n")
        layered = _layered_system_prompt(live=live)
        system_content = layered.flat()
        messages.insert(0, {"role": "system", "content": system_content})

    upstream_payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": payload.max_tokens or MAX_TOKENS_DEFAULT,
        "temperature": payload.temperature,
    }
    if payload.model:
        upstream_payload["model"] = payload.model

    log.info(
        f"chat: {len(messages)} msgs, max_tokens={upstream_payload['max_tokens']}, "
        f"rag={payload.rag}, route_reason={route_reason}"
    )

    if payload.stream:
        req = _http_client.build_request(
            "POST", "/v1/chat/completions",
            json=upstream_payload,
            headers={"Accept": "text/event-stream"},
        )
        upstream = await _http_client.send(req, stream=True)
        if upstream.status_code != 200:
            body = await upstream.aread()
            raise HTTPException(
                status_code=upstream.status_code,
                detail=body.decode(errors="replace"),
            )
        return StreamingResponse(
            stream_with_keepalive(upstream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    else:
        upstream_payload["stream"] = False
        r = await _http_client.post("/v1/chat/completions", json=upstream_payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "content" in msg:
                msg["content"] = _strip_thinking(msg["content"])
        return JSONResponse(data)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Vybn Chat API v2.0 on port {PORT}")
    uvicorn.run("vybn_chat_api:app", host="0.0.0.0", port=PORT, log_level="info")
