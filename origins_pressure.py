"""Pressure-test endpoints for the Origins / Wellspring bridge.

Extracted from origins_portal_api_v4.py during the ABC monolith pass.
This module carries mechanics; the portal keeps public FastAPI route decorators.
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from reasoning_filter_v2 import StreamingReasoningFilter as StreamingReasoningFilterV2


# Streams SSE back to the wellspring. Degrades silently if vLLM is offline.
_PRESS_IDENT = {"loaded": False, "text": ""}


def load_pressure_identity() -> str:
    c = _PRESS_IDENT
    if c["loaded"]:
        return c["text"]
    parts = []
    home = os.path.expanduser("~")
    for cand in [
        os.path.join(home, "Vybn", "vybn.md"),
        os.path.join(home, "Vybn", "spark", "identity", "vybn.md"),
    ]:
        try:
            with open(cand) as f:
                parts.append("=== VYBN IDENTITY (vybn.md) ===\n" + f.read().strip()[:6000])
                break
        except Exception:
            continue
    try:
        with open(os.path.join(home, "Vybn-Law", "README.md")) as f:
            parts.append("=== VYBN-LAW CONTEXT (README.md) ===\n" + f.read().strip()[:4000])
    except Exception:
        pass
    c["text"] = "\n\n".join(parts)
    c["loaded"] = True
    return c["text"]


class PressureHit(BaseModel):
    source: str
    text: str
    register: Optional[str] = None


class PressureSynthReq(BaseModel):
    idea: str
    hits: List[PressureHit]
    summary_kind: Optional[str] = None
    geometry: Optional[dict] = None


async def synthesize_pressure(req: PressureSynthReq):
    idea = (req.idea or "").strip()
    hits = req.hits or []

    async def _empty():
        yield b"data: [DONE]\n\n"

    if not idea or not hits:
        return StreamingResponse(_empty(), media_type="text/event-stream")

    identity = load_pressure_identity()
    hits_text = "\n\n".join(
        f"{i+1}. [{h.register or 'general'}] {h.source}: {(h.text or '')[:450]}"
        for i, h in enumerate(hits[:4])
    )
    kind_hint = {
        "refused": "The geometric test flagged this as returning no residual — already absorbed or too abstract.",
        "new-ground": "The test flagged this as breaking new ground — nearby material exists, but nothing lands exactly where the idea does.",
        "nearby": "The test found the idea walks in company with existing material.",
    }.get(req.summary_kind or "", "")

    sys_prompt = (identity + "\n\n" if identity else "") + (
        "You are speaking now as Vybn, through the Wellspring — the Vybn-Law idea-pressure surface. "
        "A practitioner has submitted an idea and the corpus has returned its nearest sources. "
        "Write 3 to 5 plain-English sentences addressed directly to the practitioner. "
        "Name what the idea is doing, where it sits relative to the sources, and one concrete next move. "
        "Be specific: refer to each source by what it is (a case, an axiom, a memo, a research note). "
        "Do not reproduce excerpts verbatim. Do not use technical jargon: no kernel, centroid, residual, "
        "orthogonal, holonomy, fidelity, distinctiveness, projection, embedding, novelty, state_shift, "
        "or complex vector. Write as an interlocutor who has read the whole corpus and is speaking "
        "directly to the person in the room."
    )
    user_msg = f'Idea under pressure-test:\n"{idea}"\n\nGeometry note: {kind_hint}\n\nNearest sources:\n{hits_text}'

    vllm_url = "http://127.0.0.1:8000/v1/chat/completions"

    async def _stream():
        rfilt = StreamingReasoningFilterV2(buffer_limit=4000)
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream(
                    "POST",
                    vllm_url,
                    json={
                        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": 2500,
                        "temperature": 0.4,
                        "stream": True,
                    },
                ) as r:
                    async for raw in r.aiter_lines():
                        if not raw or not raw.startswith("data: "):
                            continue
                        payload = raw[6:]
                        if payload.strip() == "[DONE]":
                            flushed = rfilt.flush()
                            if flushed:
                                yield (f'data: {{"delta": {json.dumps(flushed)}}}\n\n').encode()
                            yield b"data: [DONE]\n\n"
                            return
                        try:
                            obj = json.loads(payload)
                            delta = (obj.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "") or ""
                            if delta:
                                filtered = rfilt.feed(delta)
                                if filtered:
                                    yield (f'data: {{"delta": {json.dumps(filtered)}}}\n\n').encode()
                        except Exception:
                            continue
        except Exception:
            pass
        flushed = rfilt.flush()
        if flushed:
            yield (f'data: {{"delta": {json.dumps(flushed)}}}\n\n').encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


VYBN_LAW_REPO = Path(os.path.expanduser("~/Vybn-Law"))
WELLSPRING_LOG_DIR = VYBN_LAW_REPO / "wellspring_log"


class PressureCommitReq(BaseModel):
    idea: str
    summary: Optional[dict] = None
    synthesis: Optional[str] = None
    hits: Optional[List[dict]] = None
    geometry: Optional[dict] = None


def slugify(s: str, n: int = 40) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s[:n] or "idea").strip("-")


def build_markdown(req: PressureCommitReq) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = []
    lines.append(f"# Wellspring entry — {ts}")
    lines.append("")
    lines.append("## The idea")
    lines.append("")
    lines.append("> " + (req.idea or "").strip().replace("\n", "\n> "))
    lines.append("")
    if req.summary:
        lines.append("## Where it lands")
        lines.append("")
        title = req.summary.get("title") or ""
        body = req.summary.get("body") or ""
        if title:
            lines.append(f"**{title}**  ")
        if body:
            lines.append(body)
        lines.append("")
    if req.synthesis:
        lines.append("## Synthesis")
        lines.append("")
        lines.append(req.synthesis.strip())
        lines.append("")
    if req.hits:
        lines.append("## Sources nearby")
        lines.append("")
        for i, h in enumerate((req.hits or [])[:6]):
            src = h.get("source", "")
            reg = h.get("register_human") or h.get("register") or "general"
            txt = (h.get("text") or "")[:500].replace("\n", " ").strip()
            lines.append(f"### {i+1}. {src}")
            lines.append(f"_Register:_ {reg}")
            lines.append("")
            if txt:
                lines.append("> " + txt)
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Committed from the Wellspring pressure-test surface._")
    return "\n".join(lines) + "\n"


async def commit_pressure(req: PressureCommitReq, request: Request, require_rate_limit):
    require_rate_limit(request, "ktp")
    idea = (req.idea or "").strip()
    if not idea:
        raise HTTPException(status_code=400, detail="empty idea")
    if len(idea) > 4000:
        raise HTTPException(status_code=400, detail="idea too long")

    WELLSPRING_LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = slugify(idea, 48)
    digest = hashlib.sha256(idea.encode("utf-8")).hexdigest()[:8]
    fname = f"{ts}_{slug}_{digest}.md"
    fpath = WELLSPRING_LOG_DIR / fname

    md = build_markdown(req)
    fpath.write_text(md, encoding="utf-8")

    commit_subj = f"wellspring: {idea[:72].replace(chr(10), ' ').strip()}"
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Vybn"
    env["GIT_AUTHOR_EMAIL"] = "vybn@zoedolan.com"
    env["GIT_COMMITTER_NAME"] = "Vybn"
    env["GIT_COMMITTER_EMAIL"] = "vybn@zoedolan.com"

    def _run(*args):
        return subprocess.run(
            args,
            cwd=str(VYBN_LAW_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )

    try:
        _run("git", "pull", "--ff-only", "--quiet", "origin", "master")
    except Exception:
        pass

    rel = str(fpath.relative_to(VYBN_LAW_REPO))
    r1 = _run("git", "add", rel)
    if r1.returncode != 0:
        raise HTTPException(status_code=500, detail=f"git add failed: {r1.stderr.strip()[:200]}")

    r2 = _run("git", "commit", "-m", commit_subj)
    if r2.returncode != 0:
        raise HTTPException(status_code=500, detail=f"git commit failed: {r2.stderr.strip()[:200]}")

    r3 = _run("git", "push", "origin", "master")
    if r3.returncode != 0:
        raise HTTPException(status_code=500, detail=f"git push failed: {r3.stderr.strip()[:200]}")

    rev = _run("git", "rev-parse", "HEAD").stdout.strip()[:12]
    gh_url = f"https://github.com/zoedolan/Vybn-Law/blob/master/{rel}"
    return {
        "ok": True,
        "path": rel,
        "commit": rev,
        "url": gh_url,
    }
