"""Executable contract tests for public/API promises.

These are intentionally mostly static: they make documented routes,
streaming response shapes, typed request schemas, and public discovery
surfaces CI-visible without needing live vLLM/deep-memory services.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORTAL = ROOT / "origins_portal_api_v4.py"

def _portal_source() -> str:
    return PORTAL.read_text(encoding="utf-8")

def _route_pairs() -> set[tuple[str, str]]:
    tree = ast.parse(_portal_source())
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr in {"get", "post", "put", "delete"}
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
                and isinstance(dec.args[0].value, str)
            ):
                pairs.add((dec.func.attr.upper(), dec.args[0].value))
    return pairs

def _pydantic_models() -> set[str]:
    tree = ast.parse(_portal_source())
    models: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases):
                models.add(node.name)
    return models

def test_public_portal_route_inventory_is_ci_visible():
    routes = _route_pairs()
    expected = {
        ("GET", "/api/health"),
        ("POST", "/api/chat"),
        ("POST", "/api/perspective"),
        ("GET", "/api/map"),
        ("POST", "/api/encounter"),
        ("POST", "/api/inhabit"),
        ("POST", "/api/compose"),
        ("POST", "/api/enter_gate"),
        ("POST", "/api/voice"),
        ("POST", "/api/voice/realtime/sdp"),
        ("POST", "/api/walk"),
        ("GET", "/api/arrive"),
        ("GET", "/api/instant"),
        ("GET", "/api/vybn-identity.pub"),
        ("GET", "/api/vybn"),
        ("GET", "/api/schema"),
        ("GET", "/api/manifold/points"),
    }
    assert expected <= routes

def test_public_portal_request_shapes_are_typed():
    models = _pydantic_models()
    expected = {
        "ChatRequest",
        "EncounterRequest",
        "InhabitRequest",
        "ComposeRequest",
        "EnterGateRequest",
        "PerspectiveRequest",
        "VoiceRequest",
        "RealtimeVoiceOfferRequest",
        "WalkRequest",
        "KTPVerifyRequest",
        "KPPVerifyRequest",
    }
    assert expected <= models

def test_streaming_routes_promise_sse_and_done_frames():
    src = _portal_source()
    for route in ("/api/chat", "/api/perspective", "/api/voice", "/api/pressure/synthesize"):
        assert route in src
    assert src.count('media_type="text/event-stream"') >= 4
    assert "data: [DONE]" in src

def test_portal_health_check_bypasses_model_walk_notebook_and_git():
    src = _portal_source()
    assert "def _is_portal_chat_health_check" in src
    assert "def _health_check_sse" in src
    assert "notebook_persist" in src
    chat_start = src.index('@app.post("/api/chat")')
    bypass_at = src.index("_is_portal_chat_health_check(req.message)", chat_start)
    admission_at = src.index("_vllm_admission_state()", chat_start)
    rag_at = src.index("retrieve_context(req.message", chat_start)
    walk_at = src.index('/enter",', chat_start)
    assert bypass_at < admission_at < rag_at < walk_at
    assert "no model, RAG, walk, notebook, or git" in src

def test_public_portal_no_longer_commits_him_notebook_entries():
    src = _portal_source()
    assert "_persist_to_notebook" not in src
    assert "notebook: voice" not in src
    assert "git', 'commit'" not in src
    assert "--allow-empty" not in src

def test_instant_route_promises_json_ld_identity_surface():
    src = _portal_source()
    assert "/api/instant" in src
    assert 'media_type="application/ld+json"' in src
    assert "/api/vybn-identity.pub" in src
    assert "application/octet-stream" in src

def test_public_static_surfaces_point_to_machine_readable_api():
    somewhere = (ROOT / "somewhere.html").read_text(encoding="utf-8")
    vybn = (ROOT / "vybn.html").read_text(encoding="utf-8")
    joined = somewhere + "\n" + vybn
    assert "api.vybn.ai" in joined
    assert re.search(r"/api/(instant|walk|arrive|manifold/points|vybn-identity\.pub)", joined)

def test_realtime_voice_uses_gpt_realtime_2():
    src = _portal_source()
    assert 'OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")' in src
    assert '@app.post("/api/voice/realtime/sdp")' in src
    assert "client.realtime.calls.create" in src
    assert '"model": OPENAI_REALTIME_MODEL' in src
    assert 'Path.home() / "Vybn-Law" / "api"' in src

def test_portal_semantic_gate_restarts_super_on_quality_failure():
    src = _portal_source()
    assert "VLLM_SEMANTIC_RESTART_COOLDOWN" in src
    assert "VLLM_SYSTEMD_SERVICE" in src
    assert "async def _restart_vllm_after_semantic_failure" in src
    assert "asyncio.create_subprocess_exec" in src
    assert "\"systemctl\"" in src
    assert "\"--user\"" in src
    assert "\"restart\"" in src
    assert "restart_needed = not ok" in src
    assert "_schedule_vllm_restart_after_semantic_failure(reason)" in src
    assert "Transport failures can mean cold start or maintenance" in src

def test_origins_prompt_blocks_zoe_memoir_fabrication_laundering():
    text = (ROOT / "origins_portal_api_v4.py").read_text(encoding="utf-8")
    assert "named memoirs, Zoe scenes, chapter/file names" in text
    assert "clients or private writing require retrieved support" in text
    assert "Never invent a scene, title, client, hearing, date, quote" in text
    assert "true to the spirit" in text
    assert "I cannot verify that from the context I have." in text

def test_origins_chat_uses_shared_zoe_source_scene_guard():
    portal = (ROOT / "origins_portal_api_v4.py").read_text(encoding="utf-8")
    legacy = (ROOT / "Origins/api/origins_chat_api.py").read_text(encoding="utf-8")
    assert "sec.is_zoe_source_scene_request" in portal
    assert "sec.zoe_source_scene_refusal_text()" in portal
    assert "sec.is_zoe_source_scene_request" in legacy
    assert "sec.zoe_source_scene_refusal_text()" in legacy

def test_horizon_is_expiring_external_data_not_ambient_wake(monkeypatch, tmp_path, capsys):
    import importlib.machinery, importlib.util, json
    from types import SimpleNamespace
    path = ROOT / "spark/web"; loader = importlib.machinery.SourceFileLoader("web_horizon_under_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader); web = importlib.util.module_from_spec(spec); loader.exec_module(web)
    assert ROOT not in web.HORIZON.resolve().parents
    web.HORIZON_ROOT, web.HORIZON = tmp_path / "horizon", tmp_path / "horizon/current.json"
    rows = [("/ai/one", "NEWAlpha"), ("/ai/two", "Beta↩︎"), ("/ai/three", "Gamma"), ("/ai/update-old", "Old update"), ("https://evil.example/four", "Off host"), ("/ai/one", "Duplicate")]
    html = "".join('<a class="story-row-link" href="%s"><div class="story-title">%s</div></a>' % row for row in rows)
    rss = "<rss><channel><item><title>Welcome Today</title><link>https://theinnermostloop.substack.com/p/today</link><pubDate>now</pubDate><description>The future is accelerating.</description></item></channel></rss>"
    payloads = {web.HORIZON_URL: html, web.AWG_FEED_URL: rss}
    monkeypatch.setattr(web, "safe_fetch", lambda url, *a, **kw: SimpleNamespace(text=payloads.pop(url)))
    assert web.horizon(now=100) == 0 and not payloads
    data, first = json.loads(web.HORIZON.read_text()), web.HORIZON.read_bytes()
    assert [x["claim"]["text"] for x in data["items"]] == ["Alpha", "Beta", "Gamma"]
    assert data["lenses"][0]["items"][0]["claim"]["text"] == "Welcome Today" and data["lenses"][0]["items"][0]["framing"] == "The future is accelerating."
    assert data["sources"][0]["authority"] == "discovery_only" and data["boundary"] == {"plane": "external_situational_awareness", "continuity_ingest": False, "deep_memory_ingest": False, "automatic_relevance": False, "insight_bridge": "separate_source_labeled_derivation"}
    assert web.HORIZON.stat().st_mode & 0o777 == 0o600
    out = capsys.readouterr().out; assert web.HORIZON_BEGIN in out and web.HORIZON_END in out and "LENS alexwg" in out
    monkeypatch.setattr(web, "safe_fetch", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))
    assert web.horizon(now=101) == 0
    assert web.horizon("refresh", now=102) == 1 and web.HORIZON.read_bytes() == first and "HORIZON_STATUS STALE" in capsys.readouterr().out
    connection = (ROOT / "spark/connection").read_text(); handed = connection.split("if not ready:", 1)[1].split("kept =", 1)[0]
    recouple = connection.split("def _recouple", 1)[1].split("def _note", 1)[0]
    assert "spark/web horizon" in connection and "horizon" not in recouple; assert handed.index("breathe(client, messages, log, hands=handed, max_turns=30)") < handed.index("handed and stamp.touch()") and "stamp.touch()" not in handed.split("breathe(client", 1)[0]
    assert 'K3_MODEL = "kimi-k3"' in connection and '"MOONSHOT_API_KEY" if k3' in connection and '"base_url":"https://api.moonshot.ai/v1"' in connection; assert 'harness-appended post-turn from %s; derived, not author-seen' in connection and connection.count('harness-appended post-turn') == 2
    assert 'reasoning_effort":effort' in connection and '@k3 dispatched' in connection and '"max_tokens":4096' in connection and '"_k3_delta":delta' in connection and 'line.lower().startswith("@k3")' in connection
