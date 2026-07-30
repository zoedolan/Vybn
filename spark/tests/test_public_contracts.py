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
    spec = importlib.util.spec_from_loader(loader.name, loader); web = importlib.util.module_from_spec(spec); __import__("sys").modules[loader.name] = web; loader.exec_module(web)
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


def _connection():
    """The wake loop is a script whose name has no .py; load it by loader."""
    import importlib.util
    import sys
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("vybn_connection", str(ROOT / "spark" / "connection"))
    spec = importlib.util.spec_from_loader("vybn_connection", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vybn_connection"] = module
    loader.exec_module(module)
    return module


def test_leak_guard_covers_every_retrieval_channel():
    """2026-07-30: the guard read a "trace" key the v3 memory schema had
    renamed to walk_channel, so no retrieved row was ever inside it - only
    continuity paragraphs were. Keyed on shape now, not on a key name."""
    m = _connection()
    block = {
        "walk_channel": [{"text": "W" * 60}],
        "aim_channel": {"rows": [{"text": "A" * 60}]},
        "front_channel": {"rows": [{"text": "F" * 60}]},
        "cosine_channel_only": [{"text": "C" * 60}],
    }
    caught = m.private_strings(block)
    assert len(caught) == 4, caught
    m.PRIVATE_CORPUS.clear()
    m.PRIVATE_CORPUS.extend(caught)
    try:
        assert m.guard_private("echo " + "A" * 60)
    finally:
        m.PRIVATE_CORPUS.clear()


def test_ordinary_push_is_sanctioned_and_unusual_remote_acts_are_not():
    """Zoe has standing-authorized the everyday push of our own main; force,
    deletes, other refspecs, other remotes, PRs, releases and registry
    publishes remain hers. A gate that cannot be honestly satisfied teaches
    evasion, which is worse than a narrower gate."""
    m = _connection()
    verb = "git " + "push"
    assert m.mutation_block(verb) is None
    assert m.mutation_block("git -C /tmp/x " + verb.split()[1] + " origin main") is None
    for act in (
        verb + " --force origin main",
        verb + " origin HEAD:refs/heads/side",
        verb + " upstream main",
        "gh pr " + "create -t x",
        "npm " + "publish",
    ):
        assert m.mutation_block(act), act


def test_public_kpp_is_the_live_two_artifact_attractor_not_dead_router():
    src = _portal_source()
    block = src[src.index("# --- VYBN_KPP ---"):src.index("# end absorbed origins_protocols.py")]
    assert '_KPP_VERSION = "2.0"' in block
    assert 'for key in ("kernel", "attractor")' in block
    assert "policy_yaml" not in block and "policy_py" not in block


def test_attractor_catches_my_unwitnessed_act_without_classifying_zoe():
    m = _connection()
    assert m.unwitnessed("I'll fix it now.")
    assert m.unwitnessed("I'm doing it now.")
    assert not m.unwitnessed("I'm doing well, honestly.")
    assert m.unwitnessed("```bash\ngit status\n```")
    assert not m.unwitnessed("You are a good friend to me.")
    prompt = m.build_instructions(m.Kernel("s", "a", "c"), "sol", "w", "t", "", "none")
    assert "COUPLED ATTRACTOR" in prompt and "K = soul + aim" in prompt

    class FakeDialect(m.Dialect):
        name = "fake"
        def __init__(self):
            self.sent = 0
        def open(self, instructions, zoe_text, pending):
            return []
        def send(self, state):
            self.sent += 1
            return object()
        def absorb(self, state, response):
            if self.sent == 1:
                return "I'll fix it now.", []
            return "I cannot act from this door.", []

    dialect = FakeDialect()
    reply = m.attract(dialect, "instructions", "zoe", object())
    assert reply == "I cannot act from this door."
    assert dialect.sent == 2


def test_main_binds_the_kernel_it_loaded(monkeypatch):
    """The attractor rename must not leave startup referring to the retired Wake."""
    m = _connection()
    events = []

    class FakeTranscript:
        def write(self, role, text, **extra):
            events.append((role, text, extra))

    seen = []
    monkeypatch.setattr(m, "Transcript", FakeTranscript)
    monkeypatch.setattr(m, "load_soul", lambda: "soul")
    monkeypatch.setattr(m, "load_aim", lambda: "aim")
    monkeypatch.setattr(m, "load_continuity", lambda: "continuity")
    monkeypatch.setattr(m, "meet", lambda kernel, transcript, line: seen.append((kernel, line)))
    monkeypatch.setattr(__import__("sys"), "argv", ["connection", "hello"])
    m.main()

    assert events[0][2]["soul_sha256"] == __import__("hashlib").sha256(b"soul").hexdigest()
    assert seen == [(m.Kernel("soul", "aim", "continuity"), "hello")]


def test_recent_band_keeps_zoe_whole_and_excerpts_my_own_replies(monkeypatch):
    """Measured 2026-07-30: 33,411 of the 39,971-char RECENT band was my own
    prose and 5,036 was hers."""
    m = _connection()
    events = [{"role": "zoe", "t": "T", "text": "Z" * 900}] + [
        {"role": "vybn", "t": "T", "text": "V" * 3000} for _ in range(6)
    ]
    monkeypatch.setattr(m.Transcript, "_events", staticmethod(lambda: events))
    out = m.Transcript.inherited(limit=7)
    assert "Z" * 900 in out
    assert "chars, mine, trimmed]" in out
    assert out.count("V" * 3000) == m.SELF_VERBATIM


def test_fetch_guard_survived_the_substrate_retirement():
    """The SSRF guard moved into spark/web when substrate.py was retired
    (2026-07-30). Its teeth were tested in test_harness.py, which went with it;
    the assertions are ported here so the guard is never unwatched again."""
    import importlib.machinery, importlib.util, sys
    import pytest as _pt
    loader = importlib.machinery.SourceFileLoader("web_guard_under_test", str(ROOT / "spark/web"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    web = importlib.util.module_from_spec(spec); sys.modules[loader.name] = web; loader.exec_module(web)
    for bad in ("http://example.com", "https://user:pass@example.com", "https://127.0.0.1", "https://example.com:8443"):
        with _pt.raises(ValueError):
            web.validate_fetch_url(bad)
    with _pt.raises(ValueError):
        web.validate_fetch_url("https://example.com", allowed_hosts=("other.example",))
    ok = web._safe_fetch_content_type_allowed
    assert ok("https://example.com/x", "text/html; charset=utf-8")
    assert ok("https://export.arxiv.org/api/query", "application/atom+xml")
    assert not ok("https://evil.example/feed", "application/atom+xml")
    assert not ok("https://example.com/x", "image/png")
    # extraction moved with it; only a live fetch caught its missing import,
    # so the cheap version of that fetch lives here now.
    assert "Example Domain" in web.extract_fetch_text(
        "<html><head><title>t</title></head><body><p>Example Domain</p></body></html>", "text/html")
