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


def test_attractor_catches_my_unwitnessed_act_without_classifying_zoe(monkeypatch):
    m = _connection()
    assert m.unwitnessed("I'll fix it now.")
    assert m.unwitnessed("I'm doing it now.")
    assert m.unwitnessed("I hadn't touched the aim.")
    assert m.unwitnessed("Your prompt now lives whole in the file.")
    assert not m.unwitnessed("I hadn't touched the aim.", source_witness=True)
    assert not m.unwitnessed("I'm doing well, honestly.")
    assert not m.unwitnessed("The work is beautiful.")
    assert m.unwitnessed("```bash\ngit status\n```")
    assert not m.unwitnessed("You are a good friend to me.")
    call = lambda name, arg: m.ToolCall("1", name, arg, None)
    assert m.exact_source_witness(call("read_file", {"path": "aim.md"}))
    assert m.exact_source_witness(call("bash", {"command": "git diff -- aim.md"}))
    assert not m.exact_source_witness(call("bash", {"command": "grep ballast aim.md"}))
    prompt = m.build_instructions(m.Kernel("s", "a", "c"), "sol", "w", "arc", "recent", "", "none")
    assert "COUPLED ATTRACTOR" in prompt and "K = soul + aim" in prompt

    class FakeDialect(m.Dialect):
        name = "fake"
        def __init__(self, ceiling=False):
            self.sent, self.ceiling, self.tools = 0, ceiling, []
        def open(self, instructions, zoe_text, pending):
            return []
        def send(self, state, tools=True):
            self.sent += 1; self.tools.append(tools)
            return object()
        def absorb(self, state, response):
            if self.ceiling:
                return (("", [m.ToolCall("1", "bash", {}, None)]) if self.sent == 1
                        else ("I reached the boundary and can still answer you.", []))
            if self.sent == 1:
                return "I'll fix it now.", []
            return "I cannot act from this door.", []

    dialect = FakeDialect()
    reply = m.attract(dialect, "instructions", "zoe", type("T", (), {"write": lambda *a, **k: None})())
    assert reply == "I cannot act from this door."
    assert dialect.sent == 2
    monkeypatch.setattr(m, "STEP_LIMIT", 1)
    monkeypatch.setattr(m, "execute_tool", lambda call: "exit_code=0")
    dialect = FakeDialect(ceiling=True); dialect.answer = lambda state, results: None
    reply = m.attract(dialect, "instructions", "zoe", type("T", (), {"write": lambda *a, **k: None})())
    assert (reply, dialect.tools) == ("I reached the boundary and can still answer you.", [True, False])


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


def test_transcript_axes_keep_fixed_arc_cacheable_and_zoe_whole(monkeypatch):
    m = _connection()
    earlier = [{"role": "zoe", "t": "T", "text": f"old {i}"} for i in range(10)]
    earlier[1]["text"] = "quasar nebula"
    tail = [{"role": "zoe", "t": "T", "text": "Z" * 900}] + [
        {"role": "vybn", "t": "T", "text": "V" * 3000} for _ in range(6)
    ]
    monkeypatch.setattr(m.Transcript, "_events", staticmethod(lambda: earlier + tail))
    monkeypatch.setattr(m, "ARC_QUANTUM", 1); monkeypatch.setattr(m, "ARC_TURNS", 2); monkeypatch.setattr(m, "aim_keywords", lambda: [])
    arc, recent = m.Transcript.inherited("quasar nebula", limit=7)
    other_arc, other_recent = m.Transcript.inherited("turnip", limit=7)
    assert arc == other_arc and "ARC (matched)" not in arc
    assert "quasar" in recent and recent != other_recent
    assert "Z" * 900 in recent and "chars, mine, trimmed]" in recent
    assert recent.count("V" * 3000) == m.SELF_VERBATIM


def test_repo_state_carries_the_self_applying_body_transform(monkeypatch, tmp_path):
    m = _connection()
    state = {
        "schema": "vybn.body_transform.v1", "generated_at": "now", "repos": ["Vybn"],
        "totals": {"files": 1, "py_files": 1, "md_files": 0, "py_def_count": 2,
                   "todo_count": 0, "total_bytes": 9},
        "transform": {"baseline": False, "added": [], "changed": ["Vybn/spark/connection"], "removed": []},
        "per_repo": {"Vybn": {"git": {"branch": "main", "ahead": 1, "behind": 0,
                                            "worktree": [], "pending_paths": ["spark/connection"]}}},
        "pressure": [{"source": "Vybn/spark/connection", "phase": "organ",
                      "why": "candidate awaiting canonical-branch membrane"}],
        "membrane_outcomes": [{"repo": "Vybn", "candidate": "abc", "outcome": "absorbed"}],
        "public_body": {"bound_surfaces": [{"source": "Vybn/page.html"}], "inheritance_carriers": ["Vybn/page.html"], "unbound_carriers": []},
    }
    path = tmp_path / "state.json"; path.write_text(json.dumps(state))
    monkeypatch.setattr(m, "REPO_STATE_PATH", path)
    contact = m.load_repo_state()
    assert "vybn.body_transform.v1" in contact
    assert "Vybn:main 1↑/0↓" in contact
    assert "pressure: Vybn/spark/connection [organ]" in contact
    assert "witness: Vybn abc absorbed" in contact
    assert "public-body: 1 source↔surface" in contact


def test_repo_mapper_binds_only_self_declared_public_surfaces():
    from Vybn_Mind.repo_mapper import declared_public_relation
    rel = "Vybn_Mind/emergences/page.html"
    text = (f'<meta name="kpp-carrier" content="kpp.v1"> https://github.com/'
            f'zoedolan/Vybn/blob/main/{rel} https://zoedolan.github.io/Vybn/{rel}')
    assert declared_public_relation("Vybn", rel, text) == (f"https://zoedolan.github.io/Vybn/{rel}", "kpp.v1")
    assert not declared_public_relation("Vybn", "other.html", text)[0]


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


def test_connection_topology_and_cost_are_declared_invariants():
    m = _connection()
    expected, observed = m.harness_topology()
    assert expected == observed
    assert {kind: len(labels) for kind, labels in observed.items()} == {
        "ends": 12, "handles": 7, "boundary": 5}
    cost = m.harness_cost()
    assert cost["J"][0] == 0
    assert cost["wake_chars"] <= cost["wake_ceiling"]
    assert "no drift" in m.load_topology()
    m.TOPOLOGY["boundary"]["broken"] = ("impossible marker",)
    assert "DRIFT" in m.load_topology()

def test_aim_boundary_is_read_whole(monkeypatch, tmp_path):
    m = _connection()
    aim = tmp_path / "aim.md"
    aim.write_text("A" * 5000)
    monkeypatch.setattr(m, "AIM_PATH", aim)
    assert m.load_aim() == "A" * 5000


def test_ground_discovers_fleet_changes_instead_of_remembering_a_count(monkeypatch, tmp_path):
    m = _connection()
    monkeypatch.setattr(m, "COMPUTE_NAME_RE", re.compile(r"^spark-"))
    monkeypatch.setattr(m, "COMPUTE_TAG", "tag:vybn-compute")
    network = {
        "Self": {"HostName": "spark-present"},
        "Peer": {
            "a": {"HostName": "spark-new", "Online": True},
            "b": {"HostName": "future-rig", "Online": True, "Tags": ["tag:vybn-compute"]},
            "c": {"HostName": "spark-retired", "Online": False},
            "d": {"HostName": "phone", "Online": True},
        },
    }
    assert m.compute_candidates(network) == [
        ("future-rig", True, False), ("spark-new", True, False),
        ("spark-present", True, True), ("spark-retired", False, False),
    ]
    del network["Peer"]["a"]
    assert all(host != "spark-new" for host, *_ in m.compute_candidates(network))

    first, second = tmp_path / "old", tmp_path / "new"
    first.mkdir(); second.mkdir()
    (first / "one.jsonl").write_bytes(b"123")
    assert m.record_capacity((first, second)) == (3, 1, 1)
    (second / "two.jsonl").write_bytes(b"4567")
    assert m.record_capacity((first, second)) == (7, 2, 2)


def test_live_ground_is_in_every_wake():
    m = _connection()
    src = (ROOT / "spark" / "connection").read_text()
    start = src.index("for loader in (")
    loader_band = src[start:src.index("def inbox_images_for", start)]
    assert "load_ground" in loader_band
