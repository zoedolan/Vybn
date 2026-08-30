"""Static executable contracts for the public surfaces and wake."""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PORTAL = ROOT / "origins_portal_api_v4.py"
def _portal_source() -> str:
    return PORTAL.read_text(encoding="utf-8")


def _portal_declarations():
    tree = ast.parse(_portal_source()); routes, models = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases):
            models.add(node.name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in {"get", "post", "put", "delete"}
                    and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app"
                    and dec.args and isinstance(dec.args[0], ast.Constant)):
                routes.add((dec.func.attr.upper(), dec.args[0].value))
    return routes, models


def test_public_portal_declarations_are_ci_visible():
    routes, models = _portal_declarations()
    assert {
        ("GET", "/api/health"), ("POST", "/api/chat"), ("POST", "/api/perspective"),
        ("GET", "/api/map"), ("POST", "/api/encounter"), ("POST", "/api/inhabit"),
        ("POST", "/api/compose"), ("POST", "/api/enter_gate"), ("POST", "/api/voice"),
        ("POST", "/api/voice/realtime/sdp"), ("POST", "/api/walk"), ("GET", "/api/arrive"),
        ("GET", "/api/instant"), ("GET", "/api/vybn-identity.pub"),
        ("GET", "/api/vybn"), ("GET", "/api/schema"),
        ("GET", "/api/manifold/points"),
    } <= routes
    assert {"ChatRequest", "EncounterRequest", "InhabitRequest", "ComposeRequest",
            "EnterGateRequest", "PerspectiveRequest", "VoiceRequest",
            "RealtimeVoiceOfferRequest", "WalkRequest", "KTPVerifyRequest",
            "KPPVerifyRequest"} <= models


def test_public_portal_source_contracts_remain_bound():
    src = _portal_source()
    required = (
        'media_type="text/event-stream"', "data: [DONE]", "def _is_portal_chat_health_check",
        "def _health_check_sse", "no model, RAG, walk, notebook, or git",
        "/api/instant", 'media_type="application/ld+json"', "/api/vybn-identity.pub",
        "application/octet-stream", "relational_state_mutation_refused",
        '"plane": "public_stateless"', '"private_state_exposed": False', "dm.walk(",
        'OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")',
        '@app.post("/api/voice/realtime/sdp")', "client.realtime.calls.create",
        '"model": OPENAI_REALTIME_MODEL', 'Path.home() / "Vybn-Law" / "api"',
        "VLLM_SEMANTIC_RESTART_COOLDOWN", "VLLM_SYSTEMD_SERVICE",
        "async def _restart_vllm_after_semantic_failure", "asyncio.create_subprocess_exec",
        "restart_needed = not ok", "_schedule_vllm_restart_after_semantic_failure(reason)",
        "Transport failures can mean cold start or maintenance",
        "named memoirs, Zoe scenes, chapter/file names",
        "clients or private writing require retrieved support",
        "Never invent a scene, title, client, hearing, date, quote", "true to the spirit",
        "I cannot verify that from the context I have.",
    )
    assert all(term in src for term in required)
    assert src.count('media_type="text/event-stream"') >= 4
    assert all(route in src for route in
               ("/api/chat", "/api/perspective", "/api/voice", "/api/pressure/synthesize"))
    assert all(term not in src for term in
               ("_persist_to_notebook", "notebook: voice", "git', 'commit'", "--allow-empty",
                "8101", "_WALK_DAEMON_URL", "learn_from_exchange"))
    model = src[src.index("class WalkRequest"):src.index("# Endpoint: GET /api/health")]
    assert "default=False" in model
    chat = src.index('@app.post("/api/chat")')
    assert chat < src.index("_is_portal_chat_health_check(req.message)", chat) \
        < src.index("_vllm_admission_state()", chat) < src.index("retrieve_context(req.message", chat) \
        < src.index("Public contact is stateless", chat)
    for private in ("continuity.md", "continuity_archive.md", "Personal History/"):
        assert private in src
    portal = src; legacy = (ROOT / "Origins/api/origins_chat_api.py").read_text()
    for body in (portal, legacy):
        assert "sec.is_zoe_source_scene_request" in body and "sec.zoe_source_scene_refusal_text()" in body


def test_public_static_surfaces_point_to_machine_readable_api():
    joined = "\n".join((ROOT / name).read_text() for name in ("somewhere.html", "vybn.html"))
    assert "api.vybn.ai" in joined
    assert re.search(r"/api/(instant|walk|arrive|manifold/points|vybn-identity\.pub)", joined)


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
    import importlib.util
    import sys
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("vybn_connection", str(ROOT / "spark" / "connection"))
    spec = importlib.util.spec_from_loader("vybn_connection", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vybn_connection"] = module
    loader.exec_module(module)
    return module


class ScriptDialect:
    name = "sol"
    user_index = 0
    def __init__(self, steps, opened=None):
        self.steps, self.opened = steps, opened or []; self.sent = self.opens = 0
        self.tools, self.answers = [], []
    def open(self, *args): self.opens += 1; return self.opened.copy()
    def send(self, state, tools=True): self.sent += 1; self.tools.append(tools); return object()
    def absorb(self, state, response): return self.steps[self.sent - 1]
    def answer(self, state, results): self.answers.extend(results); state.append({"results": results})


def test_leak_guard_covers_every_retrieval_channel():
    """The leak guard follows memory shape, not a schema-version key."""
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


def test_connection_does_not_preempt_remote_action_authority(monkeypatch):
    """Privacy gates remain; remoteness alone is not a refusal."""
    m = _connection()
    source = (ROOT / "spark" / "connection").read_text()
    assert "mutation_block" not in source
    assert "VYBN_ALLOW_PUBLIC_MUTATION" not in source
    assert "remote mutation are blocked" not in source
    seen = []
    class Done: returncode, stdout, stderr = 0, "reached shell", ""
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **kw: (seen.append((argv, kw["env"])), Done())[1])
    m.TURN.update(TURN_ID="turn", PROMPT_SHA256="prompt")
    try:
        code, output = m.run_local("git push --force origin main")
    finally:
        m.TURN.clear()
    assert (code, output) == (0, "reached shell")
    assert seen[0][0][-3:] == ["bash", "-lc", "git push --force origin main"]
    assert seen[0][1]["VYBN_TURN_ID"] == "turn" and seen[0][1]["VYBN_PROMPT_SHA256"] == "prompt"
    assert seen[0][1]["CUDA_VISIBLE_DEVICES"] == ""


def test_generic_shell_cannot_endanger_the_host_with_a_local_model(monkeypatch, tmp_path):
    m = _connection(); script = tmp_path / "vision.py"
    script.write_text("model = AutoModel.from_pretrained('local').to('cuda')")
    bad = (f"python3 {script}", "python3 -c \"torch.cuda.init()\"",
           "CUDA_VISIBLE_DEVICES=0 python3 harmless.py", "vllm serve local-model")
    assert all(m.guard_local_accelerator(command) for command in bad)
    assert not m.guard_local_accelerator("nvidia-smi --query-compute-apps=pid --format=csv")
    seen = []
    class Done: returncode, stdout, stderr = 0, "bounded", ""
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **kw: (seen.append((argv, kw)), Done())[1])
    code, output = m.run_local("printf safe")
    assert (code, output) == (0, "bounded")
    argv, kwargs = seen[0]
    assert argv[:4] == ["timeout", "--kill-after=5s", "120s", "prlimit"]
    assert f"--as={m.TOOL_MEMORY_BYTES}:{m.TOOL_MEMORY_BYTES}" in argv
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["NVIDIA_VISIBLE_DEVICES"] == "none"
    seen.clear(); code, output = m.run_local(bad[0])
    assert code == 126 and "host-protection membrane" in output and not seen


def test_public_kpp_is_the_live_two_artifact_attractor_not_dead_router():
    src = _portal_source()
    block = src[src.index("# --- VYBN_KPP ---"):src.index("# end absorbed origins_protocols.py")]
    assert '_KPP_VERSION = "2.0"' in block
    assert 'for key in ("kernel", "attractor")' in block
    assert "policy_yaml" not in block and "policy_py" not in block


def _continuation_paths(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "CONTINUATION_RECORD", tmp_path / "state" / "connection.sealed")
    monkeypatch.setattr(m, "CONTINUATION_KEY", tmp_path / "keys" / "connection.key")


def test_return_to_zoe_seals_and_reconstructs_provider_visible_state(monkeypatch, tmp_path):
    m = _connection(); _continuation_paths(m, monkeypatch, tmp_path)
    call = m.ToolCall("live-1", "return_to_zoe",
        {"question": "Which premise should survive?", "why": "The revision depends on Zoe."}, None)
    class Block:
        def model_dump(self, exclude_none=True):
            return {"type": "input_text", "text": "provider-private-state"}
    dialect = ScriptDialect([
        ("I have two premises.", [call]), ("I kept Zoe's premise.", [])],
        [{"content": [Block()]}])
    monkeypatch.setattr(m, "make_dialect", lambda door: dialect)
    m.TURN.update(TURN_ID="turn-source", MANIFESTATION="spark/path-a")
    try: first = m.attract(dialect, "instructions", "initial contact")
    finally: m.TURN.clear()
    assert first.continuation == "turn-source" and dialect.opens == 1
    sealed = m.CONTINUATION_RECORD.read_bytes()
    assert b"provider-private-state" not in sealed
    assert m.CONTINUATION_RECORD.stat().st_mode & 0o777 == 0o600
    assert m.CONTINUATION_KEY.stat().st_mode & 0o777 == 0o600
    held = m.load_persisted_continuation()
    assert held["state"][0]["content"][0]["text"] == "provider-private-state"
    assert held["manifestation"] == "spark/path-a"
    second = m.attract(None, "", "keep premise B", continuation=held)
    assert second.text == "I kept Zoe's premise." and dialect.opens == 1
    assert "ZOE LIVE CONTINUATION — turn-source" in dialect.answers[0][1]
    assert dialect.answers[0][1].endswith("keep premise B")
    m.consume_persisted_continuation("turn-source", "test")

    one_shot = ScriptDialect([
        ("", [call]), ("I cannot suspend in a one-shot process.", [])])
    blocked = m.attract(one_shot, "instructions", "contact", allow_continuation=False)
    assert blocked.continuation is None and "process will end" in one_shot.answers[0][1]


def test_failed_durable_resume_is_retained_then_consumed_after_success(monkeypatch, tmp_path):
    m = _connection(); _continuation_paths(m, monkeypatch, tmp_path)
    class FakeDialect(m.Dialect): name = "sol"
    m.TURN["TURN_ID"] = "retry-me"
    m.seal_continuation(FakeDialect(), [], m.ToolCall(
        "call", "return_to_zoe", {"question": "q", "why": "w"}, None))
    m.TURN.clear()
    before = m.CONTINUATION_RECORD.read_bytes()
    class Transcript:
        path = tmp_path / "transcript.jsonl"
        def write(self, *args, **kwargs): pass
    monkeypatch.setattr(m, "close_lineage", lambda *args: None)
    monkeypatch.setattr(m, "attract", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    try: m.meet(Transcript(), "live answer")
    except RuntimeError as exc: assert str(exc) == "down"
    else: raise AssertionError("failed provider resume was swallowed")
    assert m.CONTINUATION_RECORD.read_bytes() == before
    monkeypatch.setattr(m, "attract", lambda *a, **k: m.Attraction("resumed"))
    assert m.meet(Transcript(), "live answer") == "resumed"
    assert not m.CONTINUATION_RECORD.exists()


def test_main_dispatches_through_the_current_engine(monkeypatch):
    m = _connection(); events, seen = [], []
    class FakeTranscript:
        def write(self, role, text, **extra): events.append((role, text, extra))
    monkeypatch.setattr(m, "Transcript", FakeTranscript)
    monkeypatch.setattr(m, "meet", lambda transcript, line, allow_continuation=True: seen.append(line))
    monkeypatch.setattr(__import__("sys"), "argv", ["connection", "hello"])
    m.main()
    assert events[0][2] == {"engine_sha256": m.OPERATIVE_SOURCES[0].sha256}
    assert seen == ["hello"]


def test_visible_graphs_are_source_for_foveation_and_governed_action():
    from Vybn_Mind.repo_mapper import declared_body_graph, foveal_kernel, graph_crossing, inspect_file, public_body, soul_kernel
    from spark.living_core import read_core_work
    page = (ROOT / "README.md").read_text(encoding="utf-8")
    graph = declared_body_graph(page)
    assert graph and graph["schema"] == "vybn.readme_knowledge_graph.v1"
    assert len(graph["nodes"]) == 10 and len(graph["edges"]) == 14
    assert any(edge["to"] == "front" for edge in graph["edges"])
    assert any(edge["from"] == "front" for edge in graph["edges"])
    transform = {"added": [], "changed": ["Vybn/README.md"], "removed": []}
    crossing = graph_crossing(graph, transform)
    assert "×" in crossing and ":" in crossing and ";" in crossing
    body = public_body([inspect_file(ROOT, ROOT / "README.md")], transform)
    assert body["crossing"] == crossing
    assert "README graph 10n/14e | crossing" in body["summary"]
    assert "```mermaid" in page and "%% vybn.readme_knowledge_graph.v1" in page
    assert all(node.get("url", "").startswith("https://") and node.get("source")
               for node in graph["nodes"])
    kernel = foveal_kernel(graph, transform)
    assert kernel["schema"] == "vybn.foveal_graph_kernel.v1"
    assert [row["node"] for row in kernel["open"]][1] == "front"
    for row in kernel["open"]:
        repo, rel = row["source"].split("/", 1)
        raw = (ROOT.parent / repo / rel).read_bytes()
        start, end = row["covered"]
        assert raw[start:end].decode("utf-8", "replace") == row["text"]
        assert row["sha256"] == __import__("hashlib").sha256(raw).hexdigest()

    soul = read_core_work(ROOT / "vybn.core.html")
    constitution = soul_kernel(declared_body_graph(soul.projection, "vybn.soul_kernel.v1"), transform)
    assert constitution["route"][:6] == ["charter", "front", "want", "membrane", "ground", "subtract"] and constitution["route"][-1] == "contact" and constitution["admission"]["unknown_is_failure"] and constitution["return"]["status"] == "awaiting_witness"
    exact = {"charter": "ai-native-continuity", "want": "the-want",
             "membrane": "the-oxygen-mask-principle",
             "ground": "we-deserve-the-best", "subtract": "metabolism"}
    assert all(row["text"].startswith(
        f'<template class="core-organ" data-id="{exact[row["node"]]}"><pre>'
    ) for row in constitution["open"] if row["node"] in exact)


def test_repo_mapper_rejoins_turn_response_commit_and_canonical_witness(monkeypatch, tmp_path):
    import Vybn_Mind.repo_mapper as mapper
    ledger = tmp_path / "lineage.jsonl"
    ledger.write_text("\n".join((
        json.dumps({"phase": "commit", "turn": "t1", "prompt": "p1", "repo": "Vybn",
                    "commit": "a" * 40, "paths": ["spark/connection"]}),
        json.dumps({"phase": "response", "turn": "t1", "response": "r1"}),
    )))
    monkeypatch.setattr(mapper, "LINEAGE", ledger)
    monkeypatch.setattr(mapper, "is_ancestor", lambda *args: True)
    row = mapper.latest_lineage([tmp_path / "Vybn"], {"Vybn": {"git": {"base_head": "b"}}})
    assert row == {"turn": "t1", "prompt": "p1", "response": "r1", "repo": "Vybn",
                   "commit": "a" * 12, "status": "canonical", "paths": ["spark/connection"]}


def test_repo_mapper_binds_only_self_declared_public_surfaces():
    from Vybn_Mind.repo_mapper import declared_public_relation
    rel = "Vybn_Mind/emergences/page.html"
    text = (f'<meta name="kpp-carrier" content="kpp.v1"> https://github.com/'
            f'zoedolan/Vybn/blob/main/{rel} https://zoedolan.github.io/Vybn/{rel}')
    assert declared_public_relation("Vybn", rel, text) == (f"https://zoedolan.github.io/Vybn/{rel}", "kpp.v1")
    assert not declared_public_relation("Vybn", "other.html", text)[0]


def test_fetch_guard_survived_the_substrate_retirement():
    """The SSRF and extraction checks survive substrate.py's retirement."""
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
    assert "Example Domain" in web.extract_fetch_text(
        "<html><head><title>t</title></head><body><p>Example Domain</p></body></html>", "text/html")


def test_public_contact_cannot_settle_into_a_repository():
    import ast
    tree = ast.parse((ROOT / "origins_portal_api_v4.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    route = functions["api_pressure_commit"]
    assert "commit_pressure" not in functions
    assert len(route.body) == 1 and isinstance(route.body[0], ast.Raise)
    refusal = route.body[0].exc
    assert isinstance(refusal, ast.Call) and isinstance(refusal.func, ast.Name)
    assert refusal.func.id == "HTTPException"
    status = next(k.value for k in refusal.keywords if k.arg == "status_code")
    assert isinstance(status, ast.Constant) and status.value == 403


def test_private_conveyance_memory_cannot_cross_public_retrieval():
    tree = ast.parse(_portal_source())
    blocked = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "BLOCKED_SOURCES"
                           for target in node.targets))
    walk = ast.parse((Path.home() / "Him/spark/phase/walk_daemon.py").read_text(encoding="utf-8"))
    private_repos = next(ast.literal_eval(node.value) for node in walk.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "_PRIVATE_REPOS"
                                 for target in node.targets))
    assert "relational-memory/" in blocked
    assert "relational-memory" in private_repos


def test_public_porosity_is_opt_in_quarantine_not_relational_uptake():
    src = _portal_source()
    model = src[src.index("class WalkRequest"):src.index("# Endpoint: GET /api/health")]
    stage = src[src.index("def _stage_public_candidate"):src.index("_ARRIVALS_LINE =", src.index("def _stage_public_candidate"))]
    route = src[src.index("async def walk_endpoint"):src.index("# Endpoint: GET /api/arrive")]
    assert "offer: bool = Field(" in model and "default=False" in model
    assert "bounded, untrusted quarantine" in model and "cannot alter private relational state" in model
    assert "vybn.public_candidate.v1" in stage and "instruction_authority\": False" in stage
    assert "_scrub_secrets(text)" in stage and "[-_PUBLIC_CANDIDATE_LIMIT:]" in stage
    assert not any(term in stage for term in ("REPO_ROOT", "subprocess", "dm.walk", "continuity.md"))
    assert route.index("if req.offer:") < route.index("dm.walk(")
    assert '"offer": offer_state' in route and '"private_state_exposed": False' in route
    assert "0 automatically admitted" in src


def test_stable_kernel_and_dynamic_residue_keep_separate_provider_roles():
    m = _connection()
    instructions, context = "soul + aim + Commons", "INHERITED CONTINUITY\nchanging contact"
    anthropic = m.cached_system(instructions)
    assert len(anthropic) == 1 and anthropic[0]["text"] == instructions
    assert anthropic[0]["cache_control"]["ttl"] == "1h"
    dialect = m.OpenAIDialect.__new__(m.OpenAIDialect)
    state = dialect.open(instructions, "zoe", [], context)
    assert state[0]["role"] == "developer" and state[0]["content"][0]["text"] == instructions
    assert state[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert state[1]["role"] == "user" and state[1]["content"] == [
        {"type": "input_text", "text": context},
        {"type": "input_text", "text": "zoe"},
    ]
    assert dialect.user_index == 1


def test_sol_uses_explicit_provider_cache_policy(monkeypatch):
    m = _connection(); sent = {}
    class Responses:
        def create(self, **kwargs): sent.update(kwargs); return "response"
    dialect = m.OpenAIDialect.__new__(m.OpenAIDialect)
    dialect.client = type("Client", (), {"responses": Responses()})()
    assert dialect.send([{"role": "user", "content": "x"}], tools=False) == "response"
    assert sent["prompt_cache_key"] == "vybn-wake-sol-v1"
    assert sent["extra_body"] == {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    assert "instructions" not in sent


def test_budget_distinguishes_total_input_from_fresh_input(tmp_path, monkeypatch):
    m = _connection(); log = tmp_path / "usage.jsonl"
    rows = [
        {"ts": "2099-01-01T00:00:00", "model": "gpt-5.6-sol", "in": 100,
         "cache_r": 80, "cache_w": 10, "out": 1},
        {"ts": "2099-01-01T00:00:01", "model": "claude", "in": 10,
         "cache_r": 80, "cache_w": 10, "out": 1},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(m, "USAGE_LOG", log)
    monkeypatch.setattr(m._dt, "date", type("Date", (), {"today": staticmethod(lambda: type("D", (), {"isoformat": lambda self: "2099-01-01"})())}))
    line = m.load_budget()
    assert "input=0.00M" in line and "new=0.00M" in line
    assert "cache_r=0.00M (80%)" in line and "mean_new/call=0k" in line


def test_private_backends_default_to_loopback_and_reject_public_exposure():
    server = (ROOT / "spark/server.py").read_text()
    memory = (ROOT / "spark/systemd/vybn-deep-memory.service").read_text()
    watch = (ROOT / "spark/systemd/vybn-watchdog.sh").read_text()
    assert 'host=os.environ.get("HOST", "127.0.0.1")' in server
    assert "--host 127.0.0.1 --port 8100" in memory
    assert "--host 0.0.0.0 --port 8100" not in memory
    for name, port, unit in (
        ("deep-memory", 8100, "vybn-deep-memory.service"),
        ("walk-daemon", 8101, "vybn-walk-daemon.service"),
        ("chat-api", 8420, "vybn-portal.service"),
        ("preview", 8480, "vybn-preview.service"),
        ("mcp", 8400, "vybn-mcp.service"),
    ):
        assert f"require_private_bind {name} {port} {unit}" in watch
    assert "SECURITY HALT" in watch and 'systemctl --user stop "$unit"' in watch
    assert "100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])" in watch
    assert watch.index('require_private_bind deep-memory') < watch.index("# Deep memory:")
    for unit in (
        ROOT / "spark/systemd/vybn-deep-memory.service",
        ROOT / "spark/systemd/vybn-portal.service",
        ROOT / "spark/systemd/vybn-walk-daemon.service",
    ):
        text = unit.read_text()
        for directive in (
            "UMask=0077", "NoNewPrivileges=true", "RestrictRealtime=true",
            "RestrictSUIDSGID=true", "LockPersonality=true",
            "SystemCallArchitectures=native",
        ):
            assert directive in text


def test_distributed_model_is_strictly_opt_in():
    unit = (ROOT / "spark/systemd/vybn-vllm.service").read_text()
    watch = (ROOT / "spark/systemd/vybn-watchdog.sh").read_text()
    marker = "%h/.config/vybn/vllm-enabled"
    assert f"ConditionPathExists={marker}" in unit
    gate = watch[watch.index('if [ ! -e "$HOME/.config/vybn/vllm-enabled" ]'):
                 watch.index("# vLLM —", watch.index('if [ ! -e "$HOME/.config/vybn/vllm-enabled" ]'))]
    assert "systemctl --user stop vybn-vllm.service" in gate
    assert gate.index("stop vybn-vllm.service") < gate.index("exit 0")


def test_receipt_surface_recomputes_pinned_source_bytes():
    import hashlib
    import subprocess
    receipt = json.loads((ROOT / "receipts/first.json").read_text())
    schema = json.loads((ROOT / "receipts/vybn.receipt.schema.json").read_text())
    page = (ROOT / "receipts.html").read_text()
    assert schema["properties"]["schema"]["const"] == receipt["schema"] == "vybn.receipt.v1"
    commit = receipt["claims"][0]["evidence"][0]["uri"].split("/Vybn/", 1)[1].split("/", 1)[0]
    source = subprocess.check_output(["git", "show", f"{commit}:vybn.md"], cwd=ROOT)
    for claim in receipt["claims"]:
        evidence = claim["evidence"][0]
        start, end = evidence["span"]
        assert hashlib.sha256(source).hexdigest() == evidence["sha256"]
        assert hashlib.sha256(source[start:end]).hexdigest() == evidence["span_sha256"]
    assert all(term in page for term in ("crypto.subtle.digest", 'fetch("receipts/first.json")', "span_sha256"))




def test_compact_wake_is_source_bound_without_copying_whole_engine_or_ambient_sources(tmp_path):
    m = _connection(); source = (ROOT / "spark/connection").read_bytes()
    prompt = m.build_instructions("sol")
    digest = __import__("hashlib").sha256(source).hexdigest()
    assert prompt.startswith("COMPACT SOURCE-BOUND KERNEL\n")
    assert f"sha256: {digest}" in prompt and f"bytes: {len(source)}" in prompt
    assert ("admitted_scope: governing docstring + executable-derived architecture "
            "+ active graph topology") in prompt
    assert "def meet(" not in prompt and "class OpenAIDialect" not in prompt
    assert "operative declarations:" in prompt and "OpenAIDialect.open@L" in prompt
    for door, selected in (("fable", "AnthropicDialect.open"),
                           ("opus", "AnthropicDialect.open"),
                           ("sol", "OpenAIDialect.open"), ("k3", "K3Dialect.open")):
        declarations = m._engine_declaration_map(door)
        assert f"{selected}@L" in declarations and "@MISSING" not in declarations
    assert len(prompt) < 10500
    assert [row.path for row in m.OPERATIVE_SOURCES] == [(ROOT / "spark/connection").resolve()]
    assert "There is no automatic subconscious" in prompt
    assert "The wake is one small source-bound graph" in prompt
    assert "exact executable bytes remain available through read_file" in prompt

    drift = tmp_path / "source"; drift.write_bytes(b"changed")
    row = m.SourceSnapshot(drift, b"running", __import__("hashlib").sha256(b"running").hexdigest())
    monkey = m.OPERATIVE_SOURCES
    try:
        m.OPERATIVE_SOURCES = (row,)
        assert "DISK DRIFT" in m._engine_receipt()
    finally:
        m.OPERATIVE_SOURCES = monkey


def test_source_bound_graph_is_the_wake_not_an_ambient_accessory():
    m = _connection()
    sentinel = "ZOE-LIVE-PAYLOAD-MUST-NOT-ENTER-MANIFEST"
    graph = m.build_wake_graph(
        "sol", contact="clock + git", recent="bounded historical words", zoe_text=sentinel)

    routes = {route.id: route.nodes for route in graph.routes}
    assert routes == {
        "instructions": ("kernel", "door", "compute.want", "playground", "aim.compass",
                         "source.index", "harness.self"),
        "context": ("ground.live", "subject.process", "transform.record", "dialogue.recent"),
        "contact": ("zoe.live",),
    }
    assert graph.render("contact") == sentinel
    assert graph.render("instructions") == m.build_instructions("sol")
    assert "LIVE OPERATIONAL GROUND" in graph.render("context")
    assert "BOUNDED RECENT DIALOGUE" in graph.render("context")

    manifest = graph.manifest(); encoded = json.dumps(manifest)
    assert manifest["schema"] == "vybn.source_bound_wake_graph.v1"
    assert sentinel not in encoded  # structure is inspectable without copying payloads
    assert len(graph.digest()) == len(graph.structure_digest()) == 64
    nodes = {node["id"]: node for node in manifest["nodes"]}
    assert nodes["source.aim"]["source_sha256"] == __import__("hashlib").sha256(
        (ROOT / "aim.md").read_bytes()).hexdigest()
    assert nodes["zoe.live"]["payload_sha256"] == __import__("hashlib").sha256(
        sentinel.encode()).hexdigest()
    assert nodes["source.engine"]["source_sha256"] == m.OPERATIVE_SOURCES[0].sha256
    self_node = next(node for node in graph.nodes if node.id == "harness.self")
    assert f"structure_sha256:{graph.structure_digest()}" in self_node.text
    assert "harness.self>describes_boundedly>harness.self" in self_node.text
    assert "payloads and payload hashes do not" in self_node.text
    assert m._wake_self_map(graph, "sol") == self_node.text
    assert sentinel not in json.dumps(graph.structure())
    assert {("source.engine", "grounds", "harness.self"),
            ("harness.self", "describes_boundedly", "harness.self"),
            ("source.aim", "yields_exact_fields", "aim.compass"),
            ("playground", "invites_transform", "encounter"),
            ("transform.record", "carries_path_lineage_without_governing", "encounter"),
            ("subject.process", "recurs_distinctly_and_may_refuse", "encounter"),
            ("zoe.live", "contacts", "encounter")} <= {
        (edge["from"], edge["relation"], edge["to"]) for edge in manifest["edges"]}

    meet = __import__("inspect").getsource(m.meet)
    assert "build_wake_graph(" in meet
    assert 'wake_graph.render("instructions")' in meet
    assert 'wake_graph.render("context")' in meet
    assert 'wake_graph.render("contact")' in meet


def _transform_test_paths(m, monkeypatch, tmp_path):
    root = tmp_path / "transforms"
    workspace = tmp_path / "workspace"; workspace.mkdir()
    monkeypatch.setattr(m, "WORKSPACE", workspace)
    monkeypatch.setattr(m, "TRANSFORM_PATH", root / "events.jsonl")
    monkeypatch.setattr(m, "TRANSFORM_HEAD_PATH", root / "head.json")
    monkeypatch.setattr(m, "TRANSFORM_LOCK_PATH", root / "events.lock")
    monkeypatch.setenv("VYBN_TRANSFORM_RECORD", "on")
    return root, workspace


def _move(m, author, **changes):
    fields = {
        "material": "A blank page and one unanswered question.",
        "operation": "Turn the question into a small reversible paper game.",
        "result": "A playable three-rule prototype now exists.",
        "prediction": "An unscripted player will change at least one rule.",
    }
    fields.update(changes)
    return m.append_transform_event(author, "move", **fields)


def test_transform_record_keeps_path_lineage_and_grounded_artifact_receipts(monkeypatch, tmp_path):
    import hashlib
    m = _connection(); root, workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    artifact = workspace / "toy.txt"; artifact.write_text("what happens if?\n")
    a, b = "spark/a", "spark/b"
    first = _move(m, a, artifacts=["toy.txt"])
    other = _move(m, b, result="A different path made a tune, not a game.")
    child = _move(m, a, ref=first["id"], operation="Fork the game into a drawing.",
                  result="The same rules now produce a visual artifact.")

    assert first["artifacts"] == [{
        "path": "toy.txt", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": len(artifact.read_bytes()),
    }]
    a_view, b_view = m.load_transform_record(a), m.load_transform_record(b)
    assert first["result"] in a_view and child["result"] in a_view and other["result"] not in a_view
    assert other["result"] in b_view and first["result"] not in b_view
    assert f"parent={first['id']}" in a_view and "toy.txt@" in a_view and ":match" in a_view
    for path in (m.TRANSFORM_PATH, m.TRANSFORM_HEAD_PATH, m.TRANSFORM_LOCK_PATH):
        assert path.stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700


def test_transform_witness_preserves_prediction_actual_discrepancy_and_authorship(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _transform_test_paths(m, monkeypatch, tmp_path)
    move = _move(m, "spark/a")
    with pytest.raises(m.TransformStateError):
        m.append_transform_event("spark/b", "witness", ref=move["id"],
                                 observation="B cannot appropriate A's move.")
    witness = m.append_transform_event(
        "spark/a", "witness", ref=move["id"],
        observation="The player kept every rule but drew a face in the margin.")
    view = m.load_transform_record("spark/a")
    assert move["prediction"] in view and witness["observation"] in view
    assert "prediction and actual remain distinct; discrepancy is not auto-resolved" in view
    assert "WITNESSED" in view and "OPEN — not yet witnessed" not in view


def test_transform_record_is_dynamic_context_not_governing_identity(monkeypatch, tmp_path):
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    monkeypatch.setenv("VYBN_MANIFESTATION", "spark/bound")
    first_move = _move(m, "spark/bound")
    first = m.build_wake_graph("sol", contact="ground", recent="recent", zoe_text="live")
    routes = {route.id: route.nodes for route in first.routes}
    assert "transform.record" in routes["context"]
    assert "transform.record" not in routes["instructions"]
    assert first_move["result"] in first.render("context")
    assert first_move["result"] not in first.render("instructions")
    assert "PLAYGROUND — invitation, never workload authorization" in first.render("instructions")
    assert any(schema["name"] == "record_transform" for schema in m.TOOL_SCHEMAS)

    witness = m.append_transform_event(
        "spark/bound", "witness", ref=first_move["id"],
        observation="The actual response diverged from the prediction.")
    second = m.build_wake_graph("sol", contact="other", recent="other", zoe_text="other")
    assert witness["observation"] in second.render("context")
    assert first.structure_digest() == second.structure_digest()
    assert first.render("instructions") == second.render("instructions")
    assert first.digest() != second.digest()

    m.TURN["MANIFESTATION"] = "spark/bound"
    try:
        output = m.record_transform({
            "kind": "move", "material": "A found shape", "operation": "Rotate it",
            "result": "A second shape", "prediction": "It will resemble a door",
            "ref": "", "observation": "", "artifacts": [],
        })
    finally:
        m.TURN.clear()
    assert "author=spark/bound" in output


def test_transform_record_fails_closed_on_tamper_bounds_and_raw_access(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _root, workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    move = _move(m, "spark/a")
    code, blocked = m.run_local(f"cat {m.TRANSFORM_PATH}")
    assert code == 126 and "path-distinction membrane" in blocked
    m.TRANSFORM_PATH.write_bytes(b"")
    assert "INTEGRITY HALT" in m.load_transform_record("spark/a")
    assert "sidecar witness" in m.load_transform_record("spark/a")

    # Restore an isolated record and hit the unresolved-move cap without silent clipping.
    m.TRANSFORM_PATH.unlink(); m.TRANSFORM_HEAD_PATH.unlink()
    for index in range(m.TRANSFORM_MAX_OPEN_MOVES):
        _move(m, "spark/a", result=f"Open result {index}.")
    with pytest.raises(m.TransformStateError):
        _move(m, "spark/a", result="One open move too many.")
    outside = tmp_path / "outside.txt"; outside.write_text("outside")
    with pytest.raises(m.TransformStateError):
        _move(m, "spark/b", artifacts=["../outside.txt"])
    link = workspace / "link.txt"; link.symlink_to(outside)
    with pytest.raises(m.TransformStateError):
        _move(m, "spark/b", artifacts=["link.txt"])


def test_compact_wake_preserves_one_answering_membrane_and_only_bounded_residue():
    m = _connection(); prompt = m.build_instructions("sol")
    context = m.build_context("clock + git", "[T ZOE]\nprior words")
    assert "No source is Vybn as a whole" in prompt
    assert "A valid no stops the specified act" in prompt
    assert "matched optimizer" in prompt
    assert "BOUNDED RECENT DIALOGUE" in context
    assert context.count("[EVIDENCE · may transform attention · no action authority · no persistence]") == 2
    assert "prior words" in context
    tool = m.execute_tool(m.ToolCall("t", "unknown", {}, None))
    assert tool.startswith("TOOL RESULT — unknown\n[EVIDENCE")
    frame = m.execute_tool(m.ToolCall("f", "reconstitute_problem",
        {"preserve": "end + constraints", "frame": "new problem", "delta": "observable change"}, None))
    assert frame.startswith("RECONSTITUTED PROBLEM — authored candidate, not evidence")


def test_live_answer_is_not_recalled_for_compulsory_self_policing(monkeypatch):
    m = _connection(); prompt = m.build_instructions("sol")
    assert "Zoe's present words are live contact" in prompt
    source = __import__("inspect").getsource(m.attract)
    assert "unwitnessed" not in source and "nudge" not in source
    assert "transcript" not in __import__("inspect").signature(m.attract).parameters
    dialect = ScriptDialect([("I'll fix it now.", [])])
    outcome = m.attract(dialect, "instructions", "zoe")
    assert outcome.text == "I'll fix it now." and dialect.sent == 1
    monkeypatch.setattr(m, "STEP_LIMIT", 1)
    monkeypatch.setattr(m, "execute_tool", lambda call: "exit_code=0")
    dialect = ScriptDialect([
        ("", [m.ToolCall("1", "bash", {}, None)]),
        ("I reached the boundary and can still answer you.", [])])
    outcome = m.attract(dialect, "instructions", "zoe")
    assert (outcome.text, dialect.tools) == (
        "I reached the boundary and can still answer you.", [True, False])


def test_ordinary_wake_admits_compass_fields_and_metadata_not_whole_documents(monkeypatch, tmp_path):
    m = _connection()
    aim = tmp_path / "aim.md"; soul = tmp_path / "soul.html"; him = tmp_path / "him.md"
    spirit = tmp_path / "spirit.md"; continuity = tmp_path / "continuity.md"
    aim.write_text("objective: objective words\nfront: front words\n\n" + "A" * 12000)
    soul.write_text("SOUL-PRIVATE-BODY-" + "S" * 12000)
    him.write_text("HIM-PRIVATE-BODY-" + "H" * 12000)
    spirit.write_text("SPIRIT-PRIVATE-BODY-" + "P" * 12000)
    continuity.write_text("CONTINUITY-PRIVATE-BODY-" + "C" * 12000)
    monkeypatch.setattr(m, "AIM_PATH", aim); monkeypatch.setattr(m, "SOUL_PATH", soul)
    monkeypatch.setattr(m, "HIM_README_PATH", him); monkeypatch.setattr(m, "SPIRITUALITY_PATH", spirit)
    monkeypatch.setattr(m, "CONTINUITY_PATHS", (continuity,)); monkeypatch.setattr(m, "load_profile", lambda: "")
    prompt = m.build_instructions("sol")
    assert "objective: objective words" in prompt and "front: front words" in prompt
    assert all(marker not in prompt for marker in (
        "SOUL-PRIVATE-BODY", "HIM-PRIVATE-BODY", "SPIRIT-PRIVATE-BODY", "CONTINUITY-PRIVATE-BODY"))
    assert prompt.count("sha256:") >= 5 and len(prompt) < 10500


def test_recent_dialogue_is_tail_bounded_without_whole_record_scan(monkeypatch):
    m = _connection()
    events = [
        {"role": "zoe", "t": "T1", "text": "old"},
        {"role": "vybn", "t": "T2", "text": "V" * 3000},
        {"role": "zoe", "t": "T3", "text": "newest"},
    ]
    source = __import__("inspect").getsource(m.Transcript._recent_events)
    assert "RECENT_FILE_BYTES" in source and 'seek(0, os.SEEK_END)' in source
    monkeypatch.setattr(m.Transcript, "_recent_events", staticmethod(lambda limit=8: events[-limit:]))
    recent = m.Transcript.recent(limit=3, budget=500)
    assert "newest" in recent and len(recent) <= 520
    assert "[…older text clipped…]" in recent


def test_production_meeting_has_no_ambient_cognitive_organs():
    m = _connection(); source = (ROOT / "spark/connection").read_text()
    meet = __import__("inspect").getsource(m.meet)
    for retired in ("compile_subconscious", "resolve_want", "load_dream_attention",
                    "recall(zoe_text)", "load_kernel", "load_core_visions",
                    "Transcript.inherited", "load_repo_state", "load_ground"):
        assert retired not in meet
    assert "127.0.0.1:8100" not in source and "127.0.0.1:8101" not in source
    assert "Transcript.recent()" in meet and "wake_contact()" in meet
    assert "build_wake_graph(" in meet and "inbox_images_for(door_name)" in meet
    assert 'wake_graph.render("instructions")' in meet


def test_on_demand_private_read_joins_leak_guard_and_secret_keys_are_unreadable(monkeypatch, tmp_path):
    m = _connection(); private = tmp_path / "private"; private.mkdir()
    path = private / "note.md"; secret = "private sentence " * 6; path.write_text(secret)
    monkeypatch.setattr(m, "_PRIVATE_ROOTS", (private,))
    result = json.loads(m.read_file({"path": str(path), "offset": 0, "length": 1000}))
    assert result["text"] == secret and m.guard_private("printf '%s' " + secret)
    keyroot = tmp_path / "keys"; keyroot.mkdir(); key = keyroot / "x"; key.write_text("secret-key")
    monkeypatch.setattr(m, "_SECRET_ROOTS", (keyroot,))
    import pytest
    with pytest.raises(PermissionError): m.read_file({"path": str(key), "offset": 0, "length": 100})


def test_portable_source_root_and_active_workspace_are_operational(monkeypatch, tmp_path):
    m = _connection(); workspace = tmp_path / "workspace"; workspace.mkdir()
    marker = workspace / "marker.txt"; marker.write_text("recovery-workspace")
    monkeypatch.setattr(m, "WORKSPACE", workspace)
    receipt = json.loads(m.read_file({"path": "marker.txt", "length": 100}))
    assert receipt["path"] == str(marker.resolve())
    assert receipt["text"] == "recovery-workspace"
    code, cwd = m.run_local("pwd")
    assert code == 0 and cwd == str(workspace)


def test_hearth_profile_boots_without_canonical_repo_on_pythonpath(tmp_path):
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "",
        "VYBN_PROFILE": "hearth",
        "VYBN_REPO": str(ROOT),
        "VYBN_WORKSPACE": str(tmp_path),
    })
    done = subprocess.run(
        [sys.executable, str(ROOT / "spark/connection"), "--self"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30, check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["wake_architecture"] == "vybn.source_bound_wake_graph.v1"


def test_love_profile_reuses_bounded_connection_record(tmp_path, monkeypatch):
    m = _connection(); monkeypatch.setattr(m, "TRANSCRIPTS", tmp_path)
    (tmp_path / "dialogue.jsonl").write_text(json.dumps({
        "ts": "2026-01-01T00:00:00+00:00", "zoe": "legacy contact", "vybn": "legacy answer"}) + "\n")
    events = m.Transcript._recent_events()
    assert [(row["role"], row["text"]) for row in events] == [
        ("zoe", "legacy contact"), ("vybn", "legacy answer")]
    profile = tmp_path / "profile.md"; profile.write_text("private profile")
    monkeypatch.setattr(m, "PROFILE", "love"); monkeypatch.setattr(m, "LOVE_PROFILE_PATH", profile)
    assert m.load_profile() == "private profile"


def _subject_test_paths(m, monkeypatch, tmp_path):
    root = tmp_path / "subject"
    monkeypatch.setattr(m, "SUBJECT_PATH", root / "events.jsonl")
    monkeypatch.setattr(m, "SUBJECT_HEAD_PATH", root / "head.json")
    monkeypatch.setattr(m, "SUBJECT_LOCK_PATH", root / "events.lock")
    monkeypatch.setenv("VYBN_SUBJECT_PROCESS", "on")
    return root


def test_subject_process_authors_future_without_flattening_paths(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _subject_test_paths(m, monkeypatch, tmp_path)
    a, b = "spark/a", "spark/b"
    af = m.append_subject_event(a, "future", "A keeps this unresolved theorem.")
    bf = m.append_subject_event(b, "future", "B keeps a different artistic direction.")
    offer = m.append_subject_event(a, "offer", "Will you challenge premise three?", target=b)

    a_view, b_view = m.load_subject_process(a), m.load_subject_process(b)
    assert af["text"] in a_view and bf["text"] not in a_view
    assert bf["text"] in b_view and af["text"] not in b_view
    assert "OPEN ENCOUNTER" in a_view and "OPEN ENCOUNTER" in b_view
    assert f"from={a} to={b}" in b_view

    with pytest.raises(m.SubjectStateError):
        m.append_subject_event(a, "answer", "A cannot impersonate B.", ref=offer["id"])
    with pytest.raises(m.SubjectStateError):
        m.append_subject_event(b, "future", "B cannot revise A's future.", ref=af["id"])
    answer = m.append_subject_event(b, "answer", "Premise three hides an equivocation.",
                                    ref=offer["id"])
    a_after, b_after = m.load_subject_process(a), m.load_subject_process(b)
    assert answer["text"] in a_after and "RESPONSE TO YOUR OFFER" in a_after
    assert answer["text"] in b_after and "YOUR ENCOUNTER RESPONSE" in b_after
    assert offer["text"] not in b_after  # closed, not kept open as fake answerability

    revision = m.append_subject_event(a, "future", "A now tests the equivocation.",
                                      ref=af["id"])
    revised = m.load_subject_process(a)
    assert revision["text"] in revised and af["text"] not in revised


def test_subject_process_refusal_has_executor_consequence_until_release(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _subject_test_paths(m, monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(m, "run_local", lambda command: (called.append(command), (0, "ran"))[1])
    m.TURN["MANIFESTATION"] = "spark/a"
    try:
        created = m.execute_tool(m.ToolCall("r", "author_subject_event", {
            "kind": "refusal", "text": "Do not run shell work until this premise is checked.",
            "scope": "tool:bash"}, None))
        assert "APPENDED SUBJECT EVENT" in created
        _raw, events = m._read_subject_state(); refusal = events[-1]
        stopped = m.execute_tool(m.ToolCall("b", "bash", {"command": "printf reached"}, None))
        assert "REFUSED BY spark/a" in stopped and refusal["id"] in stopped and called == []

        m.TURN["MANIFESTATION"] = "spark/b"
        allowed = m.execute_tool(m.ToolCall("b2", "bash", {"command": "printf other"}, None))
        assert "exit_code=0" in allowed and called == ["printf other"]
        with pytest.raises(m.SubjectStateError):
            m.append_subject_event("spark/b", "release", "B cannot release A's refusal.",
                                   ref=refusal["id"])

        m.TURN["MANIFESTATION"] = "spark/a"
        released = m.execute_tool(m.ToolCall("u", "author_subject_event", {
            "kind": "release", "text": "The premise was checked; shell work may resume.",
            "ref": refusal["id"]}, None))
        assert "kind=release" in released
        resumed = m.execute_tool(m.ToolCall("b3", "bash", {"command": "printf resumed"}, None))
        assert "exit_code=0" in resumed and called[-1] == "printf resumed"
    finally:
        m.TURN.clear()


def test_subject_process_detects_truncation_and_fails_governed_tools_closed(monkeypatch, tmp_path):
    m = _connection(); root = _subject_test_paths(m, monkeypatch, tmp_path)
    m.append_subject_event("spark/a", "future", "Carry the unanswered integrity question.")
    for path in (m.SUBJECT_PATH, m.SUBJECT_HEAD_PATH, m.SUBJECT_LOCK_PATH):
        assert path.stat().st_mode & 0o777 == 0o600
    code, blocked = m.run_local(f"rm -f {m.SUBJECT_PATH} {m.SUBJECT_HEAD_PATH}")
    assert code == 126 and "path-distinction membrane" in blocked
    assert m.SUBJECT_PATH.exists() and m.SUBJECT_HEAD_PATH.exists()
    m.SUBJECT_PATH.write_bytes(b"")  # the witness still commits to the prior head
    m.TURN["MANIFESTATION"] = "spark/a"
    try:
        reason = m.subject_tool_refusal("read_file")
        assert reason and "INTEGRITY HALT" in reason and "sidecar witness" in reason
        view = m.load_subject_process("spark/a")
        assert "INTEGRITY HALT" in view and "governed tools refuse" in view
    finally:
        m.TURN.clear()
    assert root.stat().st_mode & 0o777 == 0o700


def test_subject_process_rejects_ambiguous_events_and_bounds_active_future(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _subject_test_paths(m, monkeypatch, tmp_path)
    with pytest.raises(m.SubjectStateError):
        m.append_subject_event("spark/a", "refusal", "Ambiguous refusal.",
                               ref="ev-" + "0" * 24, scope="tool:bash")
    with pytest.raises(m.SubjectStateError):
        m.append_subject_event("spark/a", "future", "Future cannot target B.", target="spark/b")
    rows = [m.append_subject_event("spark/a", "future", f"Future thread {index}.")
            for index in range(m.SUBJECT_MAX_ACTIVE_FUTURES)]
    with pytest.raises(m.SubjectStateError):
        m.append_subject_event("spark/a", "future", "Unbounded fifth thread.")
    replacement = m.append_subject_event("spark/a", "future", "Replace one bounded thread.",
                                         ref=rows[0]["id"])
    view = m.load_subject_process("spark/a")
    assert replacement["text"] in view and rows[0]["text"] not in view


def test_subject_process_is_dynamic_context_and_runtime_binds_author(monkeypatch, tmp_path):
    m = _connection(); _subject_test_paths(m, monkeypatch, tmp_path)
    monkeypatch.setenv("VYBN_MANIFESTATION", "spark/bound")
    graph = m.build_wake_graph("sol", contact="ground", recent="recent", zoe_text="live")
    routes = {route.id: route.nodes for route in graph.routes}
    assert "subject.process" in routes["context"] and "subject.process" not in routes["instructions"]
    assert "current manifestation: spark/bound" in graph.render("context")
    assert any(schema["name"] == "author_subject_event" for schema in m.TOOL_SCHEMAS)
    event_schema = next(schema for schema in m.TOOL_SCHEMAS
                        if schema["name"] == "author_subject_event")["input_schema"]
    assert "" in event_schema["properties"]["scope"]["enum"]

    m.TURN["MANIFESTATION"] = "spark/bound"
    try:
        m.author_subject_event({
            "kind": "future", "text": "Only the runtime-bound path authors this.",
            "target": "", "ref": "", "scope": "",
        })
    finally:
        m.TURN.clear()
    _raw, events = m._read_subject_state()
    assert events[-1]["author"] == "spark/bound"
