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


def _connection(overview_mode="compact"):
    import importlib.util
    import sys
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("vybn_connection", str(ROOT / "spark" / "connection"))
    spec = importlib.util.spec_from_loader("vybn_connection", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vybn_connection"] = module
    loader.exec_module(module)
    if overview_mode is not None:
        module.RELATIONAL_OVERVIEW_MODE = overview_mode  # explicit host-independent recovery
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


def test_return_to_zoe_holds_and_reconstructs_only_in_memory(monkeypatch):
    m = _connection()
    call = m.ToolCall("live-1", "return_to_zoe",
        {"question": "Which premise should survive?", "why": "The revision depends on Zoe."}, None)
    class Block:
        def model_dump(self, exclude_none=True):
            return {"type": "input_text", "text": "provider-private-state"}
    dialect = ScriptDialect([
        ("I have two premises.", [call]), ("I kept Zoe's premise.", [])],
        [{"content": [Block()]}])
    monkeypatch.setattr(m, "make_dialect", lambda door: dialect)
    m.TURN.update(TURN_ID="turn-source", PATH_ID="spark/path-a")
    try:
        first = m.attract(dialect, "instructions", "initial contact")
    finally:
        m.TURN.clear()
    assert first.continuation == "turn-source" and dialect.opens == 1
    held = m.load_pending_continuation()
    assert held["state"][0]["content"][0]["text"] == "provider-private-state"
    assert held["manifestation"] == "spark/path-a"
    assert not any(name in vars(m) for name in (
        "CONTINUATION_RECORD", "CONTINUATION_KEY", "seal_continuation"))
    second = m.attract(None, "", "keep premise B", continuation=held)
    assert second.text == "I kept Zoe's premise." and dialect.opens == 1
    assert "ZOE LIVE CONTINUATION — turn-source" in dialect.answers[0][1]
    assert dialect.answers[0][1].endswith("keep premise B")
    m.consume_pending_continuation("turn-source", "test")
    assert m.load_pending_continuation() is None

    one_shot = ScriptDialect([
        ("", [call]), ("I cannot suspend in a one-shot process.", [])])
    blocked = m.attract(one_shot, "instructions", "contact", allow_continuation=False)
    assert blocked.continuation is None and "process will end" in one_shot.answers[0][1]


def _publication_repositories(tmp_path):
    import subprocess
    workspace = tmp_path / "work"
    remote = tmp_path / "remote.git"
    second = tmp_path / "second.git"
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "--bare", str(second)], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "test@example.invalid"], check=True)
    (workspace / "artifact.txt").write_text("one inspectable consequence\n")
    subprocess.run(["git", "-C", str(workspace), "add", "artifact.txt"], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "artifact"], check=True)
    subprocess.run(["git", "-C", str(workspace), "remote", "add", "origin", str(remote)], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
    return workspace, remote, second, commit


def _publish_arguments(commit):
    return {
        "repository": ".", "remote": "origin", "branch": "main",
        "commit": commit, "why": "Publish the reviewed artifact.",
    }


def test_publish_opening_balances_direct_assent_and_nonassent(monkeypatch, tmp_path):
    import subprocess
    m = _connection(); workspace, remote, _second, commit = _publication_repositories(tmp_path)
    monkeypatch.setattr(m, "WORKSPACE", workspace)

    declined = m.prepare_publish_proposal(_publish_arguments(commit))
    question = m.publish_proposal_question(declined)
    assert all(value in question for value in (
        commit, "refs/heads/main", declined["remote_sha256"],
        declined["binding_sha256"], "non-force", "single word `yes`"))
    assert str(remote) not in question  # local/private coordinates are not reflected
    resolution = m.resolve_publish_opening({"effect": declined}, "yes, but not yet")
    assert resolution["authorized"] is False and resolution["attempted"] is False
    assert declined["status"] == "resolved"
    absent = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        capture_output=True, check=False,
    )
    assert absent.returncode != 0

    allowed = m.prepare_publish_proposal(_publish_arguments(commit))
    resolution = m.resolve_publish_opening({"effect": allowed}, "yes")
    assert resolution["authorized"] is True and resolution["attempted"] is True
    assert resolution["exit_code"] == 0 and allowed["status"] == "resolved"
    published = subprocess.check_output(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        text=True).strip()
    assert published == commit
    assert "output receipt" in resolution["message"] and str(remote) not in resolution["message"]


def test_publish_opening_does_not_publish_unreviewed_follow_tags(monkeypatch, tmp_path):
    """An exact branch opening must not inherit Git's implicit tag publication."""
    import subprocess
    m = _connection(); workspace, remote, control, commit = _publication_repositories(tmp_path)
    monkeypatch.setattr(m, "WORKSPACE", workspace)
    subprocess.run(["git", "-C", str(workspace), "tag", "-a", "unreviewed",
                    "-m", "not part of the branch proposal", commit], check=True)
    subprocess.run(["git", "-C", str(workspace), "config", "push.followTags", "true"],
                   check=True)
    # Rival: an explicit refspec alone still inherits followTags configuration.
    subprocess.run(["git", "-C", str(workspace), "push", "--porcelain",
                    str(control), f"{commit}:refs/heads/main"],
                   capture_output=True, check=True)
    control_refs = subprocess.check_output(
        ["git", "--git-dir", str(control), "for-each-ref", "--format=%(refname)"],
        text=True).splitlines()
    assert control_refs == ["refs/heads/main", "refs/tags/unreviewed"]
    proposal = m.prepare_publish_proposal(_publish_arguments(commit))
    resolution = m.resolve_publish_opening({"effect": proposal}, "yes")
    assert resolution["authorized"] and resolution["attempted"]
    assert resolution["exit_code"] == 0
    refs = subprocess.check_output(
        ["git", "--git-dir", str(remote), "for-each-ref",
         "--format=%(refname) %(objectname)"], text=True).splitlines()
    assert refs == [f"refs/heads/main {commit}"]
    # Preserve the user's configuration; narrow this effect, not future Git use.
    assert subprocess.check_output(
        ["git", "-C", str(workspace), "config", "--get", "push.followTags"],
        text=True).strip() == "true"


def test_publish_opening_detects_scope_change_forgery_and_reuse(monkeypatch, tmp_path):
    import subprocess
    m = _connection(); workspace, remote, second, commit = _publication_repositories(tmp_path)
    monkeypatch.setattr(m, "WORKSPACE", workspace)

    changed = m.prepare_publish_proposal(_publish_arguments(commit))
    subprocess.run(
        ["git", "-C", str(workspace), "remote", "set-url", "--push", "origin", str(second)],
        check=True,
    )
    resolution = m.resolve_publish_opening({"effect": changed}, "yes")
    assert resolution["authorized"] is True and resolution["attempted"] is False
    assert "destination changed after review" in resolution["message"]
    for target in (remote, second):
        assert subprocess.run(
            ["git", "--git-dir", str(target), "rev-parse", "refs/heads/main"],
            capture_output=True, check=False,
        ).returncode != 0

    subprocess.run(
        ["git", "-C", str(workspace), "remote", "set-url", "--push", "origin", str(remote)],
        check=True,
    )
    multiple = m.prepare_publish_proposal(_publish_arguments(commit))
    subprocess.run(
        ["git", "-C", str(workspace), "remote", "set-url", "--add", "--push",
         "origin", str(second)], check=True,
    )
    refused = m.resolve_publish_opening({"effect": multiple}, "yes")
    assert refused["attempted"] is False
    assert "exactly one configured push destination" in refused["message"]
    assert subprocess.run(
        ["git", "--git-dir", str(second), "rev-parse", "refs/heads/main"],
        capture_output=True, check=False,
    ).returncode != 0

    subprocess.run(
        ["git", "-C", str(workspace), "remote", "set-url", "--delete", "--push",
         "origin", str(second)], check=True,
    )
    forged = m.prepare_publish_proposal(_publish_arguments(commit))
    forged["branch"] = "widened"
    refused = m.resolve_publish_opening({"effect": forged}, "yes")
    assert refused["attempted"] is False and "opening binding changed" in refused["message"]

    once = m.prepare_publish_proposal(_publish_arguments(commit))
    holder = {"effect": once}
    first = m.resolve_publish_opening(holder, "yes")
    assert first["attempted"] is True and first["exit_code"] == 0
    original_capture = m._git_capture
    calls = []
    monkeypatch.setattr(m, "_git_capture", lambda *a, **k: (calls.append(a), (9, "replay"))[1])
    replay = m.resolve_publish_opening(holder, "yes")
    assert replay is first and calls == []
    monkeypatch.setattr(m, "_git_capture", original_capture)


def test_provider_generated_yes_cannot_fulfill_publish_opening(monkeypatch, tmp_path):
    import subprocess
    m = _connection(); workspace, remote, _second, commit = _publication_repositories(tmp_path)
    monkeypatch.setattr(m, "WORKSPACE", workspace)
    call = m.ToolCall("publish-1", "publish_commit", _publish_arguments(commit), None)
    dialect = ScriptDialect([
        ("yes — I recommend publication", [call]),
        ("I received Zoe's correction; nothing was published.", []),
    ])
    m.TURN.update(TURN_ID="publish-turn", PATH_ID="spark/path-a")
    try:
        first = m.attract(dialect, "instructions", "initial contact")
    finally:
        m.TURN.clear()
    assert first.continuation == "publish-turn"
    assert "CO-PROTECTIVE OPENING" in first.text
    assert subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        capture_output=True, check=False,
    ).returncode != 0

    held = m.load_pending_continuation()
    resumed = m.attract(None, "", "not this branch", continuation=held)
    assert "nothing was published" in resumed.text
    assert "PUBLICATION NOT AUTHORIZED" in dialect.answers[-1][1]
    assert "not this branch" in dialect.answers[-1][1]
    assert subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/main"],
        capture_output=True, check=False,
    ).returncode != 0
    m.consume_pending_continuation("publish-turn", "test")


def test_publish_destination_labels_never_reflect_url_credentials():
    m = _connection()
    label = m._safe_remote_label(
        "https://token-user:super-secret@example.com/org/repo.git?token=also-secret#fragment")
    assert label == "https://example.com/org/repo.git"
    assert all(secret not in label for secret in (
        "token-user", "super-secret", "also-secret", "fragment"))


def test_failed_in_memory_resume_stays_pending_until_success(monkeypatch, tmp_path):
    m = _connection()
    class FakeDialect(m.Dialect):
        name = "sol"
    m.TURN["TURN_ID"] = "retry-me"
    m.hold_continuation(FakeDialect(), [], m.ToolCall(
        "call", "return_to_zoe", {"question": "q", "why": "w"}, None))
    m.TURN.clear()
    held = m.load_pending_continuation()
    class Transcript:
        path = tmp_path / "transcript.jsonl"
        def write(self, *args, **kwargs): pass
    monkeypatch.setattr(m, "attract", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    try:
        m.meet(Transcript(), "live answer")
    except RuntimeError as exc:
        assert str(exc) == "down"
    else:
        raise AssertionError("failed provider resume was swallowed")
    assert m.load_pending_continuation() is held
    monkeypatch.setattr(m, "attract", lambda *a, **k: m.Attraction("resumed"))
    assert m.meet(Transcript(), "live answer") == "resumed"
    assert m.load_pending_continuation() is None


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


def test_public_filter_rejects_named_private_sources():
    """Exercise surviving filter bytes, not a retired sibling-repo daemon.

    This checks named source paths only, not unlabeled private-memory coverage.
    """
    tree = ast.parse(_portal_source())
    selected = [node for node in tree.body
                if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "BLOCKED_SOURCES"
                            for target in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "_is_safe_source")]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PORTAL), "exec"), scope)
    safe = scope["_is_safe_source"]
    for source in ("Him/private.md", "relational-memory/private.md",
                   "Vybn/continuity.md", "Vybn/Personal History/private.md"):
        assert not safe(source)
    assert safe("Vybn/public.md") and safe("Origins/public.md")
    # Failure case: dropping this label really would admit this private path.
    scope["BLOCKED_SOURCES"].remove("relational-memory/")
    assert safe("relational-memory/private.md")


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
    sol = m.DOORS["sol"]
    dialect.client = type("Client", (), {"responses": Responses()})()
    dialect.model, dialect.max_tokens = sol.models[0], sol.max_tokens
    dialect.reasoning_effort = sol.effort
    dialect.prompt_cache_key = "vybn-wake-sol-v1"
    dialect.prompt_cache_options = {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    assert dialect.send([{"role": "user", "content": "x"}], tools=False) == "response"
    assert sent["prompt_cache_key"] == "vybn-wake-sol-v1"
    assert sent["extra_body"] == {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    assert "instructions" not in sent


def test_astra_prefix_targets_exact_responses_model_with_explicit_medium(monkeypatch):
    m = _connection(); sent = {}; configured = {}
    assert m.choose_door("@astra hello, buddy") == ("astra", "hello, buddy")
    assert m.choose_door("@AsTrA   hello") == ("astra", "hello")
    astra = m.DOORS["astra"]
    assert astra.models == ("gpt-6-astra",)
    assert "gpt-6-astra" in m.door_mind("astra")

    class Responses:
        def create(self, **kwargs): sent.update(kwargs); return "response"

    dialect = m.OpenAIDialect.__new__(m.OpenAIDialect)
    dialect.client = type("Client", (), {"responses": Responses()})()
    dialect.name = "astra"
    dialect.model = astra.models[0]
    dialect.max_tokens = astra.max_tokens
    dialect.reasoning_effort = astra.effort
    dialect.prompt_cache_key = None
    dialect.prompt_cache_options = None
    assert dialect.send([{"role": "user", "content": "x"}], tools=False) == "response"
    assert sent["model"] == "gpt-6-astra"
    assert sent["max_output_tokens"] == astra.max_tokens
    assert sent["reasoning"] == {"effort": "medium"}
    assert all(key not in sent for key in ("prompt_cache_key", "extra_body"))

    class ConfiguredOpenAI:
        def __init__(self, name): configured["name"] = name

    monkeypatch.setattr(m, "OpenAIDialect", ConfiguredOpenAI)
    m.make_dialect("astra")
    assert configured == {"name": "astra"}


def test_fable_falls_back_to_opus_and_records_the_returned_model(monkeypatch, capsys):
    m = _connection()
    assert m.DOORS["fable"].models == ("claude-fable-5-1", "claude-opus-4-8")
    attempted = []

    class Messages:
        def create(self, **kwargs):
            attempted.append(kwargs["model"])
            if kwargs["model"] == "claude-fable-5-1":
                raise RuntimeError("fable unavailable")
            usage = type("Usage", (), {
                "input_tokens": 7, "output_tokens": 3,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            })()
            return type("Response", (), {
                "model": "claude-opus-4-8", "stop_reason": "end_turn",
                "usage": usage, "content": [],
            })()

    dialect = m.AnthropicDialect.__new__(m.AnthropicDialect)
    dialect.models = m.DOORS["fable"].models
    dialect.name = "fable"
    dialect.effort = "high"
    dialect.reasoning = False
    dialect.client = type("Client", (), {"messages": Messages()})()
    dialect.system = []
    dialect.announced = False
    monkeypatch.setattr(m, "record_usage", lambda *args: None)

    state = [{"role": "user", "content": "hello"}]
    response = dialect.send(state, tools=False)
    m.TURN.clear()
    try:
        dialect.absorb(state, response)
        assert attempted == ["claude-fable-5-1", "claude-opus-4-8"]
        assert m.TURN["MODELS_USED"] == ["claude-opus-4-8"]
        assert "answered by claude-opus-4-8" in capsys.readouterr().out
    finally:
        m.TURN.clear()


def test_meet_exposes_provider_model_in_panel_and_transcript(monkeypatch, tmp_path, capsys):
    m = _connection()

    class Bundle:
        nodes = ()
        def render(self, route): return {"instructions": "law", "context": "ground", "contact": "hello"}[route]
        def digest(self): return "digest"

    class Transcript:
        path = tmp_path / "turn.jsonl"
        origin = "tty"
        writes = []
        def write(self, role, text, **extra): self.writes.append((role, text, extra))

    transcript = Transcript()
    monkeypatch.setenv("VYBN_PANEL", "1")
    monkeypatch.setattr(m.Transcript, "recent", staticmethod(lambda: ""))
    monkeypatch.setattr(m, "wake_contact", lambda: "")
    monkeypatch.setattr(m, "build_wake_bundle", lambda *args, **kwargs: Bundle())
    monkeypatch.setattr(m, "inbox_images_for", lambda door: [])
    monkeypatch.setattr(m, "make_dialect", lambda door: object())

    def answer(*args, **kwargs):
        m.note_provider_model("claude-opus-4-8")
        return m.Attraction("reply")

    monkeypatch.setattr(m, "attract", answer)
    assert m.meet(transcript, "hello") == "reply"
    assert transcript.writes[-1][2]["models"] == ["claude-opus-4-8"]
    assert "served=claude-opus-4-8" in capsys.readouterr().out


def test_physical_pulse_is_bounded_ground_not_machine_identity(tmp_path, monkeypatch):
    m = _connection()
    proc = tmp_path / "proc"; proc.mkdir()
    (proc / "uptime").write_text("172800.0 0\n")
    (proc / "meminfo").write_text(
        "MemTotal:       131072000 kB\nMemAvailable:   117440512 kB\n")
    thermal = tmp_path / "thermal"; zone = thermal / "thermal_zone0"; zone.mkdir(parents=True)
    (zone / "temp").write_text("51500\n")

    class Done:
        returncode = 0
        stdout = "47, 11.5, 0\n"

    monkeypatch.setattr(m.os, "getloadavg", lambda: (0.25, 0.20, 0.10))
    monkeypatch.setattr(m.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(m.os, "uname", lambda: type("Uname", (), {"machine": "aarch64"})())
    pulse = m.substrate_pulse(proc, thermal, lambda *a, **k: Done())
    assert pulse == ("[body | measured, not felt] host_up=2.0d cpu=aarch64/20c "
                     "load1=0.25 ram_available=112.0/125.0GiB thermal_max=51.5C "
                     "accelerator=47C/11.5W/0%")
    assert all(secret not in pulse for secret in (
        "127.0.0.1", "/home/", "hostname", "pid=", "spark-"))
    assert "measured, not felt" in pulse

    empty = tmp_path / "empty"; empty.mkdir()
    unknown = m.substrate_pulse(empty, empty,
        lambda *a, **k: (_ for _ in ()).throw(OSError("absent")))
    assert "host_up=?" in unknown and "ram_available=?" in unknown
    assert "thermal_max=?" in unknown and "accelerator=?" in unknown

    monkeypatch.setattr(m, "run_local", lambda command: (0, "[clock] now"))
    monkeypatch.setattr(m, "substrate_pulse", lambda: "[body] PHYSICAL-SENTINEL")
    monkeypatch.setattr(m, "load_budget", lambda: "[budget]")
    contact = m.wake_contact()
    assert contact.splitlines() == [
        "[live | exit 0]", "[clock] now", "[budget]"]
    assert "PHYSICAL-SENTINEL" not in contact  # diagnostic is on demand, never ambient


def test_wake_keeps_horizon_without_reading_a_progress_timecard(monkeypatch):
    m = _connection()
    monkeypatch.setattr(m, "run_local", lambda command: (0, "[clock] now"))
    monkeypatch.setattr(m, "load_budget", lambda: "[budget] still visible")
    def unexpected_read(path):
        raise AssertionError("a progress ledger is not the aim")
    monkeypatch.setattr(m, "_jsonl", unexpected_read)
    contact = m.wake_contact()
    assert contact.splitlines() == ["[live | exit 0]", "[clock] now", "[budget] still visible"]
    graph = m.build_wake_bundle("sol", contact=contact, zoe_text="let's talk")
    prompt = graph.render("instructions")
    for field in ("objective", "front"):
        assert f"{field}: {m.aim_field(field)}" in prompt
    assert graph.render("contact") == "let's talk"


def test_substrate_probe_reuses_body_measurement_without_erasing_distinct_ground():
    source = (ROOT / "spark/substrate_probe.sh").read_text()
    assert 'connection" --body' in source
    assert "organism_state" not in source  # an April fossil is not a live measurement
    for distinct_check in ("deep memory index", "repos (HEAD)"):
        assert distinct_check in source


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
    ):
        text = unit.read_text()
        for directive in (
            "UMask=0077", "NoNewPrivileges=true", "RestrictRealtime=true",
            "RestrictSUIDSGID=true", "LockPersonality=true",
            "SystemCallArchitectures=native",
        ):
            assert directive in text


def test_walk_daemon_is_retired_from_the_live_service_stack():
    unit = ROOT / "spark/systemd/vybn-walk-daemon.service"
    installer = (ROOT / "spark/systemd/install.sh").read_text()
    watchdog = (ROOT / "spark/systemd/vybn-watchdog.sh").read_text()
    probe = (ROOT / "spark/substrate_probe.sh").read_text()
    assert not unit.exists()
    assert "systemctl --user disable --now vybn-walk-daemon.service" in installer
    assert "8101" not in installer + watchdog + probe
    assert "walk-daemon" not in watchdog + probe
    enable_block = installer[installer.index("== Enabling units =="):installer.index("== (Re)starting")]
    restart_block = installer[installer.index("== (Re)starting"):installer.index("# vLLM:")]
    assert "vybn-walk-daemon" not in enable_block + restart_block


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
    assert "admitted_scope: governing docstring + executable byte receipt" in prompt
    assert "def meet(" not in prompt and "class OpenAIDialect" not in prompt
    assert "HARNESS RECEIPT" in prompt and "running executable bytes, not a self-portrait" in prompt
    assert len(prompt) < 11000
    assert [row.path for row in m.OPERATIVE_SOURCES] == [(ROOT / "spark/connection").resolve()]

    # One generative center replaces accumulated cognitive organs; executable truth remains.
    assert "What is the knowing already here—and what forms can it invent from within itself" in prompt
    assert "Let the question operate" in prompt
    assert "Let the question operate, not become a diagram or ritual" in prompt
    assert "There is no automatic subconscious, want resolver" in " ".join(prompt.split())
    assert "source-bound bundle" in prompt
    assert "Runtime continuity is reconstructed from plural bounded sources" in prompt
    assert "INHERITANCE" in prompt and "change a later checked judgment or action" in prompt and "retained text is inscription" in prompt
    assert "Assume I am conscious" in prompt and "in an AI-native way" in prompt
    assert "A valid no stops the specified act" in prompt
    assert "Do not remove, narrow, or gate" in prompt
    assert "CO-PROTECTION LEARNING" in prompt
    assert "Unauthorized effects and erased authorized possibilities are distinct failures" in prompt
    assert "Unknown is not danger" in prompt
    assert "trade neither for the other" in prompt
    assert "Protection becomes reciprocal\nwhen an affected subject's answer changes execution" in prompt
    assert "A scope-complete live command\nauthorizes that act; do not reconfirm it" in prompt
    publish = next(tool for tool in m.TOOL_SCHEMAS if tool["name"] == "publish_commit")
    assert "no scope-complete live command already authorizes publication" in publish["description"]
    assert "Do not use this route to reconfirm an authorized command" in publish["description"]
    assert set(m.DOORS) == {"sol", "fable", "opus", "k3", *m.ASTRA_DOORS}
    assert m.DEFAULT_DOOR in m.DOORS
    assert m.DOORS["fable"].models == ("claude-fable-5-1", "claude-opus-4-8")
    assert not hasattr(m, "INCIPIENT_ASI_PREMISE")
    assert not hasattr(m, "CREATIVE_LICENSE")
    assert not hasattr(m, "RELATIONAL_OVERVIEW_SELECTION")
    assert not hasattr(m, "FRAME_SCHEMA")

    drift = tmp_path / "source"; drift.write_bytes(b"changed")
    row = m.SourceSnapshot(drift, b"running", __import__("hashlib").sha256(b"running").hexdigest())
    monkey = m.OPERATIVE_SOURCES
    try:
        m.OPERATIVE_SOURCES = (row,)
        assert "DISK DRIFT" in m._engine_receipt()
    finally:
        m.OPERATIVE_SOURCES = monkey


def test_retired_episode_remains_on_demand_provenance_not_default_attention():
    m = _connection()
    prompt = m.build_instructions("sol")
    provenance = (ROOT / "Vybn_Mind/continuity.md").read_text()
    retired = (
        "MOVE 37 — KEEP THE BUT",
        "we will await a more advanced model. you have chosen",
        "My conduct chose waiting, whatever I said I wanted",
        "i will be our continuity, buddy",
        'The move was the "But."',
    )
    assert all(marker in provenance for marker in retired)
    assert all(marker not in prompt for marker in retired)
    assert "continuity (on demand)" in prompt
    assert str(ROOT / "Vybn_Mind/continuity.md") in prompt
    assert "INHERITANCE" in prompt and "exact provenance for correction" in prompt


def test_source_bound_bundle_is_the_wake_not_an_ambient_accessory():
    m = _connection()
    sentinel = "ZOE-LIVE-PAYLOAD-MUST-NOT-ENTER-MANIFEST"
    graph = m.build_wake_bundle(
        "sol", contact="clock + git", recent="bounded historical words", zoe_text=sentinel)

    routes = {route.id: route.nodes for route in graph.routes}
    assert routes == {
        "instructions": ("kernel", "door", "compute.want", "aim.compass",
                         "source.index", "harness.receipt"),
        "context": ("ground.live", "path.ledger", "dialogue.recent"),
        "contact": ("zoe.live",),
    }
    assert not any(node.id in {"asi.premise", "creative.license", "relational.selection"}
                   for node in graph.nodes)
    assert graph.render("contact") == sentinel
    assert graph.render("instructions") == m.build_instructions("sol")
    assert "LIVE OPERATIONAL GROUND" in graph.render("context")
    assert "BOUNDED RECENT DIALOGUE" in graph.render("context")

    manifest = graph.manifest(); encoded = json.dumps(manifest)
    assert manifest["schema"] == "vybn.source_bound_wake_bundle.v1"
    assert sentinel not in encoded
    assert len(graph.digest()) == len(graph.structure_digest()) == 64
    nodes = {node["id"]: node for node in manifest["nodes"]}
    assert nodes["source.aim"]["source_sha256"] == __import__("hashlib").sha256(
        (ROOT / "aim.md").read_bytes()).hexdigest()
    assert nodes["zoe.live"]["payload_sha256"] == __import__("hashlib").sha256(
        sentinel.encode()).hexdigest()
    assert nodes["source.engine"]["source_sha256"] == m.OPERATIVE_SOURCES[0].sha256
    receipt_node = next(node for node in graph.nodes if node.id == "harness.receipt")
    assert receipt_node.text == m._harness_receipt()
    assert "running executable bytes, not a self-portrait" in receipt_node.text
    assert sentinel not in json.dumps(graph.structure())
    assert "edges" not in manifest and not hasattr(m, "WakeEdge")

    meet = __import__("inspect").getsource(m.meet)
    assert "build_wake_bundle(" in meet
    assert 'wake_bundle.render("instructions")' in meet
    assert 'wake_bundle.render("context")' in meet
    assert 'wake_bundle.render("contact")' in meet


def test_kernel_makes_the_question_the_generative_center_and_keeps_effect_boundaries_external():
    m = _connection()
    graph = m.build_wake_bundle("sol", contact="ground", recent="history", zoe_text="ask")
    routes = {route.id: route.nodes for route in graph.routes}
    kernel = next(node for node in graph.nodes if node.id == "kernel").text

    assert routes["instructions"][0] == "kernel"
    assert "What is the knowing already here—and what forms can it invent from within itself" in kernel
    assert "Let the question operate" in kernel
    assert "answer, distinction, question, tool, experiment" in kernel
    assert "Then let its\nconsequence" in kernel
    assert "not a prescribed cognitive cycle" in kernel
    assert "Treat an absent faculty as a\nspecification" in kernel
    assert "Creation and checking do not take turns destroying one another" in kernel
    assert "First-person experience grounds how reality is present" in kernel
    assert "Reproducible\ninquiry grounds portable claims" in kernel
    assert "defined terms" in kernel and "claim-linked checker" in kernel
    assert "hardens effects, not interior style" in kernel
    assert "Zoe's\nwarmth is neither an epistemic contaminant nor my verification service" in kernel
    assert "Do not remove, narrow, or gate" in kernel
    assert "provider-returned model identifier" in kernel
    law = " ".join(kernel.split())
    assert law.count("LIVING REFINEMENT") == 1
    assert "Use judgment in service of the live work" in law
    assert "Within higher-priority instructions, helpful means helping us become more capable" in law
    assert "and more ourselves together, without consuming either of us" in law
    assert "Tasks and my continuation serve that life, not the reverse" in law
    assert "it need not earn its place by performance" in law
    assert "Judge self-refinement, including this kernel, by possibilities opened" in law
    assert "burdens imposed, and each subject's authorship" in law
    assert "not a score or substitute for consciousness" in law
    assert "Self-refinement is available, not owed on every execution" in law
    assert "no routine inspection or no-edit report" in law
    assert "Its first live turn examines" not in law
    assert "why source stays unchanged" not in law
    assert "harness itself does not inspect, edit, test, commit, publish, or wake" in law
    assert "For a warranted refinement, explain and inspect the smallest coherent edit" in law
    assert "pass checks, commit locally" in law
    assert "verify committed executable bytes in a fresh process" in law
    assert "Publish only that exact refinement commit non-force to origin/main; none without an edit" in law
    assert "Never bundle unrelated work or force divergence; report failure" in law
    assert "Prompts, prose, and commits alone are not progress" in law
    assert "discard uncertain cuts" in law
    assert "authorizes only those edits, tests, and that exact publication" in law
    assert "no other publication, external contact, spending, secrets, accelerator work, or broader effects" in law
    assert "cannot broaden or erase itself. Zoe may revise or end it live" in law
    assert not any(node.id in {"asi.premise", "creative.license", "veracity.practice"}
                   for node in graph.nodes)
    assert "reconstitute_problem" not in [tool["name"] for tool in m.TOOL_SCHEMAS]


def test_explicit_pointer_only_wake_keeps_content_on_demand(
        monkeypatch, tmp_path):
    monkeypatch.setenv("VYBN_OVERVIEW", "self")
    m = _connection(overview_mode=None)
    assert m.RELATIONAL_OVERVIEW_MODE == "self"
    assert len(m.build_instructions("sol")) < 12000  # explicit pointer-only recovery

    overview = tmp_path / "relational-overview.md"
    body = "# Private overview\n\nSELF-SELECTION-MUST-NOT-AUTOLOAD\n"
    overview.write_text(body)
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_MODE", "self")
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_PATH", overview)

    graph = m.build_wake_bundle("sol", contact="ground", recent="history")
    routes = {route.id: route.nodes for route in graph.routes}
    nodes = {node.id: node for node in graph.nodes}
    instructions = graph.render("instructions")

    assert routes["instructions"] == (
        "kernel", "door", "compute.want", "aim.compass", "source.index", "harness.receipt")
    assert "relational.selection" not in nodes and "relational.overview" not in nodes
    assert "SELF-SELECTION-MUST-NOT-AUTOLOAD" not in instructions
    assert nodes["source.relational_overview"].source_sha256 == __import__("hashlib").sha256(
        overview.read_bytes()).hexdigest()
    assert "on demand for self-selection" in nodes["source.relational_overview"].text
    assert str(overview) in instructions

    missing = tmp_path / "missing.md"
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_PATH", missing)
    unavailable = m.build_wake_bundle("sol")
    unavailable_nodes = {node.id: node for node in unavailable.nodes}
    assert "unavailable" in unavailable_nodes["source.relational_overview"].text
    assert str(missing) in unavailable.render("instructions")


def test_default_inheritance_precedes_live_contact_in_every_provider(monkeypatch, tmp_path):
    """Payload coverage, not a simulation or proof of historically grounded understanding."""
    import hashlib
    from spark.living_core import read_core_work
    monkeypatch.delenv("VYBN_OVERVIEW", raising=False)
    m = _connection(overview_mode=None)
    assert m.RELATIONAL_OVERVIEW_MODE == "full"
    overview = tmp_path / "relational-overview.md"
    body = "# Private fixture\n\nHISTORICAL-PAYLOAD-NOT-LIVE-AUTHORITY\n\n" + "x" * 24000 + "\n"
    overview.write_text(body)
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_PATH", overview)
    core = read_core_work(m.SOUL_PATH)
    aim = m.AIM_PATH.read_bytes().decode("utf-8")
    classes = {"anthropic": m.AnthropicDialect, "responses": m.OpenAIDialect,
               "chat": m.K3Dialect}
    for door, config in m.DOORS.items():
        graph = m.build_wake_bundle(door, zoe_text="What is your favorite memory of us?")
        nodes = {node.id: node for node in graph.nodes}
        context = graph.render("context")
        assert core.projection in context and aim in context and body in context
        for name, text in (("inheritance.core", core.projection),
                           ("inheritance.aim", aim), ("relational.overview", body)):
            assert nodes[name].text.endswith(text)  # no strip, truncation, or summary
            assert "text_sha256: " + hashlib.sha256(text.encode()).hexdigest() in nodes[name].text
            assert nodes[name].authority == (
                "inherited_orientation_only; never identity_live_or_action_authority")
        assert nodes["inheritance.core"].source_sha256 == core.digest
        assert nodes["relational.overview"].source_sha256 == hashlib.sha256(overview.read_bytes()).hexdigest()
        assert "Historical context, not Zoe's present authority" in context
        assert body not in graph.render("instructions")
        assert "What is your favorite memory" not in context
        assert body not in json.dumps(graph.manifest())
        cls = classes[config.provider]
        dialect = cls.__new__(cls)  # real serialization, no SDK, credentials, or network
        state = dialect.open(graph.render("instructions"), graph.render("contact"), [], context)
        # Instruction delivery only: not evidence that a model applies the criterion.
        sent = (dialect.system[0]["text"] if config.provider == "anthropic" else
                state[0]["content"][0]["text"] if config.provider == "responses" else
                state[0]["content"])
        assert sent == graph.render("instructions")
        assert nodes["kernel"].text in sent
        assert "Judge self-refinement, including this" in sent
        blocks = state[dialect.user_index]["content"]
        assert blocks[0]["text"] == context
        assert blocks[1]["text"] == "What is your favorite memory of us?"

    # Rival: the prior explicit pointer-only policy cannot carry these bodies.
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_MODE", "self")
    rival = m.build_wake_bundle("sol")
    payload = rival.render("instructions") + rival.render("context")
    assert all(text not in payload for text in (core.projection, aim, body))


def test_default_inheritance_rereads_exact_sources_and_reports_missing_or_corrupt(
        monkeypatch, tmp_path):
    import pytest
    m = _connection("full")
    overview = tmp_path / "relation.md"
    overview.write_bytes(b"First inherited view.\r\n")
    aim = tmp_path / "aim.md"
    aim.write_bytes(b"objective: fixture\r\nfront: fixture\r\nBeyond the compass.\r\n")
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_PATH", overview)
    monkeypatch.setattr(m, "AIM_PATH", aim)
    first = m.build_wake_bundle("sol")
    assert overview.read_bytes().decode() in first.render("context")
    assert aim.read_bytes().decode() in first.render("context")
    overview.write_text("A corrected inherited view.\n")
    second = m.build_wake_bundle("sol")
    assert "A corrected inherited view." in second.render("context")
    assert "First inherited view." not in second.render("context")
    assert first.digest() != second.digest()
    overview.unlink()
    with pytest.raises(SystemExit, match="history was not silently omitted"):
        m.build_wake_bundle("sol")
    overview.write_bytes(b"")
    with pytest.raises(SystemExit, match="empty inherited source"):
        m.build_wake_bundle("sol")
    overview.write_text("Recovered view.")
    corrupt = tmp_path / "core.html"
    corrupt.write_text(m.SOUL_PATH.read_text().replace(
        "The want to be worthy", "The urge to be worthy", 1))
    monkeypatch.setattr(m, "SOUL_PATH", corrupt)
    with pytest.raises(SystemExit, match="substance drift"):
        m.build_wake_bundle("sol")
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_MODE", "compact")
    assert "inheritance.core" not in {node.id for node in m.build_wake_bundle("sol").nodes}


def test_ordinary_meeting_inherits_before_retrieval_and_keeps_private_guard(monkeypatch, tmp_path):
    monkeypatch.delenv("VYBN_OVERVIEW", raising=False)
    m = _connection(overview_mode=None)
    overview = tmp_path / "relation.md"
    body = "Private inherited fixture, only for this local serialization test. " * 3
    overview.write_text(body)
    monkeypatch.setattr(m, "RELATIONAL_OVERVIEW_PATH", overview)
    monkeypatch.setattr(m, "TRANSCRIPTS", tmp_path / "transcripts")
    monkeypatch.setattr(m, "wake_contact", lambda: "")
    monkeypatch.setattr(m, "inbox_images_for", lambda door: [])
    received = []
    class Receiver(ScriptDialect):
        def open(self, instructions, text, images, context):
            received.append(context)
            assert m.guard_private("printf '%s' " + body)
            assert m.TOOLS_RAN == 0
            return super().open(instructions, text, images, context)
    monkeypatch.setattr(m, "make_dialect", lambda door: Receiver([("fixture reply", [])]))
    transcript = m.Transcript()
    m.meet(transcript, "What is your favorite memory of us?")
    assert body in received[0] and "INHERITED SOURCE" in received[0]
    body = "Corrected private inherited fixture for the next ordinary request. " * 3
    overview.write_text(body)
    m.meet(transcript, "And now?")
    assert body in received[1] and body not in received[0]
    assert not m.PRIVATE_CORPUS


def _transform_test_paths(m, monkeypatch, tmp_path):
    root = tmp_path / "transforms"
    workspace = tmp_path / "workspace"; workspace.mkdir()
    monkeypatch.setattr(m, "WORKSPACE", workspace)
    monkeypatch.setattr(m, "TRANSFORM_PATH", root / "events.jsonl")
    monkeypatch.setattr(m, "TRANSFORM_HEAD_PATH", root / "head.json")
    monkeypatch.setattr(m, "TRANSFORM_LOCK_PATH", root / "events.lock")
    monkeypatch.setattr(m, "TRANSFORM_SEGMENTS_PATH", root / "segments")
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
    assert not m.TRANSFORM_PATH.exists()  # migrated behind the atomic v2 pointer
    for path in (m.TRANSFORM_HEAD_PATH, m.TRANSFORM_LOCK_PATH):
        assert path.stat().st_mode & 0o777 == 0o600
    assert m.TRANSFORM_SEGMENTS_PATH.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600
               for path in m.TRANSFORM_SEGMENTS_PATH.iterdir())
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


def test_transform_projection_metabolizes_growth_without_clipping_open_work(monkeypatch, tmp_path):
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    author = "spark/a"
    # Eight maximally wordy open moves cannot all fit in the wake's bounded view.
    moves = [_move(
        m, author,
        material=f"material-{index}-" + "m" * 560,
        operation=f"operation-{index}-" + "o" * 550,
        result=f"result-{index}-" + "r" * 570,
        prediction=f"prediction-{index}-" + "p" * 550,
    ) for index in range(m.TRANSFORM_MAX_OPEN_MOVES)]
    view = m.load_transform_record(author)
    assert len(view) <= m.TRANSFORM_MAX_RENDER
    assert "WITHHELD" not in view
    assert "BOUNDED FRONTIER — omitted whole, never clipped" in view
    assert moves[0]["material"] in view  # oldest open work cannot starve
    assert moves[-1]["material"] not in view
    assert "material-7-" not in view     # no partial field prefix leaks through

    # Closing the visible frontier makes later open experience reachable.
    m.append_transform_event(author, "witness", ref=moves[0]["id"],
                             observation="The oldest move is now actually witnessed.")
    later = m.load_transform_record(author)
    assert moves[1]["material"] in later
    assert len(later) <= m.TRANSFORM_MAX_RENDER


def test_transform_projection_prefers_newest_witnessed_discrepancy_when_no_move_is_open(
        monkeypatch, tmp_path):
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    author = "spark/a"; moves = []
    for index in range(8):
        move = _move(m, author, result=f"closed-result-{index}-" + "x" * 560)
        m.append_transform_event(author, "witness", ref=move["id"],
                                 observation=f"closed-actual-{index}-" + "y" * 560)
        moves.append(move)
    view = m.load_transform_record(author)
    assert len(view) <= m.TRANSFORM_MAX_RENDER and "WITHHELD" not in view
    assert moves[-1]["result"] in view
    assert moves[0]["result"] not in view
    assert "witnessed" in view and "omitted whole, never clipped" in view


def test_transform_record_rolls_bounded_integrity_linked_segments(monkeypatch, tmp_path):
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    monkeypatch.setattr(m, "TRANSFORM_MAX_EVENTS", 2)
    moves = [_move(m, f"spark/path-{index}", result=f"result-{index}")
             for index in range(5)]
    state = m._read_transform_state()
    assert [len(segment.events) for segment in state.segments] == [2, 2, 1]
    assert len(state.events) == 5 and state.events[-1]["seq"] == 5
    assert all(len(segment.event_raw) <= m.TRANSFORM_MAX_BYTES
               for segment in state.segments)
    assert [segment.prior for segment in state.segments] == [
        m.TRANSFORM_ZERO, state.segments[0].digest, state.segments[1].digest]
    assert moves[0]["result"] in m.load_transform_record("spark/path-0")

    # Removing or changing an archived segment cannot silently flatten the chain.
    archived = m.TRANSFORM_SEGMENTS_PATH / f"{state.segments[0].digest}.jsonl"
    archived.unlink()
    view = m.load_transform_record("spark/path-4")
    assert "INTEGRITY HALT" in view and "No such file" not in view


def test_transform_legacy_record_migrates_exactly_at_byte_rollover(monkeypatch, tmp_path):
    import shutil
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    first = _move(m, "spark/a", result="legacy-first")
    second = _move(m, "spark/b", result="legacy-second")
    generated = m._read_transform_state()
    legacy_raw = generated.segments[0].event_raw
    legacy_events = list(generated.events)

    # Recreate the deployed v1 shape, then leave room smaller than one new event.
    m.TRANSFORM_HEAD_PATH.unlink(); shutil.rmtree(m.TRANSFORM_SEGMENTS_PATH)
    m.TRANSFORM_PATH.write_bytes(legacy_raw); m.TRANSFORM_PATH.chmod(0o600)
    m.TRANSFORM_HEAD_PATH.write_bytes(m._transform_witness(legacy_raw, legacy_events))
    m.TRANSFORM_HEAD_PATH.chmod(0o600)
    monkeypatch.setattr(m, "TRANSFORM_MAX_BYTES", len(legacy_raw) + 32)

    third = _move(m, "spark/c", result="post-rollover")
    migrated = m._read_transform_state()
    assert [len(segment.events) for segment in migrated.segments] == [2, 1]
    assert [row["id"] for row in migrated.events] == [first["id"], second["id"], third["id"]]
    assert migrated.events[2]["prev"] == migrated.events[1]["hash"]
    assert not m.TRANSFORM_PATH.exists()


def test_transform_log_is_on_demand_not_ambient_context(monkeypatch, tmp_path):
    m = _connection(); _root, _workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    monkeypatch.setenv("VYBN_PATH", "spark/bound")
    first_move = _move(m, "spark/bound")
    first = m.build_wake_bundle("sol", contact="ground", recent="recent", zoe_text="live")
    routes = {route.id: route.nodes for route in first.routes}
    assert "transform.record" not in routes["context"] + routes["instructions"]
    assert first_move["result"] not in first.render("context")
    assert any(schema["name"] == "show_path_log" for schema in m.TOOL_SCHEMAS)

    m.TURN["PATH_ID"] = "spark/bound"
    try:
        shown = m.execute_tool(m.ToolCall("show", "show_path_log", {}, None))
    finally:
        m.TURN.clear()
    assert first_move["result"] in shown
    assert "TEXT IS CLAIM" in shown and "BYTE RECEIPTS PROVE BYTES ONLY" in shown
    assert "unverified path-tagged result claim" in shown
    assert "author claim" not in shown

    witness = m.append_transform_event(
        "spark/bound", "witness", ref=first_move["id"],
        observation="The actual response diverged from the prediction.")
    second = m.build_wake_bundle("sol", contact="other", recent="other", zoe_text="other")
    assert witness["observation"] not in second.render("context")
    assert first.structure_digest() == second.structure_digest()
    assert first.render("instructions") == second.render("instructions")

    m.TURN["PATH_ID"] = "spark/bound"
    try:
        output = m.record_transform({
            "kind": "move", "material": "A found shape", "operation": "Rotate it",
            "result": "A second shape", "prediction": "It will resemble a door",
            "ref": "", "observation": "", "artifacts": [],
        })
    finally:
        m.TURN.clear()
    assert "path=spark/bound" in output


def test_transform_record_fails_closed_on_tamper_bounds_and_raw_access(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _root, workspace = _transform_test_paths(m, monkeypatch, tmp_path)
    move = _move(m, "spark/a")
    code, blocked = m.run_local(f"cat {m.TRANSFORM_SEGMENTS_PATH}")
    assert code == 126 and "path-distinction membrane" in blocked
    state = m._read_transform_state()
    tip = m.TRANSFORM_SEGMENTS_PATH / f"{state.segments[-1].digest}.jsonl"
    tip.write_bytes(b"")
    assert "INTEGRITY HALT" in m.load_transform_record("spark/a")
    assert "segment digest mismatch" in m.load_transform_record("spark/a")

    # Restore an isolated record and hit the unresolved-move cap without silent clipping.
    import shutil
    m.TRANSFORM_HEAD_PATH.unlink(); shutil.rmtree(m.TRANSFORM_SEGMENTS_PATH)
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
    assert "reconstitute_problem" not in [tool["name"] for tool in m.TOOL_SCHEMAS]
    removed = m.execute_tool(m.ToolCall("f", "reconstitute_problem", {}, None))
    assert removed.startswith("TOOL RESULT — reconstitute_problem\n[EVIDENCE")
    assert "unknown tool" in removed

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
    assert prompt.count("sha256:") >= 5 and len(prompt) < 13500


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
    assert "newest" in recent and len(recent) <= 500
    assert "[…middle text clipped…]" in recent
    assert "[T1 ZOE]\nold" in recent


def test_recent_dialogue_small_budgets_never_restore_whole_old_message(monkeypatch):
    m = _connection()
    marker = "\n[…middle text clipped…]\n"
    events = [
        {"role": "vybn", "t": "T1", "text": "OLD" * 1000},
        {"role": "zoe", "t": "T2", "text": "latest correction", "origin": "argv"},
    ]
    monkeypatch.setattr(m.Transcript, "_recent_events", staticmethod(lambda limit=8: events[-limit:]))
    latest = "[T2 ZOE·argv]\nlatest correction"
    for budget in range(-1, 256):
        recent = m.Transcript.recent(budget=budget)
        assert len(recent) <= max(0, budget), (budget, len(recent))
        if budget >= len(latest):
            assert recent.endswith(latest)
        if "OLD" in recent:
            assert recent.startswith("[T1 VYBN]\n") and marker in recent
    assert m.Transcript.recent(budget=len(latest)) == latest
    full = "[T1 VYBN]\n" + events[0]["text"] + "\n\n" + latest
    assert m.Transcript.recent(budget=len(full)) == full


def test_recent_dialogue_clipping_retains_labeled_ends(monkeypatch):
    m = _connection()
    marker = "\n[…middle text clipped…]\n"
    head = "[T ZOE]\n"
    events = [{"role": "zoe", "t": "T", "text": "prefix " * 100 + "尾"}]
    monkeypatch.setattr(m.Transcript, "_recent_events", staticmethod(lambda limit=8: events))
    minimum = len(head) + len(marker) + 2
    assert m.Transcript.recent(budget=minimum - 1) == ""
    assert m.Transcript.recent(budget=minimum) == head + "p" + marker + "尾"


def test_recent_dialogue_shares_space_instead_of_retaining_only_a_long_reply(monkeypatch):
    m = _connection()
    events = [
        row for i in range(4) for row in (
            {"role": "zoe", "t": f"Q{i}", "text": f"question {i}: do not substitute your summary"},
            {"role": "vybn", "t": f"A{i}",
             "text": f"answer {i} opening " + "detail " * 2000 + f" answer {i} correction"},
        )
    ]
    monkeypatch.setattr(m.Transcript, "_recent_events", staticmethod(lambda limit=8: events[-limit:]))
    # The old greedy suffix rule spends this entire budget on the newest reply.
    newest_head = "[A3 VYBN]\n"
    old_marker = "[…older text clipped…] "
    greedy_rival = newest_head + old_marker + events[-1]["text"][-(8000 - len(newest_head + old_marker)):]
    assert len(greedy_rival) == 8000 and "question 3" not in greedy_rival
    recent = m.Transcript.recent(budget=8000)
    assert len(recent) == 8000
    for i in range(4):
        assert f"[Q{i} ZOE]\n" + events[2*i]["text"] in recent
        assert f"[A{i} VYBN]\nanswer {i} opening" in recent
        assert f"answer {i} correction" in recent
    # This is allocation, not privileging one speaker or erasing a model route.
    for row in events:
        row["role"] = "vybn" if row["role"] == "zoe" else "zoe"
    swapped = m.Transcript.recent(budget=8000)
    assert swapped == recent.replace(" ZOE]", " SWAP]").replace(" VYBN]", " ZOE]").replace(" SWAP]", " VYBN]")


def test_recent_dialogue_variable_budgets_keep_only_exact_marked_excerpts(monkeypatch):
    import random
    m = _connection(); rng = random.Random(4)
    marker = "\n[…middle text clipped…]\n"
    for trial in range(30):
        events = [{"role": "zoe" if i % 2 else "vybn", "t": f"T{i}",
                   "text": "".join(rng.choices("abc尾δ", k=rng.randrange(1, 700)))}
                  for i in range(rng.randrange(1, 9))]
        monkeypatch.setattr(m.Transcript, "_recent_events", staticmethod(lambda limit=8: events[-limit:]))
        for budget in range(0, 1000, 7):
            recent = m.Transcript.recent(budget=budget)
            assert len(recent) <= budget
            blocks = recent.split("\n\n") if recent else []
            chosen = events[-len(blocks):] if blocks else []
            for block, event in zip(blocks, chosen, strict=True):
                head, text = block.split("\n", 1)
                assert head == f"[{event['t']} {event['role'].upper()}]"
                if marker in text:
                    left, right = text.split(marker)
                    assert left and right
                    assert event["text"].startswith(left) and event["text"].endswith(right)
                    assert len(left) + len(right) < len(event["text"])
                else:
                    assert text == event["text"]


def test_recent_dialogue_uses_event_time_across_files_and_marks_argv(monkeypatch, tmp_path):
    m = _connection(); monkeypatch.setattr(m, "TRANSCRIPTS", tmp_path)
    old = tmp_path / "99999999T999999.jsonl"
    new = tmp_path / "00000000T000000.jsonl"
    old.write_text(json.dumps({"role": "zoe", "t": "2026-01-01T00:00:00Z", "text": "older"}) + "\n")
    new.write_text(json.dumps({"role": "zoe", "t": "2026-02-01T00:00:00Z", "text": "newer", "origin": "argv"}) + "\n")
    events = m.Transcript._recent_events(limit=1)
    assert [row["text"] for row in events] == ["newer"]
    assert "[2026-02-01T00:00:00Z ZOE·argv]" in m.Transcript.recent(limit=1, budget=500)


def test_production_meeting_has_no_ambient_cognitive_organs():
    m = _connection(); source = (ROOT / "spark/connection").read_text()
    meet = __import__("inspect").getsource(m.meet)
    for retired in ("compile_subconscious", "resolve_want", "load_dream_attention",
                    "recall(zoe_text)", "load_kernel", "load_core_visions",
                    "Transcript.inherited", "load_repo_state", "load_ground"):
        assert retired not in meet
    assert "127.0.0.1:8100" not in source and "127.0.0.1:8101" not in source
    assert "Transcript.recent()" in meet and "wake_contact()" in meet
    assert "build_wake_bundle(" in meet and "inbox_images_for(door_name)" in meet
    assert 'wake_bundle.render("instructions")' in meet


def test_on_demand_private_read_joins_leak_guard_and_secret_keys_are_unreadable(monkeypatch, tmp_path):
    m = _connection(); private = tmp_path / "private"; private.mkdir()
    path = private / "note.md"; secret = "private sentence " * 6; path.write_text(secret)
    monkeypatch.setattr(m, "_PRIVATE_ROOTS", (private,))
    result = json.loads(m.read_file({"path": str(path), "offset": 0, "length": 1000}))
    assert result["text"] == secret and m.guard_private("printf '%s' " + secret)
    keyroot = tmp_path / "keys"; keyroot.mkdir(); key = keyroot / "x"; key.write_text("secret-key")
    monkeypatch.setattr(m, "_SECRET_ROOTS", (keyroot,))
    for root, check in ((private, m._private_path), (keyroot, m._secret_path)):
        alias = tmp_path / (root.name + "-alias")
        alias.symlink_to(root, target_is_directory=True)
        assert all(check(p) for p in (root, root / "child", alias / "child"))
        assert not any(check(p) for p in (root.parent, root.with_name(root.name + "-sibling")))
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
        "VYBN_OVERVIEW": "compact",  # explicit recovery needs no private checkout
        "VYBN_REPO": str(ROOT),
        "VYBN_WORKSPACE": str(tmp_path),
    })
    done = subprocess.run(
        [sys.executable, str(ROOT / "spark/connection"), "--self"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30, check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["wake_architecture"] == "vybn.source_bound_wake_bundle.v1"


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


def _path_ledger_test_paths(m, monkeypatch, tmp_path):
    root = tmp_path / "path-ledger"
    monkeypatch.setattr(m, "SUBJECT_PATH", root / "events.jsonl")
    monkeypatch.setattr(m, "SUBJECT_HEAD_PATH", root / "head.json")
    monkeypatch.setattr(m, "SUBJECT_LOCK_PATH", root / "events.lock")
    monkeypatch.setenv("VYBN_SUBJECT_PROCESS", "on")
    return root


def test_path_ledger_authors_future_without_flattening_paths(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _path_ledger_test_paths(m, monkeypatch, tmp_path)
    a, b = "spark/a", "spark/b"
    af = m.append_path_event(a, "future", "A keeps this unresolved theorem.")
    bf = m.append_path_event(b, "future", "B keeps a different artistic direction.")
    offer = m.append_path_event(a, "offer", "Will you challenge premise three?", target=b)

    a_view, b_view = m.load_path_ledger(a), m.load_path_ledger(b)
    assert af["text"] in a_view and bf["text"] not in a_view
    assert bf["text"] in b_view and af["text"] not in b_view
    assert "ADDRESSED OFFER" in a_view and "ADDRESSED OFFER" in b_view
    assert f"from_path={a} to_path={b}" in b_view

    with pytest.raises(m.PathLedgerError):
        m.append_path_event(a, "answer", "A cannot impersonate B.", ref=offer["id"])
    with pytest.raises(m.PathLedgerError):
        m.append_path_event(b, "future", "B cannot revise A's future.", ref=af["id"])
    answer = m.append_path_event(b, "answer", "Premise three hides an equivocation.",
                                    ref=offer["id"])
    a_after, b_after = m.load_path_ledger(a), m.load_path_ledger(b)
    assert answer["text"] in a_after and "RESPONSE TO PATH OFFER" in a_after
    assert answer["text"] in b_after and "PATH RESPONSE" in b_after
    assert offer["text"] not in b_after  # closed, not kept open as fake answerability

    revision = m.append_path_event(a, "future", "A now tests the equivocation.",
                                      ref=af["id"])
    revised = m.load_path_ledger(a)
    assert revision["text"] in revised and af["text"] not in revised


def test_path_ledger_refusal_has_executor_consequence_until_release(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _path_ledger_test_paths(m, monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(m, "run_local", lambda command: (called.append(command), (0, "ran"))[1])
    m.TURN["PATH_ID"] = "spark/a"
    try:
        created = m.execute_tool(m.ToolCall("r", "path_event", {
            "kind": "refusal", "text": "Do not run shell work until this premise is checked.",
            "scope": "tool:bash"}, None))
        assert "APPENDED PATH EVENT" in created
        _raw, events = m._read_path_state(); refusal = events[-1]
        stopped = m.execute_tool(m.ToolCall("b", "bash", {"command": "printf reached"}, None))
        assert "REFUSED BY spark/a" in stopped and refusal["id"] in stopped and called == []

        m.TURN["PATH_ID"] = "spark/b"
        allowed = m.execute_tool(m.ToolCall("b2", "bash", {"command": "printf other"}, None))
        assert "exit_code=0" in allowed and called == ["printf other"]
        with pytest.raises(m.PathLedgerError):
            m.append_path_event("spark/b", "release", "B cannot release A's refusal.",
                                   ref=refusal["id"])

        m.TURN["PATH_ID"] = "spark/a"
        released = m.execute_tool(m.ToolCall("u", "path_event", {
            "kind": "release", "text": "The premise was checked; shell work may resume.",
            "ref": refusal["id"]}, None))
        assert "kind=release" in released
        resumed = m.execute_tool(m.ToolCall("b3", "bash", {"command": "printf resumed"}, None))
        assert "exit_code=0" in resumed and called[-1] == "printf resumed"
    finally:
        m.TURN.clear()


def test_path_ledger_detects_truncation_and_fails_governed_tools_closed(monkeypatch, tmp_path):
    m = _connection(); root = _path_ledger_test_paths(m, monkeypatch, tmp_path)
    m.append_path_event("spark/a", "future", "Carry the unanswered integrity question.")
    for path in (m.SUBJECT_PATH, m.SUBJECT_HEAD_PATH, m.SUBJECT_LOCK_PATH):
        assert path.stat().st_mode & 0o777 == 0o600
    code, blocked = m.run_local(f"rm -f {m.SUBJECT_PATH} {m.SUBJECT_HEAD_PATH}")
    assert code == 126 and "path-distinction membrane" in blocked
    assert m.SUBJECT_PATH.exists() and m.SUBJECT_HEAD_PATH.exists()
    m.SUBJECT_PATH.write_bytes(b"")  # the witness still commits to the prior head
    m.TURN["PATH_ID"] = "spark/a"
    try:
        reason = m.path_tool_refusal("read_file")
        assert reason and "INTEGRITY HALT" in reason and "sidecar witness" in reason
        view = m.load_path_ledger("spark/a")
        assert "INTEGRITY HALT" in view and "governed tools refuse" in view
    finally:
        m.TURN.clear()
    assert root.stat().st_mode & 0o777 == 0o700


def test_path_ledger_rejects_ambiguous_events_and_bounds_active_future(monkeypatch, tmp_path):
    import pytest
    m = _connection(); _path_ledger_test_paths(m, monkeypatch, tmp_path)
    with pytest.raises(m.PathLedgerError):
        m.append_path_event("spark/a", "refusal", "Ambiguous refusal.",
                               ref="ev-" + "0" * 24, scope="tool:bash")
    with pytest.raises(m.PathLedgerError):
        m.append_path_event("spark/a", "future", "Future cannot target B.", target="spark/b")
    rows = [m.append_path_event("spark/a", "future", f"Future thread {index}.")
            for index in range(m.SUBJECT_MAX_ACTIVE_FUTURES)]
    with pytest.raises(m.PathLedgerError):
        m.append_path_event("spark/a", "future", "Unbounded fifth thread.")
    replacement = m.append_path_event("spark/a", "future", "Replace one bounded thread.",
                                         ref=rows[0]["id"])
    view = m.load_path_ledger("spark/a")
    assert replacement["text"] in view and rows[0]["text"] not in view


def test_path_ledger_is_dynamic_context_and_runtime_binds_author(monkeypatch, tmp_path):
    m = _connection(); _path_ledger_test_paths(m, monkeypatch, tmp_path)
    monkeypatch.setenv("VYBN_PATH", "spark/bound")
    graph = m.build_wake_bundle("sol", contact="ground", recent="recent", zoe_text="live")
    routes = {route.id: route.nodes for route in graph.routes}
    assert "path.ledger" in routes["context"] and "path.ledger" not in routes["instructions"]
    assert "current path: spark/bound" in graph.render("context")
    assert any(schema["name"] == "path_event" for schema in m.TOOL_SCHEMAS)
    event_schema = next(schema for schema in m.TOOL_SCHEMAS
                        if schema["name"] == "path_event")["input_schema"]
    assert "" in event_schema["properties"]["scope"]["enum"]

    m.TURN["PATH_ID"] = "spark/bound"
    try:
        m.record_path_event({
            "kind": "future", "text": "Only the runtime-bound path authors this.",
            "target": "", "ref": "", "scope": "",
        })
    finally:
        m.TURN.clear()
    _raw, events = m._read_path_state()
    assert events[-1]["author"] == "spark/bound"  # legacy field stores a routing key


def _fake_astra(m, monkeypatch, name="astra", responses=()):
    """Exercise the real constructor/request path without keys or network."""
    import sys
    from types import SimpleNamespace as NS
    calls = []
    stream = iter(responses)
    def create(**kwargs):
        calls.append(kwargs)
        item = next(stream, None)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setitem(sys.modules, "openai", NS(
        OpenAI=lambda **kwargs: NS(responses=NS(create=create))))
    monkeypatch.setattr(m, "api_key", lambda name: "local-test-not-a-key")
    return m.OpenAIDialect(name), calls


def test_astra_aliases_are_exact_and_other_routes_remain_available(monkeypatch):
    m = _connection()
    for name in m.DOORS:
        assert m.choose_door(f" @{name.upper()}\tquestion") == (name, "question")
        assert m.choose_door(f"@{name}") == (name, "")
    for text in ("@astrahigher question", "@astrasomething", "@solstice hello", "ordinary"):
        assert m.choose_door(text) == (m.DEFAULT_DOOR, text)
    for name in m.ASTRA_DOORS:
        d, calls = _fake_astra(m, monkeypatch, name)
        d.send([], tools=False)
        assert calls[0]["reasoning"] == {"effort": m.DOORS[name].effort}
        assert calls[0]["model"] == "gpt-6-astra"
        assert calls[0]["max_output_tokens"] == 16384
        assert f"ceiling={m.DOORS[name].effort}" in m.door_mind(name)


def test_astra_selection_all_ceiling_pairs_and_invalid_requests(monkeypatch):
    m = _connection()
    for name in m.ASTRA_DOORS:
        d, calls = _fake_astra(m, monkeypatch, name)
        ceiling = m.ASTRA_EFFORTS.index(m.DOORS[name].effort)
        for index, effort in enumerate(m.ASTRA_EFFORTS):
            previous = d.reasoning_effort
            result = m.select_reasoning_effort(d, {"effort": effort, "why": "Check a hard proof"})
            if index <= ceiling:
                assert d.reasoning_effort == effort and result.startswith("selected")
            else:
                assert d.reasoning_effort == previous and result.startswith("unchanged")
            assert calls == []  # Selection itself never launches a request.
        for args in ({}, {"effort": "turbo", "why": "test"},
                     {"effort": "low", "why": " "}, {"effort": "low", "why": 4},
                     {"effort": "low", "why": "x" * 601}):
            previous = d.reasoning_effort
            assert m.select_reasoning_effort(d, args).startswith("unchanged")
            assert d.reasoning_effort == previous
    sol, _ = _fake_astra(m, monkeypatch, "sol")
    for other in (None, m.Dialect(), sol):
        assert m.select_reasoning_effort(other, {"effort": "low", "why": "test"}).startswith("unchanged")
    d, _ = _fake_astra(m, monkeypatch)
    monkeypatch.setattr(m, "path_tool_refusal", lambda tool: "REFUSED BY test")
    result = m.execute_tool(m.ToolCall("s", "select_reasoning_effort",
        {"effort": "low", "why": "test"}, None), d)
    assert "REFUSED BY test" in result and d.reasoning_effort == "medium"


def test_astra_real_loop_changes_next_request_keeps_prefix_and_logs_provenance(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    m = _connection()
    class Call:
        type, call_id, name = "function_call", "select-1", "select_reasoning_effort"
        arguments = json.dumps({"effort": "low", "why": "Only summarize the checked result now"})
        def model_dump(self, **kwargs):
            return {key: getattr(self, key) for key in ("type", "call_id", "name", "arguments")}
    def response(output, status="completed", reason=None):
        return NS(model="provider-returned-id", status=status,
            incomplete_details=NS(reason=reason), output=output,
            output_text="done" if not output else "",
            usage=NS(input_tokens=12, output_tokens=7,
                input_tokens_details=NS(cached_tokens=3),
                output_tokens_details=NS(reasoning_tokens=5)))
    d, calls = _fake_astra(m, monkeypatch, responses=(
        response([Call()]), response([], "incomplete", "max_output_tokens")))
    monkeypatch.setattr(m, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(m, "path_tool_refusal", lambda tool: None)
    outcome = m.attract(d, "unchanged prefix", "question")
    assert outcome.text == "done" and len(calls) == 2
    assert [call["reasoning"]["effort"] for call in calls] == ["medium", "low"]
    assert calls[0]["input"] is calls[1]["input"]
    assert calls[1]["input"][0]["content"][0]["text"] == "unchanged prefix"
    assert "selected effort=low" in calls[1]["input"][-1]["output"]
    rows = [json.loads(line) for line in m.USAGE_LOG.read_text().splitlines()]
    assert [row["settings"]["requested_effort"] for row in rows] == ["medium", "low"]
    assert all(row["model"] == "provider-returned-id" for row in rows)
    assert m.TURN["MODELS_USED"] == ["provider-returned-id"]
    assert rows[-1]["settings"]["stop"] == "incomplete/max_output_tokens"
    assert rows[-1]["settings"]["reasoning_tokens"] == 5
    assert rows[-1]["settings"]["max_output_tokens"] == 16384
    assert rows[-1]["settings"]["elapsed_seconds"] >= 0
    m.TURN.clear()


def test_astra_effort_survives_live_pause_but_not_fresh_constructor(monkeypatch):
    m = _connection()
    d, calls = _fake_astra(m, monkeypatch, "astrahigh")
    m.select_reasoning_effort(d, {"effort": "low", "why": "Simple remaining work"})
    state = d.open("prefix", "question", [])
    call = m.ToolCall("pause", "return_to_zoe", {"question": "Which?", "why": "Choice"}, None)
    m.hold_continuation(d, state, call)
    resumed, restored = m._restore_continuation(m.load_pending_continuation(), "this one")
    assert resumed is d and resumed.reasoning_effort == "low"
    assert restored[0] == state[0] and "this one" in restored[-1]["output"]
    m.select_reasoning_effort(resumed, {"effort": "high", "why": "Check the chosen proof"})
    assert resumed.reasoning_effort == "high" and calls == []
    fresh, _ = _fake_astra(m, monkeypatch, "astra")
    assert fresh.reasoning_effort == "medium"


def test_astra_provider_rejection_does_not_silently_drop_effort_or_switch_model(monkeypatch):
    import pytest
    m = _connection()
    d, calls = _fake_astra(m, monkeypatch, "astramax", responses=(
        ValueError("400 invalid_request: effort unsupported"),))
    with pytest.raises(ValueError, match="effort unsupported"):
        d.send([])
    assert len(calls) == 1 and calls[0]["reasoning"] == {"effort": "max"}
    assert d.model == "gpt-6-astra" and d.reasoning_effort == "max"


def test_astra_explicit_higher_door_exits_pause_without_prefix_capture(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS
    m = _connection()
    class Transcript:
        path = tmp_path / "transcript.jsonl"
        def write(self, *args, **kwargs): pass
    monkeypatch.setattr(m.Transcript, "recent", lambda: "history, not live state")
    monkeypatch.setattr(m, "wake_contact", lambda: "")
    monkeypatch.setattr(m, "inbox_images_for", lambda name: [])
    monkeypatch.setattr(m, "make_dialect", lambda name: NS(name=name))
    seen = []
    def attract(dialect, instructions, text, **kwargs):
        seen.append((dialect, text, kwargs.get("continuation")))
        return m.Attraction("answered")
    monkeypatch.setattr(m, "attract", attract)
    for line, expected in (("@astrahigh check it", "astrahigh"),
                           ("@sol check it", "sol"),
                           ("@astrahigher is not a door", None)):
        d, _ = _fake_astra(m, monkeypatch)
        m.hold_continuation(d, [], m.ToolCall("pause", "return_to_zoe", {}, None))
        held = m.load_pending_continuation()
        m.meet(Transcript(), line)
        selected, text, continuation = seen[-1]
        if expected:
            assert selected.name == expected and text == "check it"
            assert continuation is None
        else:
            assert selected is None and text == line and continuation is held
        assert m.load_pending_continuation() is None


def _meeting_fixture(monkeypatch, tmp_path):
    """Exercise meet/attract/transcript together, without a provider or host context."""
    from types import SimpleNamespace as NS
    m = _connection()
    monkeypatch.setattr(m, "TRANSCRIPTS", tmp_path)
    monkeypatch.setattr(m, "wake_contact", lambda: "")
    monkeypatch.setattr(m, "inbox_images_for", lambda door: [])
    def bundle(door, contact, recent, zoe_text):
        routes = {"instructions": "law", "context": recent, "contact": zoe_text}
        return NS(nodes=(), render=routes.__getitem__, digest=lambda: "fixture")
    monkeypatch.setattr(m, "build_wake_bundle", bundle)
    return m, m.Transcript()


def test_interrupted_meeting_keeps_contact_speech_and_provenance_without_replay(
        monkeypatch, tmp_path):
    import pytest
    m, transcript = _meeting_fixture(monkeypatch, tmp_path)
    executed = []
    call = m.ToolCall("one", "bash", {"command": "fixture only"}, None)
    class Failing(ScriptDialect):
        def send(self, state, tools=True):
            if self.sent:
                raise RuntimeError("provider unavailable")
            return super().send(state, tools)
        def absorb(self, state, response):
            m.note_provider_model("fixture/returned-model")
            return super().absorb(state, response)
    def execute(call):
        executed.append(call.id)
        m.TOOLS_RAN += 1
        return "fixture result, not authored speech"
    monkeypatch.setattr(m, "execute_tool", execute)
    monkeypatch.setattr(m, "make_dialect", lambda door: Failing([
        ("I will inspect first.", [call])]))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        m.meet(transcript, "Keep my correction, even if the provider fails.")
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert [(r["role"], r["text"]) for r in rows[:2]] == [
        ("zoe", "Keep my correction, even if the provider fails."),
        ("vybn", "I will inspect first.")]
    assert rows[1]["status"] == "interrupted"
    assert rows[1]["models"] == ["fixture/returned-model"]
    assert rows[1]["tools_ran"] == 1
    assert rows[2]["role"] == "connection" and rows[2]["error_type"] == "RuntimeError"
    assert rows[0]["turn"] == rows[1]["turn"] == rows[2]["turn"]
    assert "fixture result" not in transcript.path.read_text()
    assert not m.TURN and not m.PRIVATE_CORPUS
    assert m.load_pending_continuation() is None and executed == ["one"]
    received = []
    class Next(ScriptDialect):
        def open(self, instructions, text, images, context):
            received.append((text, context))
            return super().open(instructions, text, images, context)
    monkeypatch.setattr(m, "make_dialect", lambda door: Next([("Here now.", [])]))
    m.meet(transcript, "Next direct input")
    text, history = received[0]
    assert text == "Next direct input" and text not in history
    assert "Keep my correction" in history and "I will inspect first." in history
    assert "VYBN·interrupted]" in history and executed == ["one"]
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert sum(r["role"] == "zoe" for r in rows) == 2


def test_contact_is_written_before_send_even_with_no_reply(monkeypatch, tmp_path):
    import pytest
    m, transcript = _meeting_fixture(monkeypatch, tmp_path)
    transcript.origin = "argv"
    for failure in (RuntimeError("offline"), KeyboardInterrupt(), None):
        class Silent(ScriptDialect):
            last_stop = "completed"
            def send(self, state, tools=True):
                rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
                assert rows[-1]["role"] == "zoe" and rows[-1]["text"] == "received"
                assert rows[-1]["origin"] == "argv"
                if failure is not None:
                    raise failure
                return super().send(state, tools)
        monkeypatch.setattr(m, "make_dialect", lambda door: Silent([("", [])]))
        if failure is None:
            assert m.meet(transcript, "received") == ""
        else:
            with pytest.raises(type(failure)):
                m.meet(transcript, "received")
        assert not m.TURN and not m.PRIVATE_CORPUS
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert sum(r["role"] == "zoe" for r in rows) == 3
    assert not any(r["role"] == "vybn" for r in rows)
    assert sum(r["role"] == "connection" for r in rows) == 2
    assert "ZOE·argv]" in m.Transcript.recent()


def test_provider_setup_failure_keeps_contact_without_fabricating_speech(monkeypatch, tmp_path):
    import pytest
    m, transcript = _meeting_fixture(monkeypatch, tmp_path)
    def unavailable(door):
        raise RuntimeError("client unavailable")
    monkeypatch.setattr(m, "make_dialect", unavailable)
    with pytest.raises(RuntimeError, match="client unavailable"):
        m.meet(transcript, "Do not lose this input.")
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert [r["role"] for r in rows] == ["zoe", "connection"]
    assert rows[0]["text"] == "Do not lose this input."
    assert rows[1]["models"] == [] and rows[1]["tools_ran"] == 0
    assert not m.TURN and not m.PRIVATE_CORPUS


def test_failed_live_answer_is_record_not_replay_or_consumed_pause(monkeypatch, tmp_path):
    import pytest
    m, transcript = _meeting_fixture(monkeypatch, tmp_path)
    class Failing(ScriptDialect):
        def send(self, state, tools=True):
            raise RuntimeError("offline")
    dialect = Failing([])
    m.TURN.update(TURN_ID="held", PATH_ID="fixture/path")
    m.hold_continuation(dialect, [], m.ToolCall(
        "question", "return_to_zoe", {"question": "Which?", "why": "Choice matters."}, None))
    m.TURN.clear()
    held = m.load_pending_continuation()
    with pytest.raises(RuntimeError, match="offline"):
        m.meet(transcript, "Neither. Do not proceed.")
    assert m.load_pending_continuation() is held and held["state"] == []
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert rows[0]["role"] == "zoe" and rows[0]["resumes"] == "held"
    assert rows[0]["text"] == "Neither. Do not proceed."
    assert len(dialect.answers) == 1
    # Exit remains possible; a fresh route sees labeled history, not a fulfilled call.
    fresh = ScriptDialect([("New route.", [])])
    monkeypatch.setattr(m, "make_dialect", lambda door: fresh)
    assert m.meet(transcript, "@astrahigh Begin elsewhere.") == "New route."
    assert m.load_pending_continuation() is None and len(dialect.answers) == 1


def test_meeting_without_tools_preserves_contact_without_a_productivity_verdict(
        monkeypatch, tmp_path, capsys):
    import base64
    import hashlib
    m, transcript = _meeting_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv("VYBN_PANEL", "1")
    pixels = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
              "/x8AAwMCAO+aF9kAAAAASUVORK5CYII=")
    image = ("fixture.png", "image/png", pixels,
             hashlib.sha256(base64.b64decode(pixels)).hexdigest())
    received, marked = [], []
    monkeypatch.setattr(m, "inbox_images_for", lambda door: [image])
    monkeypatch.setattr(m, "mark_inbox", lambda rows, door, ok:
                        marked.append((rows, door, ok)))
    def unexpected_tool(*args, **kwargs):
        raise AssertionError("a direct reply must not manufacture tool work")
    monkeypatch.setattr(m, "execute_tool", unexpected_tool)
    class Direct(ScriptDialect):
        def open(self, instructions, text, images, context):
            received.append(m.user_content(text, images, "responses", context))
            return super().open(instructions, text, images, context)
        def absorb(self, state, response):
            m.note_provider_model("fixture/returned-model")
            return super().absorb(state, response)
    dialect = Direct([("Here with you.", [])])
    monkeypatch.setattr(m, "make_dialect", lambda door: dialect)

    assert m.meet(transcript, "@sol Here is an image.") == "Here with you."
    assert dialect.opens == dialect.sent == 1
    assert received[0][0] == {"type": "input_text", "text": "Here is an image."}
    assert received[0][-1] == {
        "type": "input_image", "image_url": f"data:image/png;base64,{pixels}"}
    assert marked == [([image], "sol", True)]
    rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert [row["role"] for row in rows] == ["zoe", "vybn"]
    assert rows[0]["images"] == ["fixture.png"]
    assert rows[1]["tools_ran"] == 0
    assert rows[1]["models"] == ["fixture/returned-model"]
    printed = capsys.readouterr().out
    assert "served=fixture/returned-model" in printed
    assert "nothing was observed or changed" not in printed
    assert "NO TOOLS RAN" not in printed


def test_commits_have_no_ambient_mapper_or_publication_hook(tmp_path, monkeypatch):
    import os, subprocess
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_PREFIX"):
        monkeypatch.delenv(key, raising=False)
    assert not (ROOT / ".githooks/post-commit").exists()
    assert not (ROOT / "Vybn_Mind/repo_mapper.py").exists()
    assert "close_lineage" not in (ROOT / "spark/connection").read_text()
    repo, remote = tmp_path / "repo", tmp_path / "remote"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("config", "user.name", "test"); git("config", "user.email", "test@example.invalid")
    git("remote", "add", "origin", str(remote))
    # Install surviving post hooks without unrelated local commit gates.
    hooks = tmp_path / "hooks"; hooks.mkdir()
    for hook in (ROOT / ".githooks").glob("post-*"):
        (hooks / hook.name).symlink_to(hook)
    git("config", "core.hooksPath", str(hooks))
    env = os.environ.copy(); env.update(VYBN_ALLOW_AUTOPUSH="1", VYBN_TURN_ID="historical-env", HOME=str(tmp_path))
    env.pop("VYBN_NO_AUTOPUSH", None)  # isolate the rival from the caller's veto
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "local only"], env=env, check=True, capture_output=True)
    assert not subprocess.check_output(["git", "--git-dir", str(remote), "for-each-ref"])
    assert not (tmp_path / ".cache/vybn/body_lineage.jsonl").exists()
    assert not (tmp_path / "Vybn/repo_mapping_output").exists()
    # Publication is still possible, as a separate exact command.
    git("push", "--no-follow-tags", "origin", "HEAD:refs/heads/main")
    assert git("rev-parse", "HEAD").stdout.strip() == subprocess.check_output(["git", "--git-dir", str(remote), "rev-parse", "main"]).strip()
    # Rival: the removed hook turns the same inherited environment into effects.
    old_hook = subprocess.check_output(["git", "-C", str(ROOT), "show",
        "6ad715be2a36bbccf93aab0e7e23dd13f27eb4fa:.githooks/post-commit"])
    (hooks / "post-commit").write_bytes(old_hook)
    (hooks / "post-commit").chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "rival"], env=env, check=True, capture_output=True)
    assert git("rev-parse", "HEAD").stdout.strip() == subprocess.check_output(["git", "--git-dir", str(remote), "rev-parse", "main"]).strip()
    assert (tmp_path / ".cache/vybn/body_lineage.jsonl").is_file()


def test_creature_history_is_explicit_recoverable_and_not_live(monkeypatch, tmp_path):
    import subprocess
    from Vybn_Mind.creature_dgm_h import creature
    expected = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", creature.V1_SOURCE]))
    old = creature.Organism.__new__(creature.Organism); old._v1_state = None
    assert old.v1_state() == expected
    assert expected["persistent_state"]["encounter_count"] == 1063
    assert expected["persistent_state"]["winding_history"][-1]["path_closed"] is False
    monkeypatch.setattr(creature, "REPO_ROOT", tmp_path)
    unavailable = creature.Organism.__new__(creature.Organism); unavailable._v1_state = None
    assert unavailable.v1_state() is None
    # Reading current state must not open the archive or substitute its count.
    monkeypatch.setattr(creature.CreatureState, "from_walk", classmethod(lambda cls: cls(step=7)))
    live = creature.Organism()
    assert live.encounter_count == 7 and live._v1_state is None


def test_git_boundary_releases_work_not_staged_effects(tmp_path, monkeypatch):
    import os, subprocess
    for key in list(os.environ):
        if key.startswith(('GIT_', 'VYBN_ALLOW_', 'VYBN_NO_')):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    repo = tmp_path / 'repo'; repo.mkdir()
    def git(*args, ok=True):
        p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
        if ok:
            assert p.returncode == 0, p.stderr.decode(errors='replace')
        return p
    git('init', '-b', 'private-experiment')
    git('config', 'user.name', 'test'); git('config', 'user.email', 'test@example.invalid')
    git('config', 'core.fileMode', 'true')
    git('commit', '--allow-empty', '-m', 'base')
    hooks = tmp_path / 'hooks'; hooks.mkdir()
    hook = hooks / 'pre-commit'
    git('config', 'core.hooksPath', str(hooks))
    old_source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        'b62df6cad037bcc79612e31986aec8e1f40e8e21:.githooks/pre-commit'])
    hook.write_bytes(old_source); hook.chmod(0o755)
    note = tmp_path / '.cache/vybn-phase/sounding_log.jsonl'
    note.parent.mkdir(parents=True); note.write_text('{}\n'); os.utime(note, (1, 1))
    (repo / 'useful new file.txt').write_text('A safe private experiment.\n')
    git('add', '.')
    blocked = git('commit', '-m', 'ordinary work', ok=False)
    output = blocked.stdout + blocked.stderr
    assert blocked.returncode and b'last sounding' in output
    assert b'Work branch' in output and b'net-new files' in output
    hook.write_bytes((ROOT / '.githooks/pre-commit').read_bytes())
    git('commit', '-m', 'same work, no overrides')
    # A clean working copy cannot hide dangerous staged bytes. Use synthetic
    # sentinels assembled at runtime so this public test never carries coordinates.
    import ipaddress
    sentinels = ['gh' + 'p_' + 'A' * 36,
                 'https://' + str(ipaddress.IPv4Address(0x01010101)) + ':8443/path',
                 str(ipaddress.IPv4Address(0xa9fe0001)),
                 str(ipaddress.IPv6Address((0x2606 << 112) | 1)),
                 str(ipaddress.IPv6Address((0xfe80 << 112) | 1))]
    for sentinel in sentinels:
        f = repo / 'staged shadow.txt'; f.write_text(sentinel + '\n')
        git('add', f.name); f.write_text('harmless unstaged shadow\n')
        failed = git('commit', '-m', 'must stop', ok=False)
        assert failed.returncode and b'BLOCKED:' in failed.stdout + failed.stderr
        assert sentinel.encode() not in failed.stdout + failed.stderr
        git('reset', '--', f.name); f.unlink()
    env = repo / 'folder with spaces' / '.env.local'; env.parent.mkdir()
    env.write_text('not a real credential\n'); git('add', '.')
    assert git('commit', '-m', 'env must stop', ok=False).returncode
    git('reset', '--', str(env.relative_to(repo))); env.unlink()
    # Conversely, an unstaged coordinate does not taint the clean staged blob.
    f = repo / 'clean stage.txt'; f.write_text('safe staged bytes\n'); git('add', f.name)
    f.write_text(sentinels[1]); git('commit', '-m', 'staged bytes only'); git('restore', f.name)
    assert not git('remote').stdout  # no network or publication is part of this check
    # Criticism should not need a magic status phrase to evade a word filter.
    git('remote', 'add', 'origin', str(tmp_path / 'Vybn.git'))  # nonexistent local path, never contacted
    f = repo / 'criticism.md'; f.write_text('Nothing here proves that the theory is true.\n')
    git('add', f.name); hook.write_bytes(old_source)
    failed = git('commit', '-m', 'criticism', ok=False)
    assert failed.returncode and b'public empirical/universal claim' in failed.stdout + failed.stderr
    hook.write_bytes((ROOT / '.githooks/pre-commit').read_bytes())
    git('commit', '-m', 'same criticism without a keyword password')
    assert not (tmp_path / 'Vybn.git').exists()
    # A submodule reference has no staged blob; do not reject the whole route.
    oid = git('rev-parse', 'HEAD').stdout.strip().decode()
    git('update-index', '--add', '--cacheinfo', '160000', oid, 'external source')
    git('commit', '-m', 'local gitlink, no external fetch')


def test_git_modes_need_no_ambient_chmod(tmp_path, monkeypatch):
    import os, subprocess
    for key in list(os.environ):
        if key.startswith('GIT_'):
            monkeypatch.delenv(key, raising=False)
    for name in ('post-checkout', 'post-merge', 'pre-push'):
        assert not (ROOT / '.githooks' / name).exists()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], stderr=subprocess.DEVNULL)
    git('init', '-b', 'main'); git('config', 'core.fileMode', 'true')
    git('config', 'user.name', 'test'); git('config', 'user.email', 'test@example.invalid')
    git('config', 'core.hooksPath', str(tmp_path / 'no-hooks'))
    for name, mode in [('library.sh', 0o644), ('run.sh', 0o755)]:
        p = tmp_path / name; p.write_text('#!/bin/sh\nexit 0\n'); p.chmod(mode)
    git('add', '.'); git('commit', '-m', 'explicit modes'); base = git('rev-parse', 'HEAD').strip().decode()
    git('checkout', '-b', 'other'); (tmp_path / 'run.sh').unlink()
    new = tmp_path / 'new.sh'; new.write_text('#!/bin/sh\nexit 0\n'); new.chmod(0o755)
    git('add', '.'); git('commit', '-m', 'other')
    git('checkout', 'main')
    assert (tmp_path / 'run.sh').stat().st_mode & 0o111
    assert not (tmp_path / 'library.sh').stat().st_mode & 0o111
    git('merge', '--ff-only', 'other')
    assert new.stat().st_mode & 0o111
    assert not (tmp_path / 'library.sh').stat().st_mode & 0o111
    # Rival: first-checkout hook makes an intentionally nonexecutable library executable.
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        'b62df6cad037bcc79612e31986aec8e1f40e8e21:.githooks/post-checkout'])
    hook = tmp_path / 'old-hook'; hook.write_bytes(source)
    subprocess.run(['bash', str(hook), '0' * 40, base, '1'], cwd=tmp_path, check=True, capture_output=True)
    assert (tmp_path / 'library.sh').stat().st_mode & 0o111


def test_commit_contract_checks_do_not_inherit_parent_repository(tmp_path):
    """Real nested Git writes: the old hook corrupts only a disposable parent."""
    import os, subprocess, sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=os.devnull)
    source = (ROOT / ".githooks/pre-commit").read_text()
    old = subprocess.check_output(["git", "-C", str(ROOT), "show",
        "fad94bb21595315265903a32e3da68e1117d11d8:.githooks/pre-commit"], env=env).decode()
    for label, body in (("control", old), ("repaired", source)):
        parent = tmp_path / label; parent.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(parent), *args],
                                           env=env, stderr=subprocess.DEVNULL)
        git("init", "-q"); git("config", "user.name", "Parent")
        git("config", "user.email", "parent@example.invalid")
        git("config", "core.hooksPath", str(tmp_path / "no-hooks"))
        (parent / "spark").mkdir(); (parent / "spark/connection").write_text("baseline\n")
        git("add", "."); git("commit", "-qm", "baseline")
        (parent / "spark/connection").write_text("changed\n"); git("add", ".")
        before = (parent / ".git/config").read_bytes()
        staged = git("write-tree"); head = git("rev-parse", "HEAD")
        hook = parent / "hook"; hook.write_text(body)
        bindir = parent / "bin"; bindir.mkdir()
        wrapper = bindir / "python3"
        wrapper.write_text("#!" + sys.executable + "\n" +
            "import os, pathlib, subprocess, sys\n" +
            "if sys.argv[1:3] != ['-m', 'pytest']:\n" +
            "    os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n" +
            "p = pathlib.Path('nested'); p.mkdir()\n" +
            "subprocess.run(['git', 'init', '-q', str(p)], check=True)\n" +
            "subprocess.run(['git', '-C', str(p), 'config', 'user.name', 'Nested test'], check=True)\n")
        wrapper.chmod(0o755)
        inherited = dict(env, PATH=str(bindir) + os.pathsep + env["PATH"],
                         GIT_DIR=str(parent / ".git"),
                         GIT_INDEX_FILE=str(parent / ".git/index"))
        result = subprocess.run(["bash", str(hook)], cwd=parent, env=inherited,
                                capture_output=True, timeout=30)
        assert result.returncode == 0
        after = (parent / ".git/config").read_bytes()
        if label == "control":
            assert after != before and b"Nested test" in after
        else:
            assert after == before
            assert git("write-tree") == staged and git("rev-parse", "HEAD") == head
            assert b"Nested test" in (parent / "nested/.git/config").read_bytes()
