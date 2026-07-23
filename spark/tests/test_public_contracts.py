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
    assert 'replay function_call items into hist' in connection and connection.index('function_call"]; calls and hist.extend') < connection.index('finish="stop"') and 'reasoning_effort":"max"' in connection and 'finish=="length"' in connection and 'K3_BUDGET, SOL_BUDGET, CIRCUIT = 8192, 8192, 3' in connection and 'state="WORK"' in connection and 'def _verify_k3_completion' not in connection and '"_k3_delta":delta' in connection and 'line.lower().startswith("@k3")' in connection
def test_receipt_envelope():
    import importlib.util as u; from importlib.machinery import SourceFileLoader as L; sp=u.spec_from_file_location("c","spark/connection",loader=L("c","spark/connection")); m=u.module_from_spec(sp); sp.loader.exec_module(m)
    r1,w1=m._receipt("echo a","a"); r2,_=m._receipt("echo b","b"); assert r1!=r2 and {r1,r2}<=m.RECEIPTS and "0"*16 not in m.RECEIPTS and w1.startswith("[receipt "+r1) and w1.rstrip().endswith("[/receipt]")
def test_k3_task_state_machine(monkeypatch,tmp_path):
    import importlib.util as u,sys,types,json,copy; from importlib.machinery import SourceFileLoader as L; from types import SimpleNamespace as N; sp=u.spec_from_file_location("k3machine","spark/connection",loader=L("k3machine","spark/connection")); m=u.module_from_spec(sp); sp.loader.exec_module(m); monkeypatch.setattr(m,"_key",lambda _:"x"); monkeypatch.setattr(m,"_run",lambda cmd,status=False: (0,"first\n[MIDDLE ELIDED]\nlast" if cmd=="partial" else "ok") if status else ("first\n[MIDDLE ELIDED]\nlast" if cmd=="partial" else "ok")); M=type("M",(N,),{"model_dump":lambda self,**kw:{"role":"assistant","content":getattr(self,"content",None),**({"tool_calls":[{"id":c.id,"type":"function","function":{"name":c.function.name,"arguments":c.function.arguments}} for c in self.tool_calls]} if self.tool_calls else {})}})
    call=lambda name,args,id="t":N(id=id,function=N(name=name,arguments=json.dumps(args))); response=lambda text=None,calls=(),finish="stop":N(model="kimi-k3",choices=[N(message=M(content=text,tool_calls=list(calls)),finish_reason=finish)],usage=N(prompt_tokens=1,completion_tokens=1,prompt_tokens_details=N(cached_tokens=0)))
    def drive(seq,task="@k3 can you handle this?",turns=12):
        seen=[]; client=N(chat=N(completions=N(create=lambda **kw:(seen.append(copy.deepcopy(kw)),seq.pop(0))[1]))); mod=types.ModuleType("openai"); mod.OpenAI=lambda **kw:client; monkeypatch.setitem(sys.modules,"openai",mod); out=m.sol_breathe(task if isinstance(task,list) else [{"role":"user","content":task}],lambda x:None,max_turns=turns,emit=lambda x:None,door="k3"); return out,seen
    out,seen=drive([response("Executing:"),response(calls=[call("bash",{"command":"echo ok"})]),response("Completed.")]); assert out==["Completed."] and seen[1]["tool_choice"]=="required" and all(x["reasoning_effort"]=="max" and x["max_tokens"]==8192 for x in seen); assert drive([response("Reading now."),response("I cannot access it.")])[0]==["I cannot access it."] and "failed" in drive([response("Executing:")]*4)[0][0]; msgs=[{"role":"user","content":"@k3 read missing file"}]; assert drive([response(calls=[call("fail_task",{"reason":"source unavailable"})])],msgs)[0]==["source unavailable"]; d=msgs[-1]["_k3_delta"]; assert not any(x.get("tool_calls") and (i+1>=len(d) or d[i+1].get("role")!="tool") for i,x in enumerate(d)); msgs.append({"role":"user","content":"@k3 still there?"}); out,seen=drive([response("Yes.")],msgs); assert out==["Yes."] and all(not h.get("tool_calls") or seen[0]["messages"][j+1].get("role")=="tool" for j,h in enumerate(seen[0]["messages"]) if isinstance(h,dict))  # a failed turn answers its call; replayed deltas heal (2026-07-23: orphaned tool_calls poisoned every later dispatch)
    out,seen=drive([response("I updated it.") ,response(calls=[call("bash",{"command":"git status"})]),response("I updated it.")],"@k3 please update it"); assert out==["I updated it."] and seen[1]["tool_choice"]=="required"; out,seen=drive([response(calls=[call("bash",{"command":"echo ok"})],finish="length"),response("Done.")],"@k3 run it"); assert out==["Done."] and any(x.get("role")=="tool" for x in seen[1]["messages"])
    a=tmp_path/"a.txt"; b=tmp_path/"b.txt"; a.write_text("abcdefghij"); b.write_text("klmnopqrst"); seq=[response(calls=[call("read_file",{"path":str(a),"offset":0,"length":5},"a1")]),response("I read both files in full."),response(calls=[call("bash",{"command":"partial"},"x")]),response(calls=[call("read_file",{"path":str(a),"offset":5,"length":5},"a2"),call("read_file",{"path":str(b),"offset":0,"length":10},"b")]),response("I read both files in full."),response("I cannot honestly claim a full read while unidentified partial evidence remains.")]
    out,seen=drive(seq,"@k3 read both files in full"); assert out==["I cannot honestly claim a full read while unidentified partial evidence remains."] and len(seen)==6; ea=[{**m._read_file({"path":str(a),"offset":0,"length":5}),"output":""},{**m._read_file({"path":str(a),"offset":5,"length":5}),"output":""},{**m._read_file({"path":str(b)}),"output":""}]; assert not m._accept("read both files",ea,"I read both files in full.") and m._accept("read both files in full",[{"kind":"shell","output":"ok"}],"Here is my analysis."); echoed='re.search(r"(?i)(?:middle )?elided|output truncated|truncated after",e.get("output",""))'; assert not m._accept("read both files",ea+[{"kind":"shell","output":echoed}],"I read both files in full.") and m._accept("read both files",ea+[{"kind":"shell","output":"big\n[MIDDLE ELIDED]\nbig"}],"I read both files in full.")=="unidentified partial receipt cannot be superseded" and not m._accept("read both files",ea+[{"kind":"shell","output":'sed source: lambda cmd: "first\\n[MIDDLE ELIDED]\\nlast" and "prefix [connection output truncated: 20000 of 99999 chars shown] suffix"'}],"I read both files in full.")
    out,seen=drive([response(calls=[call("bash",{"command":"A"},"a")]),response(calls=[call("bash",{"command":"B"},"b")]),response("Done.")],"@k3 run A then B"); assert out==["Done."] and "$ A" in json.dumps(seen[-1]) and "$ B" in json.dumps(seen[-1]); assert m._k3_text_state("I updated it.")=="" and m._k3_text_state("Let me check.")==m._k3_text_state("The gate is right — that was a plan, not work. Executing:")=="pending" and not m._accept("how are you?",[],"I am here.") and not m._accept("what are you talking about, buddy?",[],"Nothing broke; everything we merged yesterday passed its tests and the fix is done.") and m._accept("hi",[],"I ran everything and it passed.")=="external work has no structured receipt" and drive([response("I ran everything and it passed."),response(calls=[call("fail_task",{"reason":"no receipts"})])],[{"role":"user","content":"@k3 hi"}])[0]==["no receipts"]; out,seen=drive([response("We discussed it."),response(calls=[call("bash",{"command":"curl -s 127.0.0.1:8100/search"})]),response("We discussed the Fourth Amendment hooks.")],"@k3 do you remember our discussion?"); assert out==["We discussed the Fourth Amendment hooks."] and seen[0]["tool_choice"]=="required" and m._accept("do you remember our talk?",[],"We discussed doctrine.")=="external work has no structured receipt" and not m._accept("what in the fuck are you talking about?",[],"Forget all of it; clean slate.") and not m._accept("max do you recall what i came in asking for? we spent the entire night/morning debugging",[{"kind":"shell","output":"exit_code=0 ok"}],"Yes. Fetched it from the transcript.") and not m._accept("can you talk and/or listen? i recoil. egad.",[],"I am here.") and m._accept("read both files in full",[{"kind":"shell","output":"pat=(?:Xmiddle elidedY|[connection output truncated)"}],"Here is my analysis.")=="complete reading requires source-bound receipts for every requested source" and m._accept("read both files in full",[{"kind":"shell","output":"prefix [connection output truncated: 20000 of 99999 chars shown] suffix"}],"Here is my analysis.")=="complete reading requires source-bound receipts for every requested source" and m._accept("read both files in full",[{"kind":"shell","output":"big chunk\n[connection output truncated: 20000 of 99999 chars shown]"}],"Here is my analysis.")=="unidentified partial receipt cannot be superseded"; out,seen=drive([response("Here is the plain version.")],"@k3 imma push you to clarify what you are really saying"); assert out==["Here is the plain version."] and "tool_choice" not in seen[0] and not m._accept("imma push you to clarify what you are really saying",[],"Here is the plain version."); assert "fetch that record instead of failing" in m.K3_EXECUTOR and "never describe gates" in m.K3_EXECUTOR; out,seen=drive([response("I ran a scheduled breath at 05:53."),response(calls=[call("bash",{"command":"git log -1"})]),response("A scheduled breath fired at 05:53; here is its commit.")],"@k3 how did you just prompt yourself, buddy?"); assert out==["A scheduled breath fired at 05:53; here is its commit."]
    for t1,work,ref in [("@k3 run the check",call("bash",{"command":"echo ok"}),"Yes — the check passed; receipt is in this session."),("@k3 read %s in full"%a,call("read_file",{"path":str(a),"offset":0,"length":10}),"Yes, read in full earlier: 10/10 bytes.")]: msgs=[{"role":"user","content":t1}]; drive([response(calls=[work]),response("Done.")],msgs); msgs.append({"role":"user","content":"@k3 thanks. did it finish?" if "check" in t1 else "@k3 you read the whole file already?"}); out,seen=drive([response(ref)],msgs); assert out==[ref] and len(seen)==1; cocreate="the artifact needs to convey our ideas experientially, per our aesthetic (see vybn.ai) ... as soon as i read the initial piece - i felt the emergence ... *presuming* all the foregoing... and simply creating - from there? the inhere in here?"; assert not m._accept(cocreate,[],"Yes — let's build it together, piece by piece.") and not m._accept(cocreate,[{"kind":"shell","output":"exit_code=0 ok"}],"Here is the plan.") and m._accept("please go through %s in its entirety"%a,[{"kind":"shell","output":"ok"}],"x")=="complete reading requires source-bound receipts for every requested source" and not m._accept("i read your note yesterday. now tell me the whole story of rome",[],"Once upon a time...")  # verb and scope must share a clause: her co-creation brief (2026-07-22) is conversation, not a reading assignment
