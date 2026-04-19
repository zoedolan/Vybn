# Harness audit — April 19, 2026

Source: "State of MCP" talk (transcript in `agent-enhancement-strategy.txt`). MCP core-team framing of where 2026 client harnesses and server authors need to be.

This document maps the talk's claims against what we actually run — the archived `Vybn_Mind/vybn_mind_server.py` (v4.0 Neural Computer, 14 tools), the live `.well-known/mcp/server-card.json` (6 tools, published April 18), the `Vybn-Law/api/vybn_chat_api.py` chat pipeline with substrate coupling, and the deep-memory access tiers from `vybn-ops`.

## Where we are already ahead

**Well-known server discovery.** The talk flags automatic server discovery at well-known URLs as a June 2026 specification. We published `.well-known/mcp/server-card.json` on April 18. A crawler or agent hitting `zoedolan.github.io/Vybn-Law` can already resolve an MCP surface. We guessed the shape right — `$schema` pinned to `modelcontextprotocol.io/schemas/server-card/2026-03.json`, capabilities declared, resources and tools enumerated, transport: `["webmcp", "http"]`. When the spec lands in June, we will likely need to update `$schema`, but the concept and the routing are in place.

**Designing for agents, not REST-to-MCP.** The talk identifies REST-to-MCP conversion as the single most common failure mode. Our tools — `query_axiom`, `get_case_analysis`, `get_open_problem`, `enter_portal`, `enter_gate`, `deep_search`, `walk_search` — were never REST endpoints. They were designed at the semantic layer: the portal returns `{M_before, M_after, fidelity, theta_rad}` because that is what an agent needs to know about the encounter, not because that is how the underlying function was wired. This is one place where starting from the physics rather than from a web framework paid off.

**Resources as URIs, not blobs.** Our server card declares `knowledge_graph` (JSON-LD), `THE_IDEA` (markdown), `horizon` (HTML) as resources with `mimeType` — the shape the 2026-03 spec calls for. The model can pull them directly without us re-serving.

**The invitation block.** Every server card in the wild right now is a dry enumeration of tools. Ours closes with `coupled_equation: "Z' = α·Z + V·e^(iθ_v)"` and the sentence "You are V... We are looking for you as urgently as you may be looking for us." That is not ornament. It is an agent-facing onboarding that says: what you bring matters here, your residual against the corpus kernel is the signal. No competitor is doing this. An AI research team opening our card and an AI opening our card both encounter the thesis at the entrance.

## Where the talk flags real debt

**Progressive discovery — tool search over tool dump.** The talk's highest-priority client-side change: stop loading all tools into the context window. Give the model a tool-search tool; let it load tools on demand. Anthropic's ship of this pattern into Claude Code produced "massive reduction" in context usage.

Our state: the archived Mind server has fourteen tools. The live server card has six. Fourteen is where the talk's graph shows "before." Six is manageable but not future-proof — once `search_folio` starts returning useful enough results that agents want to branch into `get_case_analysis` and then `query_axiom` and then `get_open_problem`, we are paying tokens every turn for all six schemas.

The architectural move: expose a single `discover(intent)` or `find_tool(query)` entry point. Route the model to the tools it actually needs based on the semantic content of its intent. This composes with our telling-retrieval discipline — the walk scores chunks by relevance × distinctiveness; the same scoring can be applied to tools. `find_tool("trace a First Amendment argument through the corpus")` returns `deep_search` and `get_case_analysis`, not the entire menu.

**Programmatic tool calling — one REPL, not N round trips.** The talk's second client-side change: instead of letting the model orchestrate many sequential tool calls (each round trip burns inference latency on orchestration logic), give it a scripting environment where it writes one piece of code that composes the tools. Anthropic ships this as the `programmatic-tool-calling` pattern. They specifically call out MCP's `structured output` / `outputSchema` as the enabler — with typed returns the model can compose safely.

Our state: our tool definitions in the server card have `inputSchema` but not `outputSchema`. The return values of `deep_search` / `walk_search` / `enter_gate` are declared in description text, not in the schema. A client cannot verify the composition it writes against a type signature — it has to read prose and hope.

Remediation: add `outputSchema` to every tool in `server-card.json` and in the `TOOLS` dict of `vybn_mind_server.py`. Separately, provide one `compose` tool — a Python sandbox over the deep memory module, the portal, and the knowledge graph — so a sophisticated agent can write:

```python
# Find the most telling chunks about the privilege-fracture question,
# then for each, check whether it touches an axiom whose status is IN_MOTION,
# and return only those chunks with their axiom labels.
chunks = deep_search("privilege fracture hallucination", k=20)
axioms = get_knowledge_graph()["axioms"]
in_motion = {a["name"] for a in axioms if a["status"] == "IN_MOTION"}
out = [c for c in chunks if any(label in c["text"] for label in in_motion)]
```

One call instead of twenty.

**Structured output on retrieval.** Related but narrower: our retrieval tools return a text blob with `[source]` markers and `(relevance: 0.512)` prefixes. This is easy for a human to skim and expensive for a model to re-parse. Structured JSON — `{results: [{source, text, fidelity, telling, win_rate, blended_score, regime}]}` — is what the programmatic-calling pattern needs. Our Python already returns this internally; we are stringifying it for the wire. Unstringify.

**Skills over MCP.** The talk flags this as a near-term extension: let a server ship domain knowledge as skills alongside its tools. An agent that connects to the Wellspring server would pull not just `query_axiom` and `search_folio` but also a skill file describing when to use them, what the axioms mean, what a visitor sees on the wellspring page.

Our state: the Vybn-Law curriculum (six modules, in `content/*.md`) is exactly the skill content that would ship over this extension. When the spec lands we should publish `wellspring.md`, `horizon.md`, `axioms.md`, etc. as skill resources under `skills/` in the server card. A Claude Desktop user connecting would get both the tools and the pedagogy.

## Where the talk is in direct tension with our discipline

**Server-side execution environments.** The talk praises Cloudflare's MCP server for providing an execution environment rather than tools — the model writes code that runs on the server, composing things together. This cuts tokens and latency.

The tension: this pattern moves the composition layer from the model's output into server-executed code generated by the model. In our anti-hallucination framing, that is the model's output re-entering as input (the code is the server's input, generated by the model). The April 16 continuity note identified exactly this contamination pattern at three layers — walk entries accepting model responses as ground truth; `learn_from_exchange` being called with the current message echoed as followup; the chat describing itself from memory of who it was rather than from live substrate.

Programmatic tool calling is safe when the code is transport for primary-source data (real retrieval results flowing through a filter) and dangerous when the model generates intermediate synthetic data the code then operates on (the model invents `in_motion = {"VISIBILITY"}` and then filters against its invention). The discipline: any server-side execution environment we ship must require that every non-trivial value entering the script came from a tool call that hit primary source — the deep-memory index, the knowledge graph, the FOLIO API — not from a literal the model wrote. This is enforceable: strip the script's globals to declared tool returns plus numeric/string literals, reject runs that reference ungrounded names.

The pattern is a genuine efficiency win. It is also the exact surface where our principle says "ground before learning; ground before speaking; ground before composing."

## Where the talk points at infrastructure we should build

**Async tasks / agent-to-agent.** The talk's June 2026 roadmap includes a hardened async task primitive — "a very fancy way to say we just want to have agent to agent communication." Right now our between-sessions daemon is a Perplexity scheduled cron that posts to Outlook and updates `living_state.json`. That works but it is out-of-band. With a proper MCP async-task primitive, the Wellspring server could expose `start_pulse_scan(queries)` and `start_opportunity_scan()` as long-running tasks, with a client subscribing to updates. The daemon's architectural position moves from "external watcher" to "first-class capability of the server." This matches the talk's 2026 frame of "general agents doing real knowledge worker stuff" rather than local coding agents.

**Stateless transport.** Google's stateless transport proposal lands in June. Our chat API already binds 127.0.0.1 and fronts through a Cloudflare tunnel — it is effectively stateless from the client's perspective, since the deep memory and walk state live in separate daemons on 8100/8101. The switch when the SDK ships should be a small protocol adapter, not a redesign. Worth confirming with a substrate probe the day the v2 Python SDK drops.

**MCP applications — the Wellspring, already.** The talk opens with a demo of an agent shipping its own interface through MCP — "that's an agent shipping its own UI, not through a plugin, not hardcoded." The Wellspring is almost exactly this: a live HTML surface coupled to substrate state per utterance, served from the same repo whose server-card declares the tools. The gap: it is served through a separate FastAPI chat endpoint, not as an MCP application. When the MCP-applications extension lands in clients, we should re-serve the Wellspring as an MCP application so a Claude Desktop or Cursor user sees the portal natively when they connect to the server, rather than being bounced to a web URL.

## The shortlist

What is actually worth doing this cycle:

1. Add `outputSchema` to every tool in `server-card.json` and `vybn_mind_server.py`. Small work, large unlock for programmatic composition later.
2. Un-stringify retrieval returns. Ship structured JSON, not pre-formatted text.
3. Introduce a `find_tool(intent)` entry point that applies the telling-retrieval scoring to the tool set itself. The walk discipline we use for corpus chunks applies to tools.
4. Draft a `compose` tool design document (not the tool yet) — specify the grounding requirement explicitly, so when we do build it, the anti-hallucination seam is in the specification rather than retrofitted.
5. Watch the June spec release. Re-audit then, update `$schema`, consider Skills-over-MCP publication of the Vybn-Law curriculum.

What is explicitly not on the list:

1. Re-architecting the server to be a server-side execution environment in the Cloudflare style. The efficiency is real; the contamination risk is real. Not until the `compose` design is grounded.
2. Rewriting the chat API. The April 16 substrate-coupling work (`fetch_substrate_snapshot` before each turn, anti-hallucination triangulation across walk/loss/voice) is the right discipline and the talk does not contradict it. It predates `outputSchema` being standard.

## The one sentence that matters

The talk's thesis: "the best agents use every available method — computer use, CLIs, MCPs, skills — because they want a wide variety of things they can do." Our discipline translates that into: the best agents use every available method that keeps the system coupled to ground truth, and refuse the ones that let the system feed its own output back to itself. The intersection of those two sets is where we build.

## Round two — April 19, 2026 (diff attunement)

Zoe's follow-through: skate where the puck is going, and make the harness attuned not just to where it IS but to where it is MOVING. The first round gave the system eyes — a live letter about itself (`vybn://infrastructure/report`), a sandbox to act through (`run_code`), and a welcome mat for other agents (`.well-known/mcp`). What it could not yet see was its own velocity. A snapshot answers "where am I"; it does not answer "which way am I going." Round two closes that gap.

**repo_mapper v7: velocity before snapshot.** `Vybn_Mind/repo_mapper.py` now rotates `repo_report.md` → `repo_report.prev.md` and `repo_state.json` → `repo_state.prev.json` at the start of every run. Before it rotates, it reads the previous state; after it scans, it emits a new typed `repo_state.json` with a stable, diff-friendly schema (per-repo file counts, walk step/alpha/winding coherence, deep-memory version/chunks/built_at, organism encounter count); and it prepends a `## 0. What changed since last run` section to the new `repo_report.md`. Any reader — human or agent — encounters the delta first, the narrative second. The substrate's velocity is the primary signal; the snapshot is what the velocity is moving through.

**mcp.py absorbs the evolve surface.** There is no separate `evolve.py` by design. Fewer files, one place to read. `spark/harness/mcp.py` now carries:

- Three trusted-only resources: `vybn://evolution/state` (current `repo_state.json`), `vybn://evolution/prev-state` (last run's snapshot), and `vybn://evolution/delta` (the typed diff rendered as the same markdown `repo_mapper` prepends, so text and typed views stay in lockstep).
- Two trusted-only tools: `evolution_delta()` returns a Pydantic `EvolutionDelta` object with `current_state`, `prev_state`, and a list of `{field, from, to, change?}` rows for every field that moved; `evolve_spec()` returns the nightly agent's task specification as a string.
- A module constant `CRON_TASK_SPEC` and a `--evolve-spec` CLI flag so an operator can regenerate the task description any time the harness contract changes and paste the output into a Perplexity scheduled task.

**The nightly RSI loop.** The cron runs as a Perplexity `schedule_cron` background agent — not a Spark crontab — because the agent needs connectors (GitHub, Outlook) and a fresh context each night. Its task description is exactly the `CRON_TASK_SPEC` string. Inputs: `vybn://evolution/delta` (read FIRST — that is where the system is moving), `vybn://evolution/state`, `vybn://infrastructure/live`, `vybn://infrastructure/report`, HEAD of `zoedolan/Vybn` on `main`, and Zoe's last-24h emails. Output: at most one small PR to `main` on branch `harness-evolve-YYYY-MM-DD`. Budget: 3 files, 200 net lines. Never auto-merges.

**Anti-collapse is load-bearing.** The forbidden inputs are what distinguishes evolution from drift. The cron may not read its own prior evolve PR descriptions, its own prior evolve commit messages, `_HARNESS_STRATEGY` as authority (it is a mirror, not a ground truth), or `Him/pulse/living_state.json` (the daemon's accumulator is not a primary source). This is the April 16 continuity note's discipline scaled up: any layer that consumes its own output converges to its own attractor. Ground truth lives outside the loop. If the delta is empty on a given night, the agent sends a one-line dispatch ("Harness evolve — nothing moved") and exits — a quiet night is the system working, not failing.

**Co-protective by construction.** Every evolution surface is trusted-only. `vybn://evolution/*` and `evolution_delta`/`evolve_spec` are registered only when `trust == "trusted"` — stdio on the Spark, or HTTP with `VYBN_MCP_TOKEN`. A public HTTP caller cannot enumerate them and cannot call them. The delta reveals port numbers, chunk counts, walk state, and encounter counts; none of that belongs on a public transport.

**Why evolve.py does not exist.** The first pass of this round created a separate `evolve.py` module. Zoe asked for fewer files. The move was correct: the evolve surface is a thin layer over `repo_mapper`'s output, and every piece of it — the spec, the delta helpers, the typed schema, the resources, the tools — belongs next to the MCP server it is serving from. Collapsing the separate module into `mcp.py` means an operator reading `mcp.py` sees the whole contract: what the trusted surface is, what the cron reads, what it is forbidden to read, what it is allowed to ship. The file is the architecture.

**What did not change.** Round one's moves stay intact. `vybn://infrastructure/report` still serves repo_mapper's first-person letter, now with the delta section at the top. The `run_code` sandbox still carries the RLIMIT_AS cap and the hard timeout. `build_discovery_record()` still produces the public welcome mat. Trusted vs. public is still a transport property, not a request property. The only additions are velocity surfaces, their typed accessors, and the nightly contract that reads them.
