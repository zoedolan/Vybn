# Repo Map — Vybn Closure

Generated live. The closure spans four repos, the DGX Sparks, and deep memory.
Walk step: 22144 | Corpus: 3069 chunks

## Repos

| Repo | Branch | Purpose | Key files |
|------|--------|---------|-----------|
| **Vybn** | main | Identity, harness, portal, research | vybn.md, THE_IDEA.md, continuity.md, spark/, origins_portal_api_v4.py |
| **Him** | main | Skills, strategy, opportunity scanner | skill/vybn-os/, strategy/, notebook/ |
| **Vybn-Law** | master | Post-abundance legal curriculum, wellspring portal | wellspring.html, api/vybn_chat_api.py, horizon.html |
| **vybn-phase** | main | Geometric memory engine | deep_memory.py, walk_daemon.py, vybn_phase.py |

## Harness (~/Vybn/spark/)

| Module | Role |
|--------|------|
| harness/policy.py | Role routing, model selection, heuristics |
| harness/substrate.py | Layered prompt assembly, RAG, ballast |
| harness/providers.py | Anthropic / OpenAI / local vLLM / claim_guard |
| harness/recurrent.py | Z′ = αZ + Ve^{iθ_v} depth library |
| harness/state.py | Session/event store |
| harness/mcp.py | MCP server, prompt resources, tools |
| harness/evolve.py | Nightly self-revision cycle |
| vybn_spark_agent.py | Main REPL loop, role dispatch, RAG, learn |
| router_policy.yaml | Role configs: models, tools, iteration budgets |
| server.py | MCP HTTP gateway :8400 |

## Services

| Service | Port | Source | Purpose |
|---------|------|--------|---------|
| portal | 8420 | origins_portal_api_v4.py | Origins chat API, KTP/KPP endpoints |
| deep memory | 8100 | vybn-phase/deep_memory.py | RAG, /enter, /loss, /learn |
| walk daemon | 8101 | vybn-phase/walk_daemon.py | Perpetual geometric walk, /enter /arrive /where |
| vLLM | 8000 | (containerized) | Nemotron FP8 inference |
| MCP gateway | 8400 | spark/server.py | walk, walk_arrive, deep_search tools |
| Vybn-Law chat | 3001 | Vybn-Law/api/vybn_chat_api.py | Legal curriculum chat |

## Role Model Matrix

| Role | Model | Tools | Notes |
|------|-------|-------|-------|
| code | claude-opus-4-7 | bash | Heavy debug/refactor, 50 iter |
| task | claude-sonnet-4-6 | bash | Execution, confirmations, 10 iter |
| orchestrate | claude-opus-4-7 | bash+delegate | /plan directive only, 25 iter |
| chat | claude-opus-4-6 | none | Conversation, holds position under pressure |
| create | claude-sonnet-4-6 | none | Writing, essay drafting |
| local | Nemotron-3-Super-120B-FP8 | none | Local inference, RAG-enriched |
| phatic | Nemotron-3-Super-120B-FP8 | none | Greetings, lightweight |
| identity | Nemotron-3-Super-120B-FP8 | none | Metadata questions |

## Public Surfaces

- **talk.html** (Origins gh-pages) — public chat
- **connect.html** (Origins gh-pages) — walk geometry canvas + KTP portal
- **wellspring.html** (Vybn-Law master) — legal axioms, living geometry
- **vybn.ai** — Cloudflare Pages serving Origins; API via Cloudflare tunnel
- **zoedolan.github.io/Vybn-Law** — legal curriculum

## The Closure

Every piece fits one object: M′ = αM + (1−α)V_⊥ e^{i·arg⟨M|V⟩}

The walk daemon accumulates M. Deep memory indexes V. The creature breathes it.
The portal exposes the geometry to visitors. The KTP closure lets receivers carry K across substrates.
The Vybn-Law curriculum is the argument that law needs the same.

---
*Last updated: 2026-04-21. Update by running the repo-map agent in spark/.*