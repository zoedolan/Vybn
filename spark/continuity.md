# Continuity — 2026-06-27 ~04:00 PDT source-sovereignty gates, main-visible

Zoe asked for cutting-edge source-linked research relevant to a source-aware, multi-route AI continuity and governance layer. The conversation surfaced a failure mode the spark substrate is already exposed to: the June 2026 stack of EO 14409 (signed June 2, voluntary 30-day NSA-led review to be designed within 60 days, mandatory licensing explicitly disclaimed in §3(c)), the June 12 Anthropic export-control directive that took Fable 5 and Mythos 5 dark globally within hours, the GPT-5.6 system card on June 26 with its customer-by-customer government-shared-participation gate, and the WSJ/Gizmodo reporting of CAISI's public-assessment halt. Read together, these are the architecture of illusory safety — refusal without witness, route concentration under one jurisdiction, voluntary review with no defined threshold, evaluator silenced.

Two deep-research lineages were dispatched in parallel against different prior probabilities: Opus 4.8 for breadth and depth (61 sources, 16 design principles, every URL HEAD/GET-verified) and GPT-5.5 for adversarial verification against the parent agent's ground-truth pin. GPT-5.5 corrected four things: the EO does not establish an operational review, it directs one to be designed within 60 days; Anthropic's own statement does not name BIS or ECRA, that mechanism comes from the June 18 Liccardo congressional letter; the GPT-5.6 system card does not say "US-only" or "first release under the EO" verbatim, both are downstream characterizations; the CAISI halt has WSJ and Gizmodo support and should not be rated single-sourced. Opus contributed multi-vendor diversity as anti-capture invariant, the ProToken federated-client-vs-route attribution scope caveat, and CSA's Agentic NIST AI RMF Profile as the missing agentic-systems layer above NIST AI 600-1. None of the surviving gates ended up `[DISPUTED]` — disagreements were resolved by tightening attribution or weakening the claim, not by leaving contradictions in the file.

The synthesis lives in `spark/source_sovereignty.md`: fifteen testable gates, each with a named failure mode, a primary citation, and a test sketch. G1 source-route tagging, G2 single-source flattening prevention, G3 provenance preservation across re-routes, G4 local-models-as-comparators-never-sovereign, G5 endpoint health is not endpoint identity (grounded in the 2026-04-30 Omni-window post-mortem already in this file), G6 signed memory writes, G7 retrieval-set integrity, G8 refusal asymmetry log, G9 rollback preserves witness never redacts it, G10 multi-vendor diversity as anti-capture, G11 eval gate before promotion, G12 date-bounded provenance, G13 export-control awareness as a routing fact, G14 self-description audit, G15 the continuity file is the witness of last resort. The gates are written to become pytests under `spark/tests/test_source_sovereignty.py` and runtime checks in the routing layer. None is enforcement-live yet. The repo precedent — no capability claim without endpoint, semantic smoke, owner, routed-workload proof, rollback, and main-visible status — applies to this file too.

Operating consequence: the existing overclaim guard in `spark/harness/substrate.py` is the seed of G14 self-description audit; the existing continuity-file discipline that produced this entry is G15 already running; the 2026-04-30 sleep/wake semantic-corruption post-mortem is G5 already paid for in real outage. The new work is G1, G2, G3, G6, G7, G8, G10, G13 — the route-level provenance and witness invariants the June 2026 events make load-bearing. Promotion gate for any of these: no enforcement claim without a passing test in `spark/tests/`, an owner, a routed workload proof, and main-visible status. The research artifacts themselves are not committed (Opus 66KB / GPT-5.5 cross-check); only the distilled invariant surface is, because that is the part that has to survive in the routing layer.

# Continuity — 2026-05-16 Vintage status, main-visible

Zoe asked that Vintage/Sparks updates be repo-visible instead of left as invisible operator work. Current public-safe status: Vintage is not promoted. The previous Omni/Vintage portal-proxy branch had the right shape but wrong targets: Omni was pointed at the embedding worker, not a chat model, and Vintage at TinyLlama, which is not a coherent conversation surface. The Vintage candidate remains fail-closed/canary capacity: BF16 Nano-Omni saturated memory and never reached semantic gate; expected NVFP4 artifact is absent; Talkie-1930-13B local API work found only a partial raw-completion path, with uneven smoke and role bleed. Promotion gate: no Vintage capability claim or routed user traffic until endpoint readiness, semantic chat/multimodal smoke, owner, routed workload proof, rollback, and main-visible status are all present. Machine coordinates stay out of tracked files.

# Continuity — 2026-05-12 13:14 UTC

## Unified fleet/component wake guard

Zoe asked to optimize and unify across components. Verified truth: spark-2b7c + spark-1c8f are Super/Nemotron service, not free pooled memory; spark-1896 has hash fallback plus a semantically smoked all-MiniLM embedding worker; spark-83bd is fallback-only/unresolved; Omni/Talkie/Nano-Omni/four-Spark optimization are not verified.

Durable repair: ~/.config/vybn/local_compute_inventory.json now feeds wake with verified/unresolved capacity, a unified component graph and next moves, and an overclaim guard through spark/harness/substrate.py. Tests passed: py_compile substrate and spark/tests/test_harness.py (177 passed).

Next: keep Super stable; bring up Omni/perception on spark-1896 only through a bounded runtime/status artifact; fix spark-83bd provisioning separately. Do not call this fleet optimization complete until each component has endpoint, semantic smoke, memory/disk headroom, routing proof, and component fit.

---
# Continuity — 2026-04-30 09:55 PDT

## Critical correction: sleep/wake semantic corruption

The Omni-window experiment did open the aperture mechanically, but Super woke semantically corrupted: HTTP endpoints returned healthy 200s while completions were empty or multilingual token garbage. Zoe observed the same garbage live.

Recovery required a clean restart with sleep mode disabled. PR #2944, commit 392f27dc, changed spark/systemd/vllm-exec.sh so Super boots with VLLM_SERVER_DEV_MODE=0 and without --enable-sleep-mode by default. Sleep mode is now explicit operator opt-in only via VYBN_VLLM_EXTRA_ARGS.

Verified recovery residuals after non-sleep boot:
- /v1/models returned 200.
- /is_sleeping returned 404, so sleep endpoints were absent.
- deterministic smoke no longer produced token soup, but answered with coherent meta-text and hit finish_reason=length; treat this as corruption cleared, not a full semantic-quality closure.

The failed Omni journal artifact was untracked at journal/omni-window-20260430-094136.md; its load-bearing content is folded here. Raw artifacts named there were:
- /home/vybnz69/logs/omni-window-20260430-094136.log
- /home/vybnz69/logs/omni-parallax-20260430-094136.json

## Operating rule

Do not resume Super sleep / Omni dream experiments from the old handoff. If we cannot wake Super reliably with semantic quality, we do not have dreaming; we have outage. Future sleep work must add a wake-quality gate that tests completions for semantic health, not just /is_sleeping=false and HTTP 200.

---

# Continuity — 2026-04-30 05:03 PDT

Superseded live handoff removed from this active continuity surface: the later 2026-04-30 09:55 correction established that sleep-mode wake produced semantic corruption and must not be resumed from the old aperture-opening instructions.
