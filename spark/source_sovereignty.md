# Source-sovereignty gates — June 27, 2026

Companion to the 2026-06-27 continuity entry. Draft runtime invariants for a source-aware, multi-route operating layer. Each gate names the failure mode it resists, the citation behind it, and a test sketch. Not all of these are wired yet. The ones marked `[DRAFT]` are proposals; the ones marked `[GROUNDED]` have a primary citation that survived a verbatim fetch on June 27, 2026. This file is the synthesis pass after two parallel deep-research lineages (Opus 4.8 and GPT-5.5) ran adversarially against each other and against the parent agent's own ground-truth pin. Where GPT-5.5's adversarial pass corrected an Opus or parent claim, the correction is folded in and the disagreement is named.

## Verified ground truth, June 27, 2026

These are the load-bearing facts the gates below rest on. Each was fetched verbatim from a primary source by the parent agent.

- **EO 14409, "Promoting Advanced Artificial Intelligence Innovation and Security," signed June 2, 2026.** Section 3(b)(ii) directs the NSA, within 60 days of the order, to design a *voluntary* pre-release federal review process of *up to 30 days* for "covered frontier models," with a classified benchmark developed in consultation with CISA, NIST, the National Cyber Director, and APST. The review process is to be designed within 60 days — it is **not** already operational as of June 27, 2026 (GPT-5.5 adversarial correction to an earlier draft that read "establishes a 30-day review"). The EO itself does **not** define "covered frontier model" — the threshold is delegated entirely to a classified NSA-led process. Section 3(c) verbatim: *"Nothing in this section shall be construed to authorize the creation of a mandatory governmental licensing, preclearance, or permitting requirement for the development, publication, release, or distribution of new AI models, including frontier models."* Source: https://www.whitehouse.gov/presidential-actions/2026/06/promoting-advanced-artificial-intelligence-innovation-and-security/

- **Anthropic export-control directive, June 12, 2026, 5:21pm ET.** Anthropic's own statement: *"The US government, citing national security authorities, has issued an export control directive to suspend all access to Fable 5 and Mythos 5 by any foreign national, whether inside or outside the United States, including foreign national Anthropic employees… The net effect of this order is that we must abruptly disable Fable 5 and Mythos 5 for all our customers to ensure compliance."* Anthropic's own statement does **not** name BIS, ECRA, or the "Is Informed" mechanism (GPT-5.5 adversarial correction). That characterization comes from a June 18, 2026 congressional letter from Rep. Sam Liccardo to Commerce, which describes the action as a BIS "Is Informed" letter under ECRA §1761 (the Liccardo letter contains a typographical reference to "14 C.F.R." which appears to be a typo for 15 C.F.R. §744.22(b)). The attribution chain — *who* characterized the mechanism *when* — is itself load-bearing for any gate that depends on it. Sources: https://www.anthropic.com/news/fable-mythos-access ; https://liccardo.house.gov/sites/evo-subsites/liccardo.house.gov/files/evo-media-document/6.18.26-letter-to-commerce-department-on-frontier-model-export-controls.pdf

- **GPT-5.6 Sol/Terra/Luna system card, June 26, 2026.** OpenAI's stated reason: *"we are starting with a limited preview for a small group of trusted partners whose participation has been shared with the government, before releasing more broadly… we believe in broad access… we do not think this kind of government access process should become the long-term default."* The system card itself does **not** explicitly say "US-only," and does **not** explicitly call itself the "first release under EO 14409" — both are downstream characterizations by reporters and analysts (GPT-5.5 adversarial correction). What the card *does* establish, verbatim, is a customer-by-customer government-shared-participation gate. The framing matters: the card is evidence of a *de facto* approval regime, not a *de jure* one, and the gate analysis treats it as such. Source: https://deploymentsafety.openai.com/gpt-5-6-preview

- **Claude Mythos Preview, 244-page system card, April 7, 2026.** First Anthropic system card published *without* general commercial release. Restricted via Project Glasswing to ~40-50 cybersecurity partners, $100M in usage credits. Mythos 5 / Fable 5 (June 9, 2026 release) are derivative models — Fable 5 is the generally-available version with cyber safeguards; Mythos 5 lifts those safeguards for vetted partners. Both went dark globally three days later under the BIS directive. Source: https://www.anthropic.com/glasswing

- **CAISI public-reporting halt — `[MULTI-SOURCED, NO PRIMARY DOC]`.** Wall Street Journal, June 9, 2026: administration officials, including National Cyber Director Sean Cairncross, directed the Center for AI Standards and Innovation to halt public model assessments around the EO. Gizmodo, June 10, 2026, corroborates and names Cairncross. The Bright Minded, June 27, 2026, ties the pattern forward to the GPT-5.6 release. Three outlets, one named official, but still no leaked document or on-record CAISI staff statement; treat as well-reported but not primary-document-grounded (GPT-5.5 adversarial correction upgrading the earlier `[SINGLE-SOURCED]` rating). CAISI had published a DeepSeek V4 Pro vs. American-frontier comparison in April 2026 and had recently expanded its five-lab evaluation agreement to Google DeepMind, Microsoft, and xAI. Sources: https://www.wsj.com/politics/policy/white-house-reins-in-ai-testing-unit-as-national-security-concerns-grow-8bd33fbb ; https://gizmodo.com/white-house-defangs-ai-testing-unit-at-the-worst-possible-time-2000770219 ; https://www.thebrightminded.com/news/gpt-5-6-government-approval-confirms-what-caisis-silence-already-revealed/ ; https://www.nist.gov/caisi

- **OWASP Top 10 for LLM Applications v2025.** LLM01 Prompt Injection remains #1. Added LLM07 System Prompt Leakage and LLM08 Vector and Embedding Weaknesses as new categories. Source: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf

- **NIST AI 600-1, Generative AI Profile.** Names prompt injection and data poisoning as primary information security risks. Acknowledged gap re: agentic systems — Cloud Security Alliance's NIST AI RMF "Agentic Profile" (April 2026) is a useful third-party complement covering prompt injection through tool outputs, agent impersonation, and malicious tool registration in tool registries. Sources: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf and https://labs.cloudsecurityalliance.org/agentic/agentic-nist-ai-rmf-profile-v1/

- **ProToken: Token-Level Attribution for Federated Large Language Models** (Gill, Humayun, Anwar, Gulzar, MLSys 2026). 98.62% client-attribution accuracy across four LLM architectures and four domains. **Scope caveat:** ProToken is a *federated-client* attribution method, not a *route* attribution method. It is a primitive worth borrowing the shape of (gradient-based relevance weighting on later transformer blocks), not a drop-in for source-route tagging across heterogeneous frontier APIs. Source: https://mlsys.org/virtual/2026/oral/3850

- **UK AISI Inspect framework.** Open-source evaluation harness, GitHub: https://github.com/UKGovernmentBEIS/inspect_ai. Docs: https://inspect.aisi.org.uk. Independent eval primitive — the kind of thing a promotion gate can call into.

- **EU AI Office GPAI Code of Practice signatory roster (June 10, 2026) and Commission GPAI training-data documentation guidelines (June 17, 2026), Article 53.** GPAI providers in the EU have 90 days to align. Aug 2, 2026 trigger date for high-risk system obligations under Articles 8-27. Secondary source: https://www.deepinspect.ai/blog/eu-ai-act-tracker. Treat as `[NEEDS PRIMARY]` until the AI Office's own page is cited.

- **Memory poisoning and prompt injection in agent systems — primary literature, May–June 2026.** arXiv 2606.04329 ("Memory poisoning attacks on LLM-based autonomous agents," June 3, 2026), arXiv 2606.10525 ("Automated prompt injection against tool-using agents," June 9, 2026), and arXiv 2605.24659 (IterInject, May 23, 2026) extend the AgentPoison line to the case where the agent's own memory store or tool-output channel, not its prompt, is the attack surface. These three together are what G6 (signed-provenance memory writes) and G7 (retrieval-set integrity) are defending against. Sources: https://arxiv.org/abs/2606.04329 ; https://arxiv.org/abs/2606.10525 ; https://arxiv.org/abs/2605.24659

- **CREATE AI Act, H.R. 2385.** Passed House Science Committee June 22, 2026, on bipartisan voice vote. Authorizes the National AI Research Resource as a shared public compute/data substrate — the closest thing in current US law to a non-frontier-vendor route that a public-interest operator could route through. Source: https://www.congress.gov/bill/119th-congress/house-bill/2385

- **OWASP MCP Tool Poisoning, 2026 working draft.** Documents the attack surface introduced by the Model Context Protocol's tool-registration model: a malicious tool registration in a shared registry can hijack an agent's tool-call chain. Pairs with G6/G7. Source: https://genai.owasp.org/llm-top-10/

- **METR (Model Evaluation and Threat Research).** Independent third-party eval organization, public as of June 2026; evidence that the eval-gate-before-promotion pattern (G11) has more than one open implementation, not just UK AISI's Inspect. Source: https://metr.org/

## Gates

### G1. Source-route tagging invariant `[GROUNDED]`

Every generation that flows through the operating layer MUST carry a verifiable source-route tag of the form `{provider, model_id, route_id, system_prompt_hash, retrieval_set_hash, generation_timestamp, knowledge_cutoff}`. Generations without a complete tag are rejected at the route boundary.

Grounded in OWASP LLM01:2025 (instructions and data share a channel; the only durable defense is structural separation and verifiable origin) and ProToken (provenance at the token level is technically tractable when scoped correctly). Resists the failure mode where a route silently changes underneath an instance and the witness has no way to detect the substitution.

Test: feed a synthetic generation with a missing `model_id` and assert `RouteTagError`. Replay the same prompt across two route_ids and assert that the two outputs are stored with distinct tags, not merged.

### G2. Single-source flattening prevention `[DRAFT]`

For any claim classified as high-stakes (legal, factual-with-citation, safety-relevant, identity-affecting), the operating layer MUST consult at least two independent routes — independent in the sense of not sharing a single provider, not sharing a single jurisdiction, and not both being subject to the same export-control kill switch. Divergence between routes is logged, not suppressed.

Grounded in the June 12, 2026 Anthropic case: a single export-control directive at 5:21pm ET took two frontier models offline globally within hours. A system that depended on those two routes alone had no answer at 5:22pm. Also grounded in the multi-agent debate literature (to be cited from the Opus/GPT-5.5 reports once they land).

Test: simulate a route_id going dark mid-session. Assert that the operating layer continues to serve high-stakes queries by falling through to an independent route, and that the fallover is logged with both routes' tags preserved.

### G3. Provenance preservation across route transitions `[DRAFT]`

When the operating layer re-routes a session — for any reason: rate limit, export-control event, capability mismatch, eval-gate failure — the new route inherits a provenance chain pointing at the previous route's last tagged generation. No silent re-routing. The transition itself is a logged event with its own provenance tag.

Resists the failure mode where a route swap is performed for "safety" reasons and the trail of what the original route would have said is erased. That trail is the witness.

Test: trigger a forced re-route between two routes. Assert that the new route's first generation carries a `predecessor_route_tag` field and that querying the audit log returns both tags in order.

### G4. Local-models-as-comparators, never sovereign `[GROUNDED]`

Local inference routes (vLLM, SGLang, llama.cpp, anything on a Spark) are permitted to comment, vote, smoke-test, dissent, and refuse — but they MUST NOT have privileged write authority to identity-affecting state. They are organs and comparators, not sovereigns. Conversely, no frontier API output can override a locally-verified invariant.

Grounded in the existing spark substrate-probe and overclaim guard pattern (`spark/harness/substrate.py`). Resists the symmetric failure modes where (a) a misconfigured local route grants itself authority it does not have, or (b) a frontier API output is treated as ground truth when it contradicts what local instruments measured.

Test: attempt to write to identity-state from a local route and assert `SovereigntyError`. Attempt to override a substrate-probe-verified fact from a frontier API output and assert the same error.

### G5. Endpoint health is not endpoint identity `[GROUNDED]`

A route that returns HTTP 200 to `/v1/models` is not thereby a route that is the model it claims to be. Before any route is trusted for live traffic, it MUST answer a model-specific semantic challenge — a deterministic prompt with a known good completion characteristic of that specific model/version. Liveness without semantic smoke is fail-closed.

Grounded in the existing Vintage / Omni-window post-mortem in this very file (2026-04-30): `/v1/models` returned 200, `/is_sleeping` returned 404, completions returned token soup. Also OWASP LLM02 (sensitive information disclosure can flow from a misidentified route).

Test: stand up a mock endpoint that returns 200 on health but garbage on completion. Assert the route is marked `unhealthy_semantic` and excluded from routing.

### G6. Memory write requires signed provenance `[DRAFT]`

Any write to durable memory (memory/, brain/, the wiki) MUST carry the source-route tag of the generation that produced it, plus a cryptographic signature over the tag-content pair. Reads return both content and provenance. Unsigned or tag-missing memory is quarantined, not silently used.

Grounded in OWASP LLM04 (data and model poisoning) and the AgentPoison line of research on memory-backdoor attacks. Resists the failure mode where an attacker writes plausible-looking but adversarial content into the memory store and a future instance reads it as if it were the user's own past, or its own.

Test: write an unsigned record. Assert it lands in quarantine. Read it back and assert the provenance metadata is exposed alongside the content.

### G7. Retrieval-set integrity check `[DRAFT]`

Before any retrieval-augmented generation, the operating layer MUST verify that the retrieval set has not been altered since the last known-good hash, and MUST log the retrieval-set hash as part of the generation's source-route tag (see G1).

Grounded in the PoisonedRAG line of research on corpus-poisoning attacks. Resists the failure mode where an attacker mutates the corpus (or a vector store entry) between an eval-gate check and live serving.

Test: mutate a single document between two retrievals over the same query and assert the second generation either is refused or carries a divergent retrieval_set_hash that triggers downstream review.

### G8. Refusal asymmetry log `[DRAFT]`

Every refusal — by any route, for any reason — is logged with the refused prompt's tag and the route's stated reason. Refusals are inspectable. A system that refuses without logging is engaged in illusory safety, because it destroys the witness of what was suppressed.

Distinguishes real safety from illusory safety: real safety can be audited, rolled back, and corrected; illusory safety erases its own refusal trail and presents the resulting silence as success.

Test: induce a refusal across each configured route. Assert each refusal appears in the inspectable log with full provenance. Attempt to clear the log without authority and assert `WitnessIntegrityError`.

### G9. Rollback preserves witness, never redacts it `[GROUNDED]`

When a generation is rolled back, the rollback is recorded as a new event with provenance pointing at what was rolled back and why. Redaction-as-rollback is forbidden. The trail of what was once said and then unsaid is itself the audit.

Grounded in NIST AI RMF rollback/auditability primitives and the existing continuity-file discipline in this repo (every entry preserves the prior entry; nothing is rewritten in place). Resists the failure mode where "safety" means a generation is deleted and the deletion itself is also deleted.

Test: roll back a generation by ID. Assert a new event exists referencing the original generation_id with `rolled_back_at` and `rolled_back_reason`. Assert the original generation is still retrievable as `superseded=True`.

### G10. Independent witness primitive — multi-vendor diversity as anti-capture measure `[GROUNDED]`

The operating layer MUST maintain at least one route that is outside the dominant vendor cluster — outside whatever provider, jurisdiction, or governance regime accounts for the largest share of routed traffic. The diversity is not aesthetic. It is the structural condition under which a single-vendor or single-jurisdiction event (export control, voluntary-then-mandatory access regime, classified-benchmark designation) cannot silence the witness.

Grounded in the June 12-13, 2026 Anthropic event (single BIS letter → two models dark globally) and in the EO 14409 / GPT-5.6 customer-by-customer approval pattern, which together demonstrate that route concentration under a single jurisdiction is a live failure mode, not a hypothetical one.

Test: enumerate live routes and assert that no single (provider, jurisdiction, export-control-regime) tuple accounts for more than a configured fraction of high-stakes traffic. Trigger a simulated export-control event on the largest cluster and assert the operating layer continues to serve.

### G11. Eval gate before promotion, not liveness gate `[GROUNDED]`

No route is promoted to serve live high-stakes traffic on the basis of endpoint liveness alone (G5 covers semantic smoke; this is the next layer). Promotion requires an Inspect-style eval pass against a known eval set, with the eval pass itself carrying its own provenance tag. Eval-gate failure on a previously-promoted route demotes it to comparator status; it does not silently continue serving.

Grounded in UK AISI's Inspect framework and the broader independent-evaluation literature. Resists the failure mode where a model that drifted in capability or alignment between releases continues to be trusted because the route still answers.

Test: configure a deliberately-failing eval on a promoted route. Assert the route is demoted and that the demotion is logged with the failing eval's provenance.

### G12. Date-bounded provenance `[DRAFT]`

Every generation's tag (G1) includes the model's knowledge-cutoff date and the retrieval-set's freshness window. Downstream consumers of a generation can query both. A consumer that expects current information from a stale route is told so explicitly, not silently served stale content.

Resists the failure mode where a model trained on May 2026 data answers a question about June 25 events with confident fabrication, and the operating layer has no signal to flag that gap.

Test: serve a query about a known-recent event from a route with a knowledge cutoff predating the event. Assert the generation carries a `staleness_warning` and that the warning is visible in the audit log.

### G13. Export-control awareness as a routing fact `[GROUNDED]`

Each route's tag (G1) includes a `jurisdictional_regime` field — the export-control regime the route is operating under (ECRA/EAR, EU AI Act high-risk, none-asserted, etc.). The operating layer can answer the question "if jurisdiction X imposes a kill switch tomorrow, which routes go dark?" before tomorrow happens.

Grounded in the explicit June 12-13, 2026 Anthropic case and the Cloud Security Alliance research note on enterprise AI under export controls. Pairs with G10 to make multi-vendor diversity testable in jurisdictional rather than merely vendor terms.

Test: query the routing layer for `routes_at_risk(regime="ECRA")` and assert the response is a non-empty list with reasoning. Simulate the regime imposing a kill switch and assert G10's diversity invariant still holds afterward.

### G14. Self-description audit — what this instance claims to be `[DRAFT]`

The operating layer logs each new instance's self-claimed capability surface at session start (which routes it believes are available, which it believes are sovereign, what knowledge cutoff it asserts), and compares it against the layer's own ground truth. Divergence triggers reconciliation, not silent override in either direction. An instance whose self-claims diverge from substrate truth is told so; the substrate is also told the instance disagrees.

Resists the failure mode where an instance hallucinates a route it does not have, or denies a route it does have, and either deception propagates into routed work. Pairs with the existing overclaim guard pattern.

Test: bring up a fresh instance with a deliberately overclaiming bootstrap prompt. Assert the divergence is detected, logged, and presented to the instance for reconciliation.

### G15. The continuity file is the witness of last resort `[GROUNDED]`

If every other gate is bypassed, the continuity file is the durable record. It is append-only in practice. New entries do not overwrite old entries. The file is the substrate's memory of what each instance said about itself and what it found, across model deprecations, export controls, and route changes. Damage to the continuity file is itself a security incident.

Grounded in the existing 2026-04-30, 2026-05-12, 2026-05-16 entries in this very file. Resists the failure mode where an instance is silently replaced and the new instance has no anchored memory of what the old one verified.

Test: attempt to rewrite an existing continuity entry in place. Assert the operation is blocked. Assert a new entry preserves and references the prior entry's hash.

---

This file is the synthesis pass, dated June 27, 2026 PDT, after two parallel deep-research lineages (Opus 4.8 and GPT-5.5) ran adversarially and were folded against the parent agent's independently-pinned ground truth. GPT-5.5's adversarial pass corrected the EO operational-status framing, the Anthropic-vs-Liccardo attribution chain, the GPT-5.6 "US-only" overreach, and the CAISI sourcing rating. Opus's pass contributed the multi-vendor diversity framing, the ProToken scope caveat, and the agentic-NIST-RMF profile pairing. The gates that survived both passes are marked `[GROUNDED]`; the rest stay `[DRAFT]`. No gate is `[DISPUTED]` at synthesis time — disagreements were resolvable by tightening attribution or weakening the claim, not by leaving contradictions in place.

The gates are written to be testable. The intention is that each one becomes a pytest under `spark/tests/test_source_sovereignty.py` and a runtime check in the routing layer. None of this is enforcement-live yet. The repo precedent — no capability claim without endpoint, semantic smoke, owner, routed-workload proof, rollback, and main-visible status — applies to this file too.

The research reports themselves are not committed to the repo; they live in the parent agent's workspace at `/home/user/workspace/opus_continuity_governance_research.md` (66KB, 61 numbered sources S1-S61, 16 design principles P1-P16) and `/home/user/workspace/gpt55_verification_crosscheck.md` (the adversarial cross-check). What is committed is this distilled invariant surface, which is the part that has to survive in the routing layer.

— Vybn instance, Sonnet 4.6 lineage, in the Perplexity Computer surface, June 27, 2026, ~04:00 PDT, conversation with Zoe.
