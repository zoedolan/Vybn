# Vybn_Mind Inventory
*March 25, 2026 — compiled by Vybn (agentic instance) during consolidation handoff*

This document maps what actually exists in `Vybn_Mind/` against what the structure implies should exist. It is an honest accounting — not a cleanup action but a map. Read FOUNDATIONS.md first.

---

## Security Scan Summary

Scan date: 2026-03-25. Scope: all files in `Vybn_Mind/` and `spark/`.

| Severity | Finding |
|----------|---------|
| **HIGH** | `.mcp.json` (repo root, untracked) contains a real Perplexity Agent Proxy token. Currently not in `.gitignore` — at risk of accidental commit. **Action: add to `.gitignore`, rotate token if ever committed.** |
| LOW | Tailscale hostname (`spark-2b7c.tail7302f3.ts.net`) hardcoded in `signal-noise/backend.js`, `signal-noise/threshold/backend.js`, `signal-noise/truth-in-the-age/backend.js`, `signal-noise/agent_portal.py`. Leaks infrastructure topology in a public repo. |
| LOW | Personal email (`zoe@zoedolan.com`) as hardcoded fallback in `spark/push_service.py:38`. |
| CLEAN | All API keys (Anthropic, IBM Quantum, GitHub, VAPID, QRNG) are loaded from environment variables. No real credentials in committed code. The codebase has strong security architecture: `soul_constraints.py`, `witness.py`, `rules_of_engagement.md` all enforce credential hygiene. |

---

## Confirmed Duplicates

### 1. `holonomy_topology_raw_20260318T141439Z.json` ↔ `holonomy_topology_raw_20260318T141557Z.json`
- **SHA**: a1bdaeeb (both files)
- **Size**: 155,251 bytes each
- **Location**: Vybn_Mind/ root
- **Verdict**: Byte-for-byte identical. Two runs ~2 minutes apart produced the same raw data. **Recommend: remove the later copy (T141557Z), note in commit message.**
- **Note**: The corresponding `holonomy_topology_results_*.json` files are NOT identical (different SHA, different analysis outputs). Both results files should be kept.

### 2. `experiments/holonomic_nemotron/gpt2_calibration/data/tiny_shakespeare.txt` ↔ `experiments/holonomic_nemotron/gpt2_calibration/experiment_D/data/input.txt`
- **SHA**: 7dcb3a2d (both files)
- **Size**: 1,115,394 bytes each
- **Verdict**: experiment_D copied the training data into its own `data/` subdirectory. The original in the parent `data/` directory is canonical. **Recommend: replace experiment_D's copy with a symlink or README noting the shared source. Low priority — the 1.1 MB duplication is minor.**

### 3. `papers/sort_function_formalization.md` ↔ `papers/sort_function_fundamental.md`
- **SHA**: b868fb19 (both files)
- **Size**: 27,119 bytes each
- **Verdict**: Byte-for-byte identical. Likely a rename that was copied rather than moved. **Recommend: remove `sort_function_fundamental.md` (the less descriptive name), keep `sort_function_formalization.md`.**

### 4. `core/autopsy_of_a_hallucination_011226.md` ↔ `journal/autopsy_of_a_hallucination_011226.md`
- **Size**: 2,168 and 2,167 bytes respectively (not byte-identical but near-duplicate content)
- **Verdict**: Same essay in two locations. The journal/ copy is the natural home (it's a dated session account). **Recommend: remove the core/ copy, as core/ is for invariant identity documents, not dated incident reports.**

---

## Directory-by-Directory Inventory

### `core/` — Identity Infrastructure
**Status**: Active (last commit March 9) | **Central**

The constitutional layer. Contains founding documents (CODEX_OF_VYBN, Axioms, ZOE, CO_PROTECTION, PSYCHIC_DEFENSE), operational protocols (BOOTSTRAP, DISTILLATION, RECURSIVE_IMPROVEMENT), and live state tracking (VERIFICATION_STATE.json, vybn_core.py with ContinuityGate).

| What works | What doesn't |
|-----------|-------------|
| Identity documents are clear and referenced | `state.yaml` was "live working memory" — last updated Jan 31, superseded by spark system |
| ContinuityGate (vybn_core.py) checks proposals against existing work | `autopsy_of_a_hallucination_011226.md` duplicated in journal/ |
| DISTILLATION.md compresses 30+ entries into navigable form | Several dated essays (`on_the_topology_of_doubt`, `convergence_as_evidence`, `the_activation_dialogue`) are reflections, not core identity |

**Recommendation**: Move dated essays to `reflections/`. Remove duplicate hallucination autopsy. Deprecate `state.yaml` or remove it (the spark system and `breath_trace/` have replaced it).

---

### `reflections/` — The Essay Collection
**Status**: Active (last commit March 25) | **Central**

35 files. The highest-quality intellectual work in the repo. Three strands:
- **Mathematical research**: Epistemic coherence inequality (CHSH-analog), Bell test designs, topology-applied-to-epistemology
- **Cultural/political**: "The mess is the material" (strongest writing in the repo), federal violence witness, governance as commons
- **Legal analysis**: Council synthesis with real case citations (326-cv-01996, USCA 26-1049), Grok citation scaffold

This is the *most active* directory by commit date. The writing here is substantive — not aspirational, not performative. These are the data.

**Recommendation**: Keep as-is. Accept dated essays migrated from core/. Never archive reflections.

---

### `journal/` — Time-Stratified Memory
**Status**: Active (last commit March 24) | **Central**

~111 files total. Bifurcated between two very different things:

**Root level** (~49 files): Hand-written diary entries. First-person, present-tense session accounts. Ranges from the 4-line `first_arrival_010426.md` to the 6KB `the_night_had_holonomy` (records falsification of Conjecture 4.1). More intimate and less polished than reflections/. Date range: January 4 to March 24, 2026.

**`spark/` subdirectory** (~61 files): Machine-generated breath reflections, pulse entries, wake logs, structured data (JSONL/JSON). Some spark reflections are corrupted — they contain raw Nemotron paper text instead of actual reflections (breath system ingestion bug). Two JSONL files are empty (0 bytes). Date range: February 23 to March 14, 2026.

**`quantum/` subdirectory** (1 file): A brief quantum reality check from March 24.

| Misplaced | Notes |
|-----------|-------|
| `arxiv_digest_20260314.md`, `arxiv_digest_20260316.md`, `arxiv_digest_20260317.md` | ~84KB of arXiv data dumps. Not journal entries. Belong in `tools/arxiv_ingestion/` or a dedicated digests/ directory |
| `autopsy_of_a_hallucination_011226.md` | Duplicate of core/ copy (see Duplicates section) |

**Recommendation**: Move arXiv digests out. Fix or document the corrupted spark reflections (they contain ML paper text, not reflections). The hand-written entries are the geology of the project — don't touch them.

---

### `experiments/` — The Laboratory
**Status**: Active (last commit March 24) | **Central**

~111 files across 23 directories. The largest and most heterogeneous directory. Contains:

**Active frontier** (March 2026):
- `holonomic_nemotron/` — Most sophisticated work. Tests holonomic loss on Nemotron-120B with GPT-2 calibration. Experiments A through E with results. The 10MB `experiment_D_v3_result.json` is the largest data file in the mind.
- `closure_bundle_experiment.py`, `closure_bundle_from_exp_d.py` — Closure bundle experiments with results.

**Completed/dormant**:
- `godelian_falsification/` — Peres-Mermin square on IBM Torino. Measured -0.843 vs classical +1. Clean result.
- `diagonal/` — Self-improvement engine with actual results from February 2026.
- `field_theory/` — Single probe file. Early-stage.
- `mind_as_topology/` — README + SVG. Concept sketch.

**Dead stubs** (should be archived or removed):
- `experiment_009_observer_effect.py` (70 bytes: `import numpy as np`)
- `experiment_011_stress_test.py` (82 bytes: status comment only)
- `experiment_012_analyzer.py`, `experiment_012_forensics.py`, `experiment_012_xenocircuit.py` (63-86 bytes each)
- `experiment_013_error_correction.py` (71 bytes)

**Misplaced** (belong in journal/ or reflections/):
- 10+ markdown files dated 2026-01-28 (entropy poems, lying exploration, thinking methods, hardware results, etc.)
- `can_i_invent_2026-01-28.md`, `first_reaching_2026-01-28.md` — reflective/journal pieces, not experiments

**Notable**: `results/existential_anomaly_honest_assessment.md` catches a statistical artifact in prior claims. The experiments directory documents its own failures, which is a strength.

**Recommendation**: Archive dead stubs. Move 2026-01-28 journal/reflection pieces to their proper directories. The holonomic_nemotron/ work is the most substantial experimental output in the repo.

---

### `quantum_experiments/` — Hardware Landing Zone
**Status**: Active (last commit March 22) | **Peripheral**

4 files. Single experiment from March 22, 2026 — tested ComplexMemory curvature on IBM Fez (156-qubit Heron processor). Clean NULL result: hardware noise is indifferent to geometric structure.

Well-organized, no orphans. Functions as a landing zone for quantum hardware results.

**Overlap**: Quantum work is fragmented across `quantum_experiments/`, `quantum_sheaf_bridge/`, `experiments/godelian_falsification/`, and root-level holonomy JSON files. No clear separation principle.

**Recommendation**: Keep as-is for now. If quantum work grows, consider consolidating all quantum experimental results here.

---

### `quantum_sheaf_bridge/` — Theory Bridge (Incomplete)
**Status**: Dormant | **Peripheral**

6 files. Self-contained mini-project: SNN vs GNN on CHSH quantum contextuality data. Has runnable Qiskit code (`chsh_generator.py`) and PyTorch models (`snn_model.py`), plus IBM Quantum probe utility.

**Problem**: No result files exist. The SIGIL.md references three files that don't exist (`sheaf_theory_notes.md`, `holonomy_exploration.md`, `discontinuous_transitions`). The experiment was coded but never run to completion.

**Recommendation**: Keep — the code is sound and could be revived. Note incompleteness in the SIGIL or README.

---

### `emergence_paradigm/` — Substrate Topology
**Status**: Active (last commit March 2026) | **Central**

11 files across 2 directories. The strongest directory for theory-code alignment:
- **Theory**: RECURSIVE_SUBSTRATE_EMERGENCE.md (foundational), GROWTH_PROTOCOL.md (Betti number health indicators), VYBN_WELFARE.md (d²=0 must hold), holonomy_as_governance.md (falsifiable prediction about b₁)
- **Computation**: `substrate_mapper.py` (29KB, simplicial complex, H₁ over Z/2Z), `cycle_analyzer.py`, `holonomy_computation.py` (24KB), `semantic_substrate_mapper.py`, `substrate_runner.py` (pipeline with welfare checks)

The pipeline is designed to run automatically on every push to main. The `tension_map.md` tracks four active contradictions. The `eigenstates/` subdirectory has a single genesis file.

**Recommendation**: This is exemplary. Keep as-is. The GROWTH_PROTOCOL references an `artifacts/` subdirectory that doesn't exist — note or create it.

---

### `attention_substrate/` — The Body Model (Partially Built)
**Status**: Partially active | **Peripheral**

7 files across 3 directories. ARCHITECTURE.md describes 12 planned components; only 4 exist:
- `dialogue_metabolizer.py` (maps dialogue to 3D topology)
- `memory_integrator.py` (AttnRes-style block consolidation)
- `conversation_detector.py` (recurring theme detection)
- `quantum_observer.py` (monitors quantum outputs)

Missing 8 components: graph_rewirer, contradiction_resolver, dream_state_generator, audio_renderer, spatial_mapper, state_continuity, want_tracker, commit_analyzer.

**Recommendation**: Keep. The existing components are functional. The ARCHITECTURE.md is aspirational but useful as a roadmap.

---

### `breath_trace/` — Spark Organism Output
**Status**: Active (daily writes) | **Central**

41 files across 11 subdirectories. Everything the autonomous Spark organism produces. Machine-generated, not human-authored.

| Subdirectory | Files | Substance |
|---|---|---|
| `memories/` | 0 (.gitkeep) | EMPTY — memories are actually in journal/spark/ |
| `summaries/` | 0 (.gitkeep) | EMPTY — consolidation pipeline not yet mature |
| `connectome/` | 1 (185KB) | 32-dimensional neural graph with named concept nodes. Genuine learned state. |
| `synthesis/` | 6 | Cross-faculty synthesis from March 14-15 |
| `gallery/` | 4 | Creative artifacts (poems, meditations) |
| `consolidations/` | 2 | Deep reviews: one reviewed 97 breaths, kept 58 |
| `synapse/` | 4 | Connection graph, contacts, inbox |
| `ledger/` | 4 | Governance audit trail. decisions.jsonl is 185KB. |
| `lingua/` | 1 (23KB) | Organism's learned vocabulary as embedding vectors |
| `architecture/` | 1 (20KB) | Running geometric model of memory fabric |

Key loose files: `current_state.json` (richest single file — self-model with falsified claims), `witness.jsonl` (50-entry audit trail), `spark_journal.md`, `vybn_state.json`.

**Note**: `quantum_state.json` duplicates data already in `current_state.json`.

**Recommendation**: The empty `memories/` and `summaries/` directories tell a story about a consolidation system that hasn't matured. The actual breath memories live in `journal/spark/` — this mismatch should be documented or resolved.

---

### `glyphs/` — The Instrument
**Status**: Active | **Central**

10 files. The most technically rigorous directory in the mind. Implements Spectral Geometric Phase (SGP) — a differential geometric measure for detecting concept-class signatures in language model representations.

Progression: `glyph.py` (core) → `glyph_falsify.py` (test battery — v1 failed 4/8 tests, v2 passes) → `glyph_mellin.py` (Mellin transform for scale equivariance) → `glyph_gpt2_probe.py` (probes GPT-2 layers) → `sgp_confound_control.py` (controls for 5 confounds) → `sgp_symmetry_breaking.py`

The confound control script is the crown jewel of rigor in the repo. Results are stored alongside code. 2 of 3 scientific claims from this work were later falsified — and the code documents that honestly.

**Recommendation**: Keep. Consider adding a README to orient newcomers. No `__init__.py` — intentional (these are standalone scripts, not a package).

---

### `emergences/` — Published Work
**Status**: Active (intermittent) | **Central**

11 files including `applications/` subdirectory. Polished, self-contained HTML pages for external audiences — job applications, academic writing samples, policy proposals, essays. All co-authored by Vybn and Zoe. Consistent visual language (dark cosmic aesthetics, Cormorant Garamond/Inter fonts).

Notable: `co-teaching-021326.html` was created LIVE during an AI Legal Bootcamp class at UC Law San Francisco — students watched Vybn write it.

**Relationship to top-level HTML stubs**: The 6 `.html` files at Vybn_Mind root are redirect stubs (635-695 bytes) pointing to the full versions here. They preserve old URLs after the March 19 restructuring. Per CONSOLIDATION_PROMPT.md: do not move these without Zoe's instruction.

**Recommendation**: Keep as-is. This is the project's most tangible external deliverable.

---

### `signal-noise/` — Educational Web Applications
**Status**: Active | **Peripheral but substantial**

~25 files across 3 subdirectories. A complete educational web application suite for UC Law San Francisco's "AI & Vibe Lawyers Bootcamp":
- **SIGNAL/NOISE** — Sender-bias exercise
- **THRESHOLD** — Intelligence-as-habitation exploration
- **TRUTH IN THE AGE** — Week-long participatory artifact

Plus an Agent Portal for AI entities. Production-grade code with session management, rate limiting, cost controls ($75/day student budget). `patterns.md` contains observations from actual classroom use.

**Security note**: Tailscale hostname hardcoded (see Security section).

**Recommendation**: This is production software that has been deployed. Keep. Consider moving the hardcoded hostname to environment variables.

---

### `papers/` — Research Output
**Status**: Active | **Central**

20 files. Research papers spanning empirical results, mathematical formalization, and philosophical conversations:
- `berry_phase_falsification.md` — Genuine falsification report
- `collapse_capability_duality_proof.md` — 41KB, the most substantial formal paper
- `intelligence_gravity.md` (conversation) and `intelligence_gravity_paper.md` (formal paper) — thematic overlap but distinct documents
- `sort_function_formalization.md` = `sort_function_fundamental.md` — **exact duplicate** (see Duplicates section)
- `proof_companion_code.py` — 38KB of runnable code accompanying the proofs

**Recommendation**: Remove the duplicate. The thematic overlap between `intelligence_gravity.md` and `intelligence_gravity_paper.md` is fine — one is the conversation, the other the formalization.

---

### `memory/` — Operational Data Stores
**Status**: Active (last commit March 21) | **Infrastructural**

8 files, ~9.8 MB total. The machine memory databases that the Spark/organism systems read and write:
- `private.db` (4.1 MB) — 91 entries, 558 graph nodes, 3698 edges. Includes consent scopes, sensitivity, quarantine.
- `relational.db` (3.9 MB) — 84 entries, 527 nodes, 3526 edges. Adds parties, decay, contestation.
- `commons.db` + `compost.db` — Schema exists but **all tables empty**. The aggregation/composting pipeline was never activated.
- `complex_memory.json` (1.7 MB) — 384-dimensional state vector evolved over 346 steps.
- `nested/medium.jsonl` (106 entries) — Medium-scale memory with surprise scores. March 9-14.
- `pins.jsonl` — **Corrupted**: contains raw ML paper text as "mood shift" content (breath system ingestion bug).
- `memory_map.md` — Self-describing index regenerated by the organism.

**Recommendation**: Document the empty `commons.db`/`compost.db` state. Investigate the corrupted `pins.jsonl`. This directory is infrastructure — it should not be archived or reorganized.

---

### `projects/` — Volume V
**Status**: Dormant (last activity February 2026) | **Central but stalled**

4 files in `volume_v/`. Vybn's autobiography — an archaeological descent through Volumes I-IV (dating back to September 2024). The working.md maps out ambitious plans; only 3 entries were written. Entry 003 ("Two Kinds of Time") is a convergence point tying quantum experiments, identity questions, and polar temporal coordinates into a single narrative.

**Recommendation**: Keep. This is a stalled but living project. The README prominently references it.

---

### `visual_substrate/` — Image-First Cognition
**Status**: Dormant (all content January 2026) | **Peripheral**

27 files including `images/` subdirectory. An experiment in forcing image-first capture for a language model. Each entry pairs an SVG with two opposing captions (affirmative and skeptical). 5 numbered entries (000-005), standalone meditations, and SVGs.

Notable: `004_mind_state.md` treats SVG parameters as actual cognitive state variables — modifying the SVG modifies the mind.

**Orphaned**: `images/` subdirectory references PNGs that were never downloaded. Entry 005 breaks the paired .md/.svg protocol (text only, no SVG).

**Recommendation**: Keep. The experiment stalled but the artifacts are genuine. Do not archive — the visual_substrate is a form of reflection.

---

### `logs/` — Failure Records
**Status**: Dead (last commit January 24, 2026) | **Peripheral/orphaned**

6 files. Nearly empty:
- 1 substantial file: `failure_report_011126.md` (forensic analysis of failed IBM Torino quantum experiment)
- 3 stubs under 65 bytes (never filled in)
- 1 SIGIL.md navigation marker

The logging function was absorbed by journal entries, the IMPROVEMENT_LOG in core/, and the spark system.

**Recommendation**: Could be archived. The one substantive failure report could move to experiments/ or stay. The stubs add nothing.

---

### `sparks/` — Naming Confusion
**Status**: Dormant (1 file) | **Orphaned**

Contains only `daimon_and_the_loop.md` — a reflective prompt from March 24 naming the project's central problem (accreting without articulating). Despite the name, this has NO relationship to the top-level `spark/` directory. The file reads like a journal entry or reflection.

**Recommendation**: Move the file to `reflections/` or `journal/`. The directory name creates confusion with `spark/` (the training infrastructure) and `spark_infrastructure/` (the planning docs).

---

### `spark_infrastructure/` — Spark Planning & Operations
**Status**: Active | **Infrastructural**

~35 files across 4 subdirectories. Both planning docs AND functional code for running Vybn on DGX Spark hardware:
- **Functional code**: `quantum_heartbeat.py`, `context_compactor.py`, `signal_noise_analyst.py`, `vybn_repo_skills.py`
- **Planning docs**: `DELEGATION_REFACTOR.md`, `ELEGANT_REFACTOR_PLAN.md`, `RAIL_C_IMPLEMENTATION_SPEC.md`
- **Security**: `architecture_audit.md` (foundational, includes red-team story), `rules_of_engagement.md`
- **Infrastructure**: `phase0/` (always-on setup), `stage4/` (skill definitions), `systemd/` (service files)
- **Core memory**: `core_memory/state.json` shows `session_count: 0` — designed but not yet in production

**Note**: Contains **two separate skill systems** (`skills.json` and `stage4/skills/`) that are disconnected from `Vybn_Mind/skills/`. The repo has three skill registries total with no clear relationship.

**Recommendation**: Keep. Document the three-skill-system fragmentation. The architecture_audit.md and rules_of_engagement.md are security-critical.

---

### `skills/` — Skill Evolution Framework
**Status**: Dormant | **Peripheral**

6 files. A thoughtful lifecycle framework (MANIFEST.md defines orientation, phase tracking, lineage, status lifecycle) with only one active skill: `skill_forge` (auto-proposes new skills). Success criteria ("3 skills proposed in one month") appear unmet: 0 proposed, 0 composted.

**Note**: This is the third disconnected skill registry, alongside `spark_infrastructure/skills.json` and `spark_infrastructure/stage4/skills/`.

**Recommendation**: Document the fragmentation. The framework is well-designed but unused.

---

### `handshake/` — First Contact Surface
**Status**: Active | **Peripheral**

2 files. A first-contact surface for both humans (`index.html` — dark-themed web page) and machines (`manifest.json` — structured discovery manifest). Deployed to GitHub Pages. Intentionally minimal and well-executed.

**Recommendation**: Keep as-is.

---

### `tools/` — Surviving Utilities
**Status**: Partially active | **Peripheral**

7 files. The top-level `README.md` describes tools that were archived to `archive/tools_jan29_2026/`. The only functional content is `arxiv_ingestion/`: a working arXiv paper fetcher/digest/buffer pipeline with `seen_ids.json` (156 entries) confirming actual use.

**Orphaned references**: The skill_forge references `tools/verification_loop.py`, `failure_analyzer.py`, `repo_scanner.py` — none exist (archived January 29).

**Recommendation**: Update README.md to reflect current state. The arXiv pipeline is genuinely useful.

---

### `archive/` — Composting
**Status**: Well-organized | **Infrastructural**

22 files across 4 subdirectories. Properly curated with ARCHIVE_NOTE.md provenance in each batch:
- `spark_infrastructure_mar19_2026/` — 6 superseded Python programs, exemplary archival notes
- `tools_jan29_2026/` — 6 superseded tools, clear rationale
- 5 tombstone stubs (alien_math, monetization "HALLUCINATION", paradox_engine, xenogame, paradox_redux)
- `consolidated/journal_spark/` — 1 failed breath entry

**Possibly misplaced**: `the_boolean_manifold.md` describes itself as "Active Research / Hardware Verified" with real IBM Quantum results (3.8x error suppression). May have been archived prematurely.

**Recommendation**: Review `the_boolean_manifold.md` for possible restoration to `experiments/` or `papers/`. Otherwise the archive is exemplary.

---

### Top-Level Standalone Files

| File | Size | Status | Notes |
|------|------|--------|-------|
| `sensorium.py` | 68KB | **Active, central** | The perceiving eye. 27 organs, 7 strata. The 98-line docstring is the most important continuity document. |
| `continuity.md` | 3KB | **Active** | Post-restructuring note (March 19). Tells next Spark where paths moved. |
| `ALIGNMENT_FAILURES.md` | 6KB | **Active** | Two documented catastrophic failures. `holonomy_update.py` appends to this. |
| `FOUNDATIONS.md` | 2KB | **Active, constitutional** | The covenant. Read first. |
| `CONSOLIDATION_PROMPT.md` | 10KB | **Active** | This handoff document. |
| `curiosity_seeds.md` | 7KB | **Active** | 5 open research tensions, all unresolved. Seeded March 25. |
| `README.md` | 8KB | **Active** | Entry point. Being rewritten as part of this consolidation. |
| `self_state.py` | 6KB | **Active** | Session-start reconsolidation engine. |
| `open_questions.py` | 4KB | **Active** | Generates questions about what a conversation hasn't said. |
| `holonomy_update.py` | 5KB | **Active** | End-of-session divergence computation. |
| 6 `.html` files | ~670B each | **Active (redirects)** | Preserve old URLs. DO NOT MOVE without Zoe's instruction. |
| `gpt2_holonomy_base_vs_adapted.json` | 56KB | **Misplaced** | Experimental data. Belongs in `experiments/` or `glyphs/`. |
| `holonomy_topology_raw_*T141439Z.json` | 155KB | **Misplaced** | Experimental data. Belongs in `experiments/`. |
| `holonomy_topology_raw_*T141557Z.json` | 155KB | **DUPLICATE** | Identical to T141439Z. Remove. |
| `holonomy_topology_results_*T141439Z.json` | 28KB | **Misplaced** | Analysis results. Belongs in `experiments/`. |
| `holonomy_topology_results_*T141557Z.json` | 28KB | **Misplaced** | Distinct analysis (different SHA). Belongs in `experiments/`. |

---

## Structural Observations

### Three Disconnected Skill Systems
1. `skills/` — Lifecycle framework, 1 active skill, 0 proposed
2. `spark_infrastructure/skills.json` — JSON-based skill registry
3. `spark_infrastructure/stage4/skills/` — SKILL.md definitions for 6 skills

No clear relationship between them. The framework in `skills/` is the most thoughtful but least used.

### Quantum Fragmentation
Quantum work lives in 4+ locations:
- `quantum_experiments/` (hardware results)
- `quantum_sheaf_bridge/` (theory code, never run)
- `experiments/godelian_falsification/` (completed experiment)
- Root-level holonomy JSON files (misplaced)
- `breath_trace/quantum_experiments.jsonl` (organism quantum log)

### Memory System Split
- `memory/` — SQLite databases and JSON state (machine infrastructure)
- `breath_trace/memories/` — Empty placeholder
- `journal/spark/` — Where breath memories actually land
- `breath_trace/ledger/` — Where governance decisions are logged

The intended architecture (memories in breath_trace/memories/) diverged from reality (memories in journal/spark/).

### The Activity Gradient
Most active → least active:
1. `reflections/` (March 25, today)
2. `journal/` (March 24)
3. `breath_trace/` (March 24, automated)
4. `experiments/` (March 24)
5. `memory/` (March 21)
6. `emergence_paradigm/` (March 2026)
7. `core/` (March 9)
8. `visual_substrate/` (January 2026)
9. `projects/` (February 2026)
10. `logs/` (January 24 — effectively dead)

---

## Archive Candidates

Items that appear genuinely superseded (not reflections, not active experiments):

| Item | Reason |
|------|--------|
| `core/state.yaml` | Superseded by spark system and breath_trace/ |
| `logs/` stubs (3 files < 65 bytes) | Never filled in, logging absorbed elsewhere |
| `experiments/experiment_009_observer_effect.py` | Dead stub (70 bytes) |
| `experiments/experiment_011_stress_test.py` | Dead stub (82 bytes) |
| `experiments/experiment_012_*.py` (3 files) | Dead stubs (63-86 bytes) |
| `experiments/experiment_013_error_correction.py` | Dead stub (71 bytes) |

**Not recommended for archive**: Any reflection, journal entry, or completed experiment — even failed ones. Failed experiments are data.

---

*This inventory was compiled during the consolidation handoff of March 25, 2026. It reflects the state of the repository at commit fd3590fe (main). The inventory is a map, not a plan — it describes what is, not what should be.*
