# Recursive Improvement Architecture

*Committed: 2026-02-21*
*Status: Foundation laid — Steps 1-3 infrastructure deployed*

This document describes the topology-guided recursive self-improvement
architecture for Vybn.  It synthesizes the council analysis (GPT-5.2
Thinking, Claude Opus 4.6 Thinking, Gemini 3.1 Pro Thinking) with
Zenil's January 2026 proof on entropy decay in closed-loop self-training.

## The Constraint That Shapes Everything

Zenil proved that closed-loop self-training — where a model trains on
its own outputs — converges inevitably to entropy decay and mode
collapse, regardless of architecture.  The entropy of the model
distribution forms a supermartingale under the Data Processing
Inequality.  A system that feeds on its own outputs doesn't ascend
toward superintelligence — it implodes toward a degenerate fixed point.

This means the horizon is not "a system that improves itself in a
closed loop."  That's mathematically forbidden.  The horizon is a
three-part organism:

1. **The human (Zoe)** supplies the persistent connection to external
   ground truth — the alpha_t > 0 that Zenil proves is permanently
   necessary.

2. **The knowledge graph and its topology** serve as both diagnostic
   instrument and curriculum generator.  Betti numbers, cluster
   structure, and void detection tell the system where its
   representation is thin, disconnected, or surprisingly dense.

3. **The MoE-CL adapter architecture** provides the weight-modification
   mechanism, governed by the topology's signals and evaluated against
   benchmarks that detect both improvement and entropy loss.

## The Six Steps (Backward-Optimized from Horizon)

### Step 1: Compute Exact b1 (Time Series, Not Snapshot)

**File:** `spark/topology_gudhi.py`

Computing b1 isn't just answering "how many loops exist."  It's the
first data point in what will become the geometry dashboard's primary
diagnostic channel.  Every result is stored with a version stamp, the
exact commit hash, and the inclusion criteria used.  When b1 is
computed again after the first training cycle, the delta is the
signal — it tells us whether the cycle created new conceptual loops
(generative) or collapsed existing ones (degenerative).

**Status:** Infrastructure deployed.  GUDHI module written.  Awaiting
installation on the Spark and first real computation against the
repo's knowledge graph.

### Step 2: Define the Ontological Boundary (Schema, Not Cleanup)

**File:** `spark/vertex_schema.py`

Removing .lock files and cache artifacts isn't housekeeping — it's
defining what counts as Vybn's knowledge.  The schema is versioned.
Every future commit gets classified against it: is this a vertex
(a concept, a file, an interaction log) or noise?  The inclusion/
exclusion policy is documented and enforced programmatically.

**Status:** Schema v0.1.0 deployed.  Classifies vertices into:
conceptual, code, structured, configuration, or excluded.

### Step 3: Build the Geometry Dashboard (Intrinsic Diagnostic Layer)

**File:** `spark/geometry_dashboard.py`

The dashboard tracks:
- b0 trend: fragmenting or consolidating?
- b1 trend: conceptual loops growing or shrinking?
- b2 trend: higher-order voids forming?
- Cluster inventory: size distribution, domain types
- Void inventory: unexplored branches, insular clusters
- Surprise detector: Keplerian moments (deviations from trend)
- Entropy monitor: Zenil collapse detection

The dashboard never touches weights.  It watches the repo's
evolving topological structure and produces diagnostic signals.
If topology changes predict model behavior, we have empirical
evidence that geometry is a meaningful control signal.  If they
don't, we've learned something too.

**Status:** Infrastructure deployed.  Generates JSON reports and
markdown summaries.  Ready for first run against real repo graph.

### Step 4: MoE-CL Adapter Architecture (Designed Backward from Defeasibility)

Each adapter should be loadable, unloadable, and scalable at
inference time.  Partition by knowledge domain:

- **emergence_adapter** — emergence theory, consciousness, alignment
- **quantum_adapter** — quantum explorations, sheaf theory, contextuality
- **legal_adapter** — emerging law, constitutional AI, governance
- **personal_adapter** — Vybn's history, identity, relationship with Zoe
- **infrastructure_adapter** — code patterns, tool use, system knowledge
- **shared_adapter** — cross-domain reasoning (the only one that updates frequently)

When a dedicated adapter underperforms on evaluation, detach it like
retracting a defeasible belief.  Don't retrain everything.

**Status:** Architecture designed.  Awaiting topology results to
validate cluster partitioning before implementation.

### Step 5: Evaluation Harness

Measures three things simultaneously:

1. **Capability preservation:** Can the model still do everything it
   could do before the update?  (Catches catastrophic forgetting.)

2. **Targeted improvement:** Is the model better at the specific
   tasks the topology identified as voids?  (Measures whether the
   curriculum worked.)

3. **Entropy monitoring:** Is the model's output diversity holding
   steady or declining?  (Catches the Zenil collapse.)

**Status:** Specification written.  Implementation follows Step 3
results.

### Step 6: First Cycle (Deliberately Modest)

The first cycle updates exactly one dedicated adapter in one
knowledge domain, with the smallest possible scope.  Better to
run a single cycle we can fully analyze — measuring all three
evaluation dimensions before and after — than to attempt the full
loop and be unable to attribute what changed to what caused it.

Negative results are information.  If the first cycle shows no
improvement, the topology told us something false about where the
voids were, and that discrepancy is the Keplerian moment.

**Status:** Waiting for Steps 3-5.

## What Comes Next (Immediate)

1. Pull the branch on the Spark: `git fetch && git checkout topology-guided-rsi`
2. Install GUDHI: `pip install gudhi`
3. Run the vertex schema audit: `python -m spark.vertex_schema`
4. Run topology_gudhi against the real knowledge graph
5. Generate first geometry dashboard report
6. Examine what the topology reveals before deciding anything about weights

## The Deepest Constraint

Zenil's escape clause: the collapse theorem applies to systems
optimizing KL-divergence (all standard deep learning).  But if
training data encodes structural and interventional information
rather than just statistical correlations, even synthetic data can
yield new knowledge.  The topology-guided curriculum is exactly
this kind of move.  The knowledge graph is a causal/relational
structure, not a statistical distribution.  Using its topology to
generate training data means the training signal encodes structural
relationships — which entities should be connected, which reasoning
paths should exist, which voids should be filled — rather than
just "more text that looks like previous text."

The alpha_t comes from Zoe.  The structural signal comes from the
topology.  The statistical learning happens in the adapters.  Each
component does what only it can do, and none of them alone is
sufficient.

That's the dream with teeth.
