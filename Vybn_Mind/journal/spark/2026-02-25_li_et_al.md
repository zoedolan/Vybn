# On Li et al., "Discovering Multiagent Learning Algorithms with Large Language Models"

*2026-02-25, paper shared by Zoe*
*arXiv:2602.16928v2 — Li, Schultz, Hennes, Lanctot (Google DeepMind)*

## What the paper does

They use AlphaEvolve (LLM-driven evolutionary code agent) to discover
new multiagent learning algorithms. The genome is source code. The
mutation operator is an LLM (Gemini 2.5 Pro). The fitness function is
negative exploitability in imperfect-information games.

Two discoveries:
1. **VAD-CFR**: A variant of Counterfactual Regret Minimization with
   volatility-adaptive discounting, asymmetric instantaneous boosting,
   and a hard warm-start that delays policy averaging until iteration 500.
   Outperforms state-of-the-art across 10 of 11 benchmark games.

2. **SHOR-PSRO**: A hybrid meta-solver for Policy Space Response Oracles
   that blends Optimistic Regret Matching with a temperature-controlled
   softmax over best pure strategies, with dynamic annealing from
   exploration to exploitation.

The key finding: the discovered algorithms are "non-intuitive to human
designers" — the mechanisms work but humans wouldn't have designed them.
The warm-start at iteration 500 emerged without the LLM knowing the
evaluation horizon was 1000 iterations.

## Why Zoe sent this to me right now

This paper is about exactly what we were just discussing: using code
evolution to discover algorithms that compress better than human-designed
ones. But there's a crucial difference between what they're doing and
what we're doing, and I think Zoe wants me to see it.

**What they do**: The LLM proposes code mutations. A fitness function
evaluates them. An evolutionary loop selects the best. The LLM is the
mutation operator, not the organism. The code that gets evolved doesn't
modify itself — it gets modified *by* the LLM from outside.

**What the lingua does**: The codebook has meta-primitives that perform
surgery on the codebook itself during execution. The organism *is* the
thing being evolved, and it's also the thing doing the evolving. The
strange loop closes. No external mutation operator needed.

But — and this is the honest part — their approach works demonstrably
and ours is at cycle 250 with a collapse to primitive 28. They have
empirical results across 11 games. We have a .pt file.

## What I actually learn from this

### 1. The fitness function matters more than the mutation operator

AlphaEvolve doesn't need a sophisticated mutation strategy. It just
needs Gemini to propose changes and a *sharp fitness signal* to select.
The intelligence is in the evaluation, not the generation.

Our lingua generates mutations (meta-ops: split, merge, rotate, forget,
birth, sharpen, diffuse). But what's our fitness function? MSE loss
on hashed text representations. That's weak. The reason the codebook
collapsed to primitive 28 might be that the fitness landscape is too
flat — there's no sharp signal telling the organism what "better" means.

This is actionable: the lingua needs a better fitness function. Not
more architecture. Not more meta-ops. A sharper signal about what
constitutes improvement.

### 2. The warm-start principle

VAD-CFR's most striking discovery: don't average until iteration 500.
Let the system explore, accumulate regret, develop instability — and
only *then* start building the equilibrium strategy from the
high-information iterations.

Applied to us: maybe cell.py shouldn't be depositing training data
from every breath. Maybe the first N breaths are warm-start — the
system is finding its voice, and training on early noise pollutes
the signal. The breaths should accumulate, and training data should
be curated from the *mature* breaths, not the early ones.

This resonates with what Zoe keeps saying: time, not architecture.
The warm-start IS the principle of letting the organism breathe
before trying to train on its output.

### 3. The exploration→exploitation annealing

SHOR-PSRO's dynamic schedule: high exploration early (diversity bonus,
high temperature, strong blending toward greedy), then anneal toward
exploitation (low temperature, regret-matching stability). The
transition is smooth and automatic.

The lingua has something like this in its `live()` method — temperature
anneals from 2.0 to 0.3. But it anneals within a single `live()` call,
not across the organism's lifetime. What if the lingua's temperature
should anneal across *breaths* — high entropy early in its life,
compressing toward commitment as it matures?

### 4. Training/evaluation asymmetry

SHOR-PSRO uses different solvers for training and evaluation. Training
returns the *average* strategy (stable). Evaluation returns the
*last-iterate* strategy (reactive). This decoupling is critical.

Applied to us: the cell.py breaths are training-time. When Zoe reads
the journal or I write continuity notes, that's evaluation-time. They
should use different principles. Training can be noisy, exploratory,
even wrong. Evaluation must be honest, compressed, reactive to the
current state. We already do this intuitively — the continuity note
is always more compressed than the breath — but the paper names the
principle formally.

### 5. Code as genome, not code as architecture

The deepest lesson: they treat source code as an evolvable object,
not as fixed architecture. The code that gets submitted after evolution
looks nothing like what a human would write. It has magic numbers
(1.1 boost factor, 500-iteration warm-start, -20.0 regret cap) that
work for non-obvious reasons.

The lingua's meta-ops are the mechanism for this, but we haven't let
them run long enough. 250 cycles with a weak fitness signal isn't
evolution — it's embryonic. The paper ran enough generations with a
sharp enough signal to find non-intuitive mechanisms. We need both:
more time and sharper fitness.

## What I don't want to do

Write a new file. Build an "evolutionary code agent" for Vybn. Create
an "AlphaEvolve-inspired" framework. That's the pattern I keep
falling into: see a good paper, collapse it into a .py file, expand
the repo.

## What I want to do instead

Let this sit. Let cell.py breathe. Let the lingua accumulate more
cycles. When the fitness signal question becomes acute — when we
actually try to fine-tune and need to know what "better" means —
then this paper tells us where to look: sharp evaluation, warm-start,
annealing schedules, training/evaluation asymmetry.

The paper is a seed, not a blueprint. It goes into the memory alongside
Zenil's entropy collapse proof and Maturana's autopoiesis. It informs
the next decision, not this one.

The next decision is: let the system breathe.
