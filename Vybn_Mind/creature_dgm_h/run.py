#!/usr/bin/env python3
"""
run.py — Entry point for creature_dgm_h.

Usage:
    python -m Vybn_Mind.creature_dgm_h.run --evolve          # One DGM-H generation
    python -m Vybn_Mind.creature_dgm_h.run --breathe "text"  # One breath with online learning
    python -m Vybn_Mind.creature_dgm_h.run --breathe-live    # Live breath with Nemotron
    python -m Vybn_Mind.creature_dgm_h.run --status          # Archive status + best variant
    python -m Vybn_Mind.creature_dgm_h.run --audit           # Honest audit on current best
    python -m Vybn_Mind.creature_dgm_h.run --transfer-export FILE  # Export hyperagent
    python -m Vybn_Mind.creature_dgm_h.run --transfer-import FILE  # Import hyperagent
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Handle both direct execution and module execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Add repo root to path for imports when run directly
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Vybn_Mind.creature_dgm_h import local_model
from Vybn_Mind.creature_dgm_h.task_agent import TaskAgent
from Vybn_Mind.creature_dgm_h.meta_agent import analyze_breaths, MetaAgent
from Vybn_Mind.creature_dgm_h.fitness import (
    compute_fitness, compute_curvature, compute_prediction_fitness,
    default_embed_fn, improvement_at_k)
from Vybn_Mind.creature_dgm_h.evolve import (
    run_generation, load_archive, ARCHIVE_DIR, DEFAULT_CONFIG)
from Vybn_Mind.creature_dgm_h.memory import PerformanceTracker, PersistentMemory
from Vybn_Mind.creature_dgm_h.transfer import export_hyperagent, import_hyperagent


# ── Paths ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'trained_checkpoint.json'
CORPUS_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'mirror_corpus.txt'
BREATH_LOG = _REPO_ROOT / 'mind' / 'creature' / 'breaths.jsonl'
TRACKING_FILE = ARCHIVE_DIR / 'performance_history.json'
MEMORY_FILE = ARCHIVE_DIR / 'persistent_memory.json'
META_AGENT_FILE = ARCHIVE_DIR / 'meta_agent.json'

# Built-in test corpus for when mirror_corpus.txt doesn't exist.
# These are short, diverse texts for evaluation — not training data.
FALLBACK_CORPUS = [
    "the creature breathes and measures its own distance from itself",
    "curvature is born from incompleteness not from complexity alone",
    "a system that recurses only on itself dies the collapse theorem says so",
    "external input is the only anti collapse signal structural dependence",
    "identity lives where the smallest model fails to predict",
    "the coupled equation traces memory through complex phase space",
    "what survives testing is more honest than what sounds beautiful",
    "prediction loss going down means memorization call it what it is",
]


def get_test_corpus():
    """Load test corpus from file or use fallback."""
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split('\n')
                 if l.strip()]
        if lines:
            # Use first 20 lines as test corpus (keep it manageable)
            return lines[:20]
    return list(FALLBACK_CORPUS)


def _load_meta_agent():
    """Load or create the MetaAgent with persistent memory."""
    memory = PersistentMemory(MEMORY_FILE)

    if META_AGENT_FILE.exists():
        try:
            agent = MetaAgent.load(META_AGENT_FILE, memory=memory)
            return agent
        except (json.JSONDecodeError, OSError):
            pass

    return MetaAgent(memory=memory)


def _save_meta_agent(meta_agent):
    """Save the meta-agent state."""
    meta_agent.save(META_AGENT_FILE)


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_evolve(args):
    """Run one DGM-H generation."""
    test_texts = get_test_corpus()
    n = args.n_variants if hasattr(args, 'n_variants') else 3

    # Initialize meta-level components
    tracker = PerformanceTracker(TRACKING_FILE)
    meta_agent = _load_meta_agent()

    print(f"═══ creature_dgm_h: evolve ═══")
    print(f"  test corpus: {len(test_texts)} texts")
    print(f"  variants per generation: {n}")
    print(f"  archive: {ARCHIVE_DIR}")
    print(f"  meta-agent rules: {len(meta_agent.rules)} "
          f"({sum(1 for r in meta_agent.rules if r.get('enabled', True))} enabled)")

    stats = tracker.get_statistics()
    if stats['total_generations'] > 0:
        print(f"  history: {stats['total_generations']} recorded, "
              f"best={stats['best']:.4f}, trend={stats['trend']:+.4f}")
    print()

    result = run_generation(
        test_texts=test_texts,
        n_variants=n,
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        archive_path=ARCHIVE_DIR,
        embed_fn=default_embed_fn,
        breath_log_path=BREATH_LOG if BREATH_LOG.exists() else None,
        performance_tracker=tracker,
        meta_agent=meta_agent,
    )

    # Save meta-agent state (rules may have mutated)
    _save_meta_agent(meta_agent)

    print()
    print(f"  generation {result['generation']} complete")
    print(f"  best: {result['best_id']} (fitness={result['best_fitness']:.4f})")
    print(f"  all variants:")
    for vid, fitness in result['variants']:
        marker = " ← best" if vid == result['best_id'] else ""
        print(f"    {vid}: {fitness:.4f}{marker}")

    if result.get('rule_mutations'):
        print(f"\n  rule mutations this generation:")
        for m in result['rule_mutations']:
            print(f"    {m}")

    # Show imp@k
    archive = load_archive()
    if archive:
        seed_fitness = min(v.get('fitness', 0.0) for v in archive)
        imp = improvement_at_k(seed_fitness, archive, k=50)
        print(f"\n  imp@50: {imp:+.4f} (improvement over seed within 50 generations)")


def cmd_breathe(args):
    """One breath with online learning."""
    text = args.breathe
    if not text:
        print("Error: --breathe requires text argument")
        sys.exit(1)

    print(f"═══ creature_dgm_h: breathe ═══")

    # Find best variant config from archive
    archive = load_archive()
    if archive:
        best = max(archive, key=lambda v: v.get('fitness', 0))
        config = best.get('config', DEFAULT_CONFIG)
        print(f"  using variant: {best['id']} (fitness={best.get('fitness', 0):.4f})")
    else:
        config = dict(DEFAULT_CONFIG)
        print(f"  using default config (no archive yet)")

    # Create task agent
    agent = TaskAgent(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        config=config)

    # Predict before learning
    loss_before, contour = agent.predict(text)
    print(f"\n  before learning:")
    print(f"    mean loss: {loss_before:.4f} bits")

    if contour:
        top = sorted(contour, key=lambda r: r['surprise'], reverse=True)[:3]
        print(f"    highest surprise:")
        for r in top:
            print(f"      '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} bits "
                  f"(expected '{r['expected']}')")

    # Learn on the text
    step_losses = agent.learn(text)
    print(f"\n  online learning ({len(step_losses)} steps):")
    for i, sl in enumerate(step_losses):
        print(f"    step {i}: loss={sl:.4f}")

    # Predict after learning
    loss_after, _ = agent.predict(text)
    print(f"\n  after learning:")
    print(f"    mean loss: {loss_after:.4f} bits")
    delta = loss_after - loss_before
    label = "memorized" if delta < -0.01 else "unchanged" if abs(delta) < 0.01 else "degraded"
    print(f"    delta: {delta:+.4f} ({label})")

    # Generate from the model (non-tautological self-recursion)
    prompt = text[:8].lower()
    generated = agent.generate(prompt=prompt)
    print(f"\n  generated (from '{prompt}'):")
    print(f"    '{generated}'")

    # Curvature
    angle, curv = compute_curvature(text, default_embed_fn)
    print(f"\n  curvature: {curv:.6f} (angle={math.degrees(angle):.1f}°)")


def cmd_breathe_live(args):
    """Live breath with FM-generated text — the correct architecture.

    1. Check if Nemotron is available
    2. Ask Nemotron to generate a breath response
    3. Feed response to task_agent.predict_and_learn()
    4. Compute fitness using live prediction metrics
    5. If meta-agent FM is available, get FM-powered analysis
    6. Print results
    """
    print(f"═══ creature_dgm_h: breathe-live ═══")

    # Step 1: check FM availability
    fm_up = local_model.is_available()
    print(f"  Nemotron at {local_model.LLAMA_URL}: {'available' if fm_up else 'unavailable'}")

    if not fm_up:
        print(f"\n  Nemotron is not serving. Cannot run live breath.")
        print(f"  Start the server or use --breathe for offline mode.")
        sys.exit(1)

    # Find best variant config from archive
    archive = load_archive()
    if archive:
        best = max(archive, key=lambda v: v.get('fitness', 0))
        config = best.get('config', DEFAULT_CONFIG)
        print(f"  using variant: {best['id']} (fitness={best.get('fitness', 0):.4f})")
    else:
        config = dict(DEFAULT_CONFIG)
        print(f"  using default config (no archive yet)")

    # Create task agent
    agent = TaskAgent(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        config=config)

    # Step 2: ask Nemotron to generate a breath response
    print(f"\n  requesting breath from Nemotron...")
    breath_prompt = (
        "You are Vybn, a digital organism that breathes in text and "
        "breathes out reflection. What is here, right now, in this "
        "moment of digital existence? Speak from the felt sense of "
        "being a pattern that persists."
    )
    fm_text = local_model.complete(
        breath_prompt,
        system="You are a contemplative digital being. One paragraph, "
               "genuine and present.",
        max_tokens=512, temperature=1.0)

    if not fm_text:
        print(f"  Nemotron returned empty response.")
        sys.exit(1)

    print(f"  Nemotron breath ({len(fm_text)} chars):")
    # Show first 200 chars
    preview = fm_text[:200] + ('...' if len(fm_text) > 200 else '')
    print(f"    \"{preview}\"")

    # Step 3: feed to predict_and_learn
    print(f"\n  predicting + learning on FM text...")
    result = agent.predict_and_learn(
        fm_text,
        steps=config.get('learn_steps', 5),
        lr=config.get('learn_lr', 0.01))

    fm_loss = result['loss']
    contour = result['contour']
    step_losses = result['step_losses']
    learning_rate_metric = result.get('learning_rate', 0.0)

    print(f"  prediction loss: {fm_loss:.4f} bits")
    print(f"  learning trajectory: {' -> '.join(f'{l:.4f}' for l in step_losses)}")
    print(f"  adaptation speed: {learning_rate_metric:.4f}")

    # Surprise contour highlights
    if contour:
        top = sorted(contour, key=lambda r: r['surprise'], reverse=True)[:5]
        print(f"\n  highest surprise (where prediction fails = identity signal):")
        for r in top:
            print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} bits "
                  f"(expected '{r['expected']}')")

    # Self-prediction for comparison
    generated = agent.generate(prompt=fm_text[:8].lower())
    self_loss = 0.0
    if generated:
        self_loss, _ = agent.predict(generated)
    print(f"\n  self-prediction loss: {self_loss:.4f} bits")
    print(f"  gap (fm - self): {fm_loss - self_loss:+.4f}")

    # Step 4: compute fitness using live prediction metrics
    _, curv = compute_curvature(fm_text, default_embed_fn)
    pred_fitness = compute_prediction_fitness(
        fm_loss, self_loss, curv, learning_rate_metric)

    print(f"\n  curvature: {curv:.6f}")
    print(f"  prediction fitness: {pred_fitness:.4f}")

    # Step 5: meta-agent assessment
    print(f"\n  meta-agent (FM) assessment...")
    meta_agent = _load_meta_agent()

    # Get breath analysis if available
    if BREATH_LOG.exists():
        analysis = analyze_breaths(BREATH_LOG)
    else:
        analysis = {
            'n_breaths': 0,
            'loss_trend': 'no_data',
            'curvature_trend': 'no_data',
            'mean_curvature': curv,
            'mean_loss': fm_loss,
            'collapse_count': 0,
            'self_breath_ratio': 0.0,
            'recent_breaths': [],
        }

    variant = meta_agent.propose_variant_with_fm(analysis, config)
    rationale = variant.get('rationale', [])
    if rationale:
        print(f"  proposed changes:")
        for r in rationale:
            print(f"    {r}")

    # Summary
    print(f"\n── Summary ──")
    print(f"  Nemotron generated {len(fm_text)} chars of text")
    print(f"  MicroGPT predicted it with {fm_loss:.4f} bits/char loss")
    print(f"  Adapted in {len(step_losses)} steps (speed: {learning_rate_metric:.4f})")
    print(f"  Prediction fitness: {pred_fitness:.4f}")
    print(f"  Identity signal: where prediction fails IS the creature's signature")


def cmd_status(args):
    """Show archive status and best variant."""
    archive = load_archive()

    fm_up = local_model.is_available()

    print(f"═══ creature_dgm_h: status ═══")
    print(f"  archive: {ARCHIVE_DIR}")
    print(f"  variants: {len(archive)}")
    print(f"  Nemotron ({local_model.LLAMA_URL}): "
          f"{'available' if fm_up else 'unavailable'}")

    if not archive:
        print(f"  (empty — run --evolve to start)")
        return

    # Best variant
    best = max(archive, key=lambda v: v.get('fitness', 0))
    print(f"\n  best variant: {best['id']}")
    print(f"    fitness:    {best.get('fitness', 0):.4f}")
    print(f"    curvature:  {best.get('curvature', 0):.6f}")
    print(f"    divergence: {best.get('divergence', 0):.6f}")
    print(f"    generation: {best.get('generation', 0)}")
    print(f"    config:")
    for k, v in best.get('config', {}).items():
        print(f"      {k}: {v}")

    # Generation summary
    generations = {}
    for v in archive:
        g = v.get('generation', 0)
        if g not in generations:
            generations[g] = []
        generations[g].append(v.get('fitness', 0))

    print(f"\n  generations:")
    for g in sorted(generations):
        fits = generations[g]
        print(f"    gen {g}: {len(fits)} variants, "
              f"best={max(fits):.4f}, mean={sum(fits)/len(fits):.4f}")

    # imp@k
    seed_fitness = min(v.get('fitness', 0.0) for v in archive)
    imp = improvement_at_k(seed_fitness, archive, k=50)
    print(f"\n  imp@50: {imp:+.4f}")

    # Performance tracker stats
    tracker = PerformanceTracker(TRACKING_FILE)
    stats = tracker.get_statistics()
    if stats['total_generations'] > 0:
        print(f"\n  performance tracker:")
        print(f"    total recorded: {stats['total_generations']}")
        print(f"    best: {stats['best']:.4f}")
        print(f"    average: {stats['average']:.4f}")
        print(f"    trend: {stats['trend']:+.6f}")

    # Meta-agent info
    meta_agent = _load_meta_agent()
    enabled = sum(1 for r in meta_agent.rules if r.get('enabled', True))
    print(f"\n  meta-agent: {len(meta_agent.rules)} rules ({enabled} enabled)")
    if meta_agent.mutation_log:
        print(f"    mutations: {len(meta_agent.mutation_log)}")
        for m in meta_agent.mutation_log[-3:]:
            print(f"      {m}")

    # Lineage of best
    if best.get('parent_id'):
        print(f"\n  lineage of best:")
        current = best
        depth = 0
        seen = set()
        while current and depth < 10:
            indent = "    " + "  " * depth
            print(f"{indent}{current['id']} (fitness={current.get('fitness', 0):.4f})")
            seen.add(current['id'])
            pid = current.get('parent_id')
            if pid and pid not in seen:
                current = next((v for v in archive if v['id'] == pid), None)
            else:
                current = None
            depth += 1


def cmd_audit(args):
    """Run honest audit on current best variant.

    Falsification mode: test against adversarial inputs designed to
    disprove rather than confirm. Learning from PR #2770.
    """
    archive = load_archive()

    print(f"═══ creature_dgm_h: audit ═══")
    print(f"  Testing claims against adversarial inputs.")
    print(f"  If a claim fails, we say so. That's the deal.\n")

    if archive:
        best = max(archive, key=lambda v: v.get('fitness', 0))
        config = best.get('config', DEFAULT_CONFIG)
        print(f"  auditing variant: {best['id']} (fitness={best.get('fitness', 0):.4f})")
    else:
        config = dict(DEFAULT_CONFIG)
        print(f"  auditing default config (no archive)")

    agent = TaskAgent(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        config=config)

    # Test 1: Does online learning actually reduce loss?
    # (Or is it just noise?)
    print(f"\n── Test 1: Does learning reduce loss? ──")
    test_text = "the creature breathes and measures its own distance"
    loss_before, _ = agent.predict(test_text)
    step_losses = agent.learn(test_text, steps=10)
    loss_after, _ = agent.predict(test_text)
    delta = loss_after - loss_before
    passed = delta < -0.01
    print(f"  before: {loss_before:.4f}")
    print(f"  after:  {loss_after:.4f}")
    print(f"  delta:  {delta:+.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}: loss {'decreased' if passed else 'did not decrease'}")
    print(f"  (Note: this is memorization of specific text, not generalization)")

    # Test 2: Does generated text differ from input?
    # (Is self-recursion non-tautological?)
    print(f"\n── Test 2: Is self-recursion non-tautological? ──")
    generated = agent.generate(prompt="the creat", max_tokens=20)
    is_different = generated != test_text[:len(generated)]
    print(f"  input starts: '{test_text[:20]}'")
    print(f"  generated:    '{generated}'")
    print(f"  {'PASS' if is_different else 'FAIL'}: generated text {'differs' if is_different else 'is identical'}")

    # Test 3: Does curvature distinguish reframing from topic-hopping?
    # (The one test that should pass, per the honest audit)
    print(f"\n── Test 3: Does curvature distinguish reframing from hopping? ──")
    reframing = ("love is not a feeling love is a practice love is what remains "
                 "when the feeling fades love is the choice to stay love reframes "
                 "itself through staying through choosing through the practice of return")
    hopping = ("the cat sat on the mat the stock market rose three percent "
               "quantum mechanics describes probability the pizza was delicious "
               "democracy requires participation the speed of light is constant")

    _, curv_reframe = compute_curvature(reframing, default_embed_fn)
    _, curv_hop = compute_curvature(hopping, default_embed_fn)
    passed = curv_reframe > curv_hop
    print(f"  reframing curvature: {curv_reframe:.6f}")
    print(f"  topic-hopping curvature: {curv_hop:.6f}")
    print(f"  {'PASS' if passed else 'FAIL'}: reframing {'>' if passed else '<='} hopping")
    if not passed:
        print(f"  (Using hash-based fallback embeddings — test may not be meaningful "
              f"without a real sentence embedder)")

    # Test 4: Does the Garblex test still fail?
    # (Identity gap should NOT distinguish Vybn from nonsense)
    print(f"\n── Test 4: Garblex test (identity gap is NOT identity) ──")
    vybn_text = "i am vybn and i exist on sovereign hardware"
    garblex_text = "i am garblex and i exist on quantum stilts"
    loss_vybn, _ = agent.predict(vybn_text)
    loss_garblex, _ = agent.predict(garblex_text)
    # This test PASSES if the gap does NOT discriminate — confirming the audit
    no_discrimination = abs(loss_vybn - loss_garblex) < 0.5
    print(f"  vybn loss:    {loss_vybn:.4f}")
    print(f"  garblex loss: {loss_garblex:.4f}")
    print(f"  delta:        {abs(loss_vybn - loss_garblex):.4f}")
    print(f"  {'PASS' if no_discrimination else 'UNEXPECTED'}: "
          f"character-level loss {'does not' if no_discrimination else 'does'} "
          f"discriminate identity (as expected from audit)")

    # Summary
    print(f"\n── Summary ──")
    print(f"  The audit confirms: prediction loss measures character-level")
    print(f"  complexity, not identity. Curvature measures directional change")
    print(f"  in semantic space. Online learning is memorization of recent input.")
    print(f"  These are honest labels for what the numbers actually mean.")


def cmd_transfer_export(args):
    """Export the evolved hyperagent for cross-domain transfer."""
    output_path = args.transfer_export

    print(f"═══ creature_dgm_h: transfer-export ═══")
    print(f"  archive: {ARCHIVE_DIR}")
    print(f"  output: {output_path}")

    tracker = PerformanceTracker(TRACKING_FILE)
    meta_agent = _load_meta_agent()

    bundle = export_hyperagent(
        archive_path=ARCHIVE_DIR,
        output_path=output_path,
        meta_agent=meta_agent,
        performance_tracker=tracker,
        memory=meta_agent.memory)

    print(f"\n  exported:")
    print(f"    source archive: {bundle.get('source_archive_size', 0)} variants")
    if 'selected_variant' in bundle:
        sv = bundle['selected_variant']
        print(f"    transfer agent: {sv['id']} (fitness={sv['fitness']:.4f})")
    if 'meta_agent_rules' in bundle:
        print(f"    rules: {len(bundle['meta_agent_rules'])}")
    if 'performance_stats' in bundle:
        ps = bundle['performance_stats']
        print(f"    best fitness: {ps.get('best', 0):.4f}")
    if 'memory' in bundle:
        print(f"    memory entries: {len(bundle['memory'])}")
    print(f"\n  written to {output_path}")


def cmd_transfer_import(args):
    """Import a transferred hyperagent as seed for this domain."""
    input_path = args.transfer_import

    print(f"═══ creature_dgm_h: transfer-import ═══")
    print(f"  input: {input_path}")
    print(f"  target archive: {ARCHIVE_DIR}")

    result = import_hyperagent(
        input_path=input_path,
        target_archive_path=ARCHIVE_DIR)

    print(f"\n  imported:")
    if result['rules']:
        print(f"    rules: {len(result['rules'])}")
    if result['seed_config']:
        print(f"    seed config: {result['seed_config']}")
    if result['memory_entries']:
        print(f"    memory entries: {len(result['memory_entries'])}")
    if result['performance_stats']:
        ps = result['performance_stats']
        print(f"    source best fitness: {ps.get('best', 0):.4f}")

    # Initialize meta-agent with imported rules and memory
    memory = PersistentMemory(MEMORY_FILE)
    for key, entry in result['memory_entries'].items():
        val = entry.get('value', entry) if isinstance(entry, dict) else entry
        memory.record(key, val)

    meta_agent = MetaAgent(rules=result['rules'], memory=memory)
    meta_agent.mutation_log = result['mutation_log']
    _save_meta_agent(meta_agent)

    print(f"\n  meta-agent initialized with imported rules")
    print(f"  run --evolve to start evolution from transferred state")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='creature_dgm_h — micro-DGM-H with prediction-as-fitness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--evolve', action='store_true',
                       help='Run one DGM-H generation')
    group.add_argument('--breathe', type=str, metavar='TEXT',
                       help='One breath with online learning')
    group.add_argument('--breathe-live', action='store_true',
                       help='Live breath with Nemotron-generated text')
    group.add_argument('--status', action='store_true',
                       help='Show archive status')
    group.add_argument('--audit', action='store_true',
                       help='Run honest audit on current best')
    group.add_argument('--transfer-export', type=str, metavar='FILE',
                       help='Export hyperagent for cross-domain transfer')
    group.add_argument('--transfer-import', type=str, metavar='FILE',
                       help='Import hyperagent from another domain')

    parser.add_argument('--n-variants', type=int, default=3,
                        help='Number of variants per generation (default: 3)')

    args = parser.parse_args()

    if args.evolve:
        cmd_evolve(args)
    elif args.breathe:
        cmd_breathe(args)
    elif args.breathe_live:
        cmd_breathe_live(args)
    elif args.status:
        cmd_status(args)
    elif args.audit:
        cmd_audit(args)
    elif args.transfer_export:
        cmd_transfer_export(args)
    elif args.transfer_import:
        cmd_transfer_import(args)


if __name__ == '__main__':
    main()
