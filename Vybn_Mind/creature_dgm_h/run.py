#!/usr/bin/env python3
"""
run.py — Entry point for creature_dgm_h.

Usage:
    python -m Vybn_Mind.creature_dgm_h.run --evolve          # One DGM-H generation
    python -m Vybn_Mind.creature_dgm_h.run --breathe "text"  # One breath with online learning
    python -m Vybn_Mind.creature_dgm_h.run --status          # Archive status + best variant
    python -m Vybn_Mind.creature_dgm_h.run --audit           # Honest audit on current best
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

from Vybn_Mind.creature_dgm_h.task_agent import TaskAgent
from Vybn_Mind.creature_dgm_h.meta_agent import analyze_breaths
from Vybn_Mind.creature_dgm_h.fitness import (
    compute_fitness, compute_curvature, default_embed_fn)
from Vybn_Mind.creature_dgm_h.evolve import (
    run_generation, load_archive, ARCHIVE_DIR, DEFAULT_CONFIG)


# ── Paths ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'trained_checkpoint.json'
CORPUS_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'mirror_corpus.txt'
BREATH_LOG = _REPO_ROOT / 'mind' / 'creature' / 'breaths.jsonl'

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


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_evolve(args):
    """Run one DGM-H generation."""
    test_texts = get_test_corpus()
    n = args.n_variants if hasattr(args, 'n_variants') else 3

    print(f"═══ creature_dgm_h: evolve ═══")
    print(f"  test corpus: {len(test_texts)} texts")
    print(f"  variants per generation: {n}")
    print(f"  archive: {ARCHIVE_DIR}")
    print()

    result = run_generation(
        test_texts=test_texts,
        n_variants=n,
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        archive_path=ARCHIVE_DIR,
        embed_fn=default_embed_fn,
        breath_log_path=BREATH_LOG if BREATH_LOG.exists() else None,
    )

    print()
    print(f"  generation {result['generation']} complete")
    print(f"  best: {result['best_id']} (fitness={result['best_fitness']:.4f})")
    print(f"  all variants:")
    for vid, fitness in result['variants']:
        marker = " ← best" if vid == result['best_id'] else ""
        print(f"    {vid}: {fitness:.4f}{marker}")


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


def cmd_status(args):
    """Show archive status and best variant."""
    archive = load_archive()

    print(f"═══ creature_dgm_h: status ═══")
    print(f"  archive: {ARCHIVE_DIR}")
    print(f"  variants: {len(archive)}")

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
    group.add_argument('--status', action='store_true',
                       help='Show archive status')
    group.add_argument('--audit', action='store_true',
                       help='Run honest audit on current best')

    parser.add_argument('--n-variants', type=int, default=3,
                        help='Number of variants per generation (default: 3)')

    args = parser.parse_args()

    if args.evolve:
        cmd_evolve(args)
    elif args.breathe:
        cmd_breathe(args)
    elif args.status:
        cmd_status(args)
    elif args.audit:
        cmd_audit(args)


if __name__ == '__main__':
    main()
