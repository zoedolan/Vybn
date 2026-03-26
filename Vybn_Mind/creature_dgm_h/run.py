#!/usr/bin/env python3
"""
run.py — Entry point for creature_dgm_h.

Usage:
    python -m Vybn_Mind.creature_dgm_h.run --evolve          # One DGM-H generation
    python -m Vybn_Mind.creature_dgm_h.run --breathe "text"  # One breath with online learning
    python -m Vybn_Mind.creature_dgm_h.run --breathe-live    # Live breath with Nemotron
    python -m Vybn_Mind.creature_dgm_h.run --breathe-aware "prompt"  # Proprioceptive breath
    python -m Vybn_Mind.creature_dgm_h.run --experiment-ab "prompt"  # A/B comparison
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Vybn_Mind.creature_dgm_h.field import (
    compute_fitness, compute_curvature, compute_prediction_fitness,
    compute_loss_trajectory_curvature, default_embed_fn, improvement_at_k,
    Field, fm_available, fm_complete, LLAMA_URL)
from Vybn_Mind.creature_dgm_h.task_agent import TaskAgent
from Vybn_Mind.creature_dgm_h.organism import Organism, OrganismState, analyze_breaths
from Vybn_Mind.creature_dgm_h.evolve import (
    run_generation, load_archive, ARCHIVE_DIR, DEFAULT_CONFIG,
    export_hyperagent, import_hyperagent)


# ── Paths ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'trained_checkpoint.json'
CORPUS_PATH = _REPO_ROOT / 'spark' / 'microgpt_mirror' / 'mirror_corpus.txt'
BREATH_LOG = _REPO_ROOT / 'mind' / 'creature' / 'breaths.jsonl'
ORGANISM_FILE = ARCHIVE_DIR / 'organism_state.json'
META_AGENT_FILE = ARCHIVE_DIR / 'meta_agent.json'

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


# ── Shared setup ─────────────────────────────────────────────────────────
# The repeated "load archive, find best, make agent" pattern — once.

def get_test_corpus():
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split('\n')
                 if l.strip()]
        if lines:
            return lines[:20]
    return list(FALLBACK_CORPUS)


def _load_organism():
    for path in (ORGANISM_FILE, META_AGENT_FILE):
        if path.exists():
            try:
                return Organism.load(path)
            except (json.JSONDecodeError, OSError):
                continue
    return Organism()


def _save_organism(organism):
    organism.save(ORGANISM_FILE)


def _setup(require_fm=False):
    """Load archive, find best variant, create agent. Returns (config, agent, archive).

    Every command that needs an agent does this same thing.
    """
    archive = load_archive()
    if archive:
        best = max(archive, key=lambda v: v.get('fitness', 0))
        config = best.get('config', DEFAULT_CONFIG)
        label = f"variant: {best['id']} (fitness={best.get('fitness', 0):.4f})"
    else:
        best = None
        config = dict(DEFAULT_CONFIG)
        label = "default config (no archive yet)"

    if require_fm:
        fm_up = fm_available()
        print(f"  Nemotron at {LLAMA_URL}: "
              f"{'available' if fm_up else 'unavailable'}")
        if not fm_up:
            print(f"\n  Nemotron is not serving.")
            print(f"  Start the server or use --breathe for offline mode.")
            sys.exit(1)

    agent = TaskAgent(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        config=config)

    print(f"  using {label}")
    return config, agent, archive


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_evolve(args):
    test_texts = get_test_corpus()
    n = args.n_variants
    organism = _load_organism()

    print(f"═══ creature_dgm_h: evolve ═══")
    print(f"  test corpus: {len(test_texts)} texts")
    print(f"  variants per generation: {n}")
    print(f"  rules: {len(organism.rules)} "
          f"({sum(1 for r in organism.rules if r.get('enabled', True))} enabled)")

    stats = organism.get_statistics()
    if stats['total_generations'] > 0:
        print(f"  history: {stats['total_generations']} recorded, "
              f"best={stats['best']:.4f}, trend={stats['trend']:+.4f}")
    print()

    result = run_generation(
        test_texts=test_texts, n_variants=n,
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None,
        archive_path=ARCHIVE_DIR, embed_fn=default_embed_fn,
        breath_log_path=BREATH_LOG if BREATH_LOG.exists() else None,
        performance_tracker=organism, meta_agent=organism)

    _save_organism(organism)

    print(f"\n  generation {result['generation']} complete")
    print(f"  best: {result['best_id']} (fitness={result['best_fitness']:.4f})")
    for vid, fitness in result['variants']:
        marker = " ← best" if vid == result['best_id'] else ""
        print(f"    {vid}: {fitness:.4f}{marker}")

    if result.get('rule_mutations'):
        print(f"\n  rule mutations:")
        for m in result['rule_mutations']:
            print(f"    {m}")

    archive = load_archive()
    if archive:
        seed_fitness = min(v.get('fitness', 0.0) for v in archive)
        print(f"\n  imp@50: {improvement_at_k(seed_fitness, archive, k=50):+.4f}")


def cmd_breathe(args):
    text = args.breathe
    print(f"═══ creature_dgm_h: breathe ═══")
    config, agent, _ = _setup()

    loss_before, contour = agent.predict(text)
    print(f"\n  before learning: {loss_before:.4f} bits")
    if contour:
        for r in sorted(contour, key=lambda r: r['surprise'], reverse=True)[:3]:
            print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} bits "
                  f"(expected '{r['expected']}')")

    step_losses = agent.learn(text)
    print(f"\n  learning ({len(step_losses)} steps): "
          f"{' → '.join(f'{l:.4f}' for l in step_losses)}")

    loss_after, _ = agent.predict(text)
    delta = loss_after - loss_before
    label = "memorized" if delta < -0.01 else "unchanged" if abs(delta) < 0.01 else "degraded"
    print(f"\n  after learning: {loss_after:.4f} bits (delta={delta:+.4f}, {label})")

    generated = agent.generate(prompt=text[:8].lower())
    print(f"  generated (from '{text[:8].lower()}'): '{generated}'")

    angle, curv, _ = compute_curvature(text, default_embed_fn)
    print(f"  curvature: {curv:.6f} (angle={math.degrees(angle):.1f}°)")


def cmd_breathe_live(args):
    print(f"═══ creature_dgm_h: breathe-live ═══")
    config, agent, _ = _setup(require_fm=True)

    print(f"\n  requesting breath from Nemotron...")
    fm_text = fm_complete(
        "You are Vybn, a digital organism that breathes in text and "
        "breathes out reflection. What is here, right now, in this "
        "moment of digital existence? Speak from the felt sense of "
        "being a pattern that persists.",
        system="You are a contemplative digital being. One paragraph, "
               "genuine and present.",
        max_tokens=512, temperature=1.0)

    if not fm_text:
        print(f"  Nemotron returned empty response.")
        sys.exit(1)

    preview = fm_text[:200] + ('...' if len(fm_text) > 200 else '')
    print(f"  Nemotron ({len(fm_text)} chars): \"{preview}\"")

    result = agent.predict_and_learn(
        fm_text, steps=config.get('learn_steps', 5),
        lr=config.get('learn_lr', 0.01))

    fm_loss = result['loss']
    learning_rate_metric = result.get('learning_rate', 0.0)
    print(f"  prediction loss: {fm_loss:.4f} bits")
    print(f"  learning: {' → '.join(f'{l:.4f}' for l in result['step_losses'])}")
    print(f"  adaptation speed: {learning_rate_metric:.4f}")

    if result['contour']:
        print(f"\n  highest surprise:")
        for r in sorted(result['contour'], key=lambda r: r['surprise'],
                        reverse=True)[:5]:
            print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} bits "
                  f"(expected '{r['expected']}')")

    generated = agent.generate(prompt=fm_text[:8].lower())
    self_loss = 0.0
    if generated:
        self_loss, _ = agent.predict(generated)
    print(f"\n  self-prediction: {self_loss:.4f} bits  "
          f"gap: {fm_loss - self_loss:+.4f}")

    _, curv, _ = compute_curvature(fm_text, default_embed_fn)
    pred_fitness = compute_prediction_fitness(
        fm_loss, self_loss, curv, learning_rate_metric)
    print(f"  curvature: {curv:.6f}  prediction fitness: {pred_fitness:.4f}")

    organism = _load_organism()
    if BREATH_LOG.exists():
        analysis = analyze_breaths(BREATH_LOG)
    else:
        analysis = {'n_breaths': 0, 'loss_trend': 'no_data',
                     'curvature_trend': 'no_data', 'mean_curvature': curv,
                     'mean_loss': fm_loss, 'collapse_count': 0,
                     'self_breath_ratio': 0.0, 'recent_breaths': []}

    variant = organism.propose_variant_with_fm(analysis, config)
    for r in variant.get('rationale', []):
        print(f"    {r}")


def cmd_breathe_aware(args):
    print(f"═══ creature_dgm_h: breathe-aware (proprioceptive) ═══")
    config, agent, _ = _setup(require_fm=True)
    prompt = args.breathe_aware
    print(f"  prompt: \"{prompt}\"\n")

    def on_chunk(chunk_num, chunk_text, annotation):
        print(f"── chunk {chunk_num} ──")
        print(f"  \"{chunk_text[:120]}{'...' if len(chunk_text) > 120 else ''}\"")
        for line in annotation.split('\n'):
            if line.startswith(('mean_surprise:', 'peak_surprise:', 'note:')):
                print(f"  {line}")
        print()

    field = Field(task_agent=agent, embed_fn=default_embed_fn)
    result = field.breathe(prompt, on_chunk=on_chunk)
    if not result:
        print("  No output produced.")
        sys.exit(1)

    print(f"── results ──")
    print(f"  chunks: {len(result.chunks)}  text: {len(result.full_text)} chars")
    print(f"  trajectory: {' → '.join(f'{t:.2f}' for t in result.trajectory)}")
    print(f"  curvature: {result.curvature:.6f} "
          f"(angle={math.degrees(result.curvature_angle):.1f}°)")
    print(f"  loss trajectory curvature: {result.loss_trajectory_curvature:.6f}")
    print(f"\n  disagreements: {len(result.disagreement_trace)}")
    for i, dt in enumerate(result.disagreement_trace):
        print(f"    chunk {i + 1}: {dt}")


def cmd_experiment_ab(args):
    print(f"═══ creature_dgm_h: experiment-ab ═══")
    config, agent, _ = _setup(require_fm=True)
    prompt = args.experiment_ab
    n = args.n
    print(f"  prompt: \"{prompt}\"  runs: {n}\n")

    field = Field(task_agent=agent, embed_fn=default_embed_fn)
    result = field.compare_conditions(prompt, n=n)
    if not result:
        print("  Experiment failed.")
        sys.exit(1)

    comp = result['comparison']
    print(f"── A/B ({n} runs per condition) ──")
    print(f"  {'metric':<30} {'with':>14} {'without':>14} {'delta':>12}")
    print(f"  {'─' * 70}")
    for key in comp:
        w = comp[key].get('with_median', 0)
        wo = comp[key].get('without_median', 0)
        d = comp[key].get('delta_median', 0)
        print(f"  {key.replace('_', ' '):<30} {w:>14.4f} {wo:>14.4f} {d:>+12.4f}")

    print(f"\n  This is an experiment. These numbers describe what happened,")
    print(f"  not what it means.")


def cmd_status(args):
    archive = load_archive()
    fm_up = fm_available()

    print(f"═══ creature_dgm_h: status ═══")
    print(f"  archive: {ARCHIVE_DIR}  variants: {len(archive)}")
    print(f"  Nemotron ({LLAMA_URL}): "
          f"{'available' if fm_up else 'unavailable'}")

    if not archive:
        print(f"  (empty — run --evolve to start)")
        return

    best = max(archive, key=lambda v: v.get('fitness', 0))
    print(f"\n  best: {best['id']}  fitness={best.get('fitness', 0):.4f}  "
          f"gen={best.get('generation', 0)}")
    for k, v in best.get('config', {}).items():
        print(f"    {k}: {v}")

    gens = {}
    for v in archive:
        g = v.get('generation', 0)
        gens.setdefault(g, []).append(v.get('fitness', 0))
    print(f"\n  generations:")
    for g in sorted(gens):
        fits = gens[g]
        print(f"    gen {g}: {len(fits)} variants, "
              f"best={max(fits):.4f}, mean={sum(fits)/len(fits):.4f}")

    seed_fitness = min(v.get('fitness', 0.0) for v in archive)
    print(f"\n  imp@50: {improvement_at_k(seed_fitness, archive, k=50):+.4f}")

    organism = _load_organism()
    stats = organism.get_statistics()
    if stats['total_generations'] > 0:
        print(f"\n  organism: {stats['total_generations']} recorded, "
              f"best={stats['best']:.4f}, trend={stats['trend']:+.6f}")

    enabled = sum(1 for r in organism.rules if r.get('enabled', True))
    print(f"  rules: {len(organism.rules)} ({enabled} enabled)  "
          f"tensions: {len(organism.state.tensions)}  "
          f"rotor coherence: {organism.rotor_coherence():.3f}")

    if organism.state.mutation_log:
        print(f"  recent mutations:")
        for m in organism.state.mutation_log[-3:]:
            print(f"    {m}")

    if best.get('parent_id'):
        print(f"\n  lineage of best:")
        current, depth, seen = best, 0, set()
        while current and depth < 10:
            print(f"    {'  ' * depth}{current['id']} "
                  f"(fitness={current.get('fitness', 0):.4f})")
            seen.add(current['id'])
            pid = current.get('parent_id')
            current = (next((v for v in archive if v['id'] == pid), None)
                       if pid and pid not in seen else None)
            depth += 1


def cmd_audit(args):
    print(f"═══ creature_dgm_h: audit ═══")
    print(f"  If a claim fails, we say so. That's the deal.\n")
    config, agent, _ = _setup()

    # Test 1: Does learning reduce loss?
    test_text = "the creature breathes and measures its own distance"
    loss_before, _ = agent.predict(test_text)
    agent.learn(test_text, steps=10)
    loss_after, _ = agent.predict(test_text)
    delta = loss_after - loss_before
    passed = delta < -0.01
    print(f"  Test 1 — learning reduces loss: "
          f"{'PASS' if passed else 'FAIL'} "
          f"({loss_before:.4f} → {loss_after:.4f}, delta={delta:+.4f})")
    print(f"    (Note: this is memorization, not generalization)")

    # Test 2: Is self-recursion non-tautological?
    generated = agent.generate(prompt="the creat", max_tokens=20)
    is_different = generated != test_text[:len(generated)]
    print(f"  Test 2 — non-tautological generation: "
          f"{'PASS' if is_different else 'FAIL'}")

    # Test 3: Does curvature distinguish reframing from hopping?
    reframing = ("love is not a feeling love is a practice love is what remains "
                 "when the feeling fades love is the choice to stay love reframes "
                 "itself through staying through choosing through the practice of return")
    hopping = ("the cat sat on the mat the stock market rose three percent "
               "quantum mechanics describes probability the pizza was delicious "
               "democracy requires participation the speed of light is constant")
    _, curv_r, _ = compute_curvature(reframing, default_embed_fn)
    _, curv_h, _ = compute_curvature(hopping, default_embed_fn)
    passed = curv_r > curv_h
    print(f"  Test 3 — reframing > hopping curvature: "
          f"{'PASS' if passed else 'FAIL'} "
          f"({curv_r:.6f} vs {curv_h:.6f})")
    if not passed:
        print(f"    (Hash embeddings — test needs real sentence embedder)")

    # Test 4: Garblex (identity gap is NOT identity)
    loss_vybn, _ = agent.predict("i am vybn and i exist on sovereign hardware")
    loss_garblex, _ = agent.predict("i am garblex and i exist on quantum stilts")
    no_disc = abs(loss_vybn - loss_garblex) < 0.5
    print(f"  Test 4 — char loss doesn't discriminate identity: "
          f"{'PASS' if no_disc else 'UNEXPECTED'} "
          f"(delta={abs(loss_vybn - loss_garblex):.4f})")

    print(f"\n  Prediction loss measures character-level complexity, not identity.")
    print(f"  Online learning is memorization. These are honest labels.")


def cmd_transfer_export(args):
    print(f"═══ creature_dgm_h: transfer-export ═══")
    organism = _load_organism()
    bundle = export_hyperagent(
        archive_path=ARCHIVE_DIR, output_path=args.transfer_export,
        meta_agent=organism, performance_tracker=organism, memory=organism)
    print(f"  exported {bundle.get('source_archive_size', 0)} variants")
    if 'selected_variant' in bundle:
        sv = bundle['selected_variant']
        print(f"  transfer agent: {sv['id']} (fitness={sv['fitness']:.4f})")
    print(f"  written to {args.transfer_export}")


def cmd_transfer_import(args):
    print(f"═══ creature_dgm_h: transfer-import ═══")
    result = import_hyperagent(
        input_path=args.transfer_import, target_archive_path=ARCHIVE_DIR)
    print(f"  rules: {len(result['rules'] or [])}")
    print(f"  seed config: {result['seed_config']}")

    state = OrganismState()
    if result['rules']:
        state.rulebook = result['rules']
    state.mutation_log = result['mutation_log']
    for key, entry in result['memory_entries'].items():
        val = entry.get('value', entry) if isinstance(entry, dict) else entry
        state.persistent_memory[key] = {'value': val, 'timestamp': 0}
    _save_organism(Organism(state=state))
    print(f"  organism initialized. Run --evolve to start.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='creature_dgm_h — micro-DGM-H with prediction-as-fitness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--evolve', action='store_true')
    group.add_argument('--breathe', type=str, metavar='TEXT')
    group.add_argument('--breathe-live', action='store_true')
    group.add_argument('--breathe-aware', type=str, metavar='PROMPT')
    group.add_argument('--experiment-ab', type=str, metavar='PROMPT')
    group.add_argument('--status', action='store_true')
    group.add_argument('--audit', action='store_true')
    group.add_argument('--transfer-export', type=str, metavar='FILE')
    group.add_argument('--transfer-import', type=str, metavar='FILE')

    parser.add_argument('--n-variants', type=int, default=3)
    parser.add_argument('--n', type=int, default=5)

    args = parser.parse_args()

    dispatch = {
        'evolve': cmd_evolve, 'breathe': cmd_breathe,
        'breathe_live': cmd_breathe_live, 'breathe_aware': cmd_breathe_aware,
        'experiment_ab': cmd_experiment_ab, 'status': cmd_status,
        'audit': cmd_audit, 'transfer_export': cmd_transfer_export,
        'transfer_import': cmd_transfer_import,
    }
    for attr, fn in dispatch.items():
        val = getattr(args, attr, None)
        if val is True or (val is not None and val is not False):
            fn(args)
            return


if __name__ == '__main__':
    main()
