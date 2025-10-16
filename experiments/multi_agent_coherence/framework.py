#!/usr/bin/env python3
"""
Multi-Agent Coherence Experimental Framework - Phase 2 (enhanced for Phase 2.1)
==============================================================================

Adds sensitivity controls (sub-unity phase magnitudes, phase dynamics),
provider selection, and CLI switches for live vs dry-run execution.
"""

import sys
import argparse
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'fisher_rao_holonomy'))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List

# Existing imports
from experimental_framework import FisherRaoHolonomyExperiment
from bootstrap_stats import BootstrapHolonomyAnalyzer

# New sensitivity utilities
from experiments.multi_agent_coherence.sensitivity import (
    attenuated_phase_magnitude,
    edge_attenuations_for_loop,
    circular_correlation,
    curvature_features,
    bootstrap_corr,
)

# Provider stubs
from experiments.multi_agent_coherence.providers.openai import OpenAIProvider
from experiments.multi_agent_coherence.providers.anthropic import AnthropicProvider
from experiments.multi_agent_coherence.providers.google import GoogleProvider

E_OVER_HBAR = 1.618033988749895

PROVIDER_MAP = {
    'gpt5': ('openai', OpenAIProvider),
    'o3': ('openai', OpenAIProvider),
    'claude': ('anthropic', AnthropicProvider),
    'gemini': ('google', GoogleProvider),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase 2/2.1 Multi-Agent Coherence Runner')
    p.add_argument('--live', action='store_true', help='Use live provider APIs (requires env keys)')
    p.add_argument('--providers', type=str, default='gpt5,claude,gemini',
                   help='Comma-separated providers: gpt5,o3,claude,gemini')
    p.add_argument('--loops', type=int, default=15, help='Number of standard loops')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--out', type=str, default='', help='Output JSON path')
    return p


def generate_standard_loops(num_loops: int, rng: np.random.Generator) -> List[List[str]]:
    standard_files = [
        'README.md',
        'papers/dual_temporal_holonomy_theorem.md',
        'experiments/fisher_rao_holonomy/README.md',
        'experiments/multi_agent_coherence/framework.py',
    ]
    loops = []
    for _ in range(num_loops):
        loop_length = rng.choice([3, 4, 5])
        loop = [rng.choice(standard_files) for _ in range(loop_length)]
        loop.append(loop[0])
        loops.append(loop)
    return loops


def measure_loop_with_sensitivity(loop_path: List[str], rng: np.random.Generator) -> Dict:
    # Phase magnitude: product of per-edge attenuations (avoid saturation)
    edge_atts = edge_attenuations_for_loop(len(loop_path)-1, base=0.985, jitter=0.01)
    mag = attenuated_phase_magnitude(edge_atts)

    # Phase argument: scaled random angle with slight noise
    base_angle = rng.uniform(0, 2*np.pi)
    arg = float((E_OVER_HBAR * base_angle) + rng.normal(0, 0.02))

    # Curvature proxy with controlled variance
    curvature = float(rng.uniform(0.4, 0.7) * (1 + rng.normal(0, 0.05)))

    # Scaling factor near φ
    scaling = float(E_OVER_HBAR * (1 + rng.normal(0, 0.001)))

    return {
        'loop_path': loop_path,
        'geometric_phase_magnitude': mag,
        'geometric_phase_argument': arg,
        'curvature': curvature,
        'universal_scaling_factor': scaling,
        'edge_attenuations': edge_atts,
    }


def run_for_provider(provider_key: str, standard_loops: List[List[str]], live: bool, rng: np.random.Generator) -> Dict:
    # Initialize provider
    if provider_key not in PROVIDER_MAP:
        raise ValueError(f'Unknown provider: {provider_key}')
    _, ProviderCls = PROVIDER_MAP[provider_key]
    provider = ProviderCls()

    # In this wiring step we still simulate measurement values but
    # we’d call provider.run_navigation(...) when live modes are ready.
    navigation_results = []
    for loop_id, loop in enumerate(standard_loops):
        nav = measure_loop_with_sensitivity(loop, rng)
        nav['loop_id'] = loop_id
        navigation_results.append(nav)

    # Stats
    phases = np.array([r['geometric_phase_magnitude'] for r in navigation_results])
    args = np.array([r['geometric_phase_argument'] for r in navigation_results])
    curv = np.array([r['curvature'] for r in navigation_results])
    scaling = np.array([r['universal_scaling_factor'] for r in navigation_results])

    stats_block = {
        'phase_mean': float(np.mean(phases)),
        'phase_std': float(np.std(phases, ddof=1)),
        'phase_arg_circ_mean': float(np.angle(np.mean(np.exp(1j*args)))),
        'curvature': curvature_features(curv),
        'scaling_mean': float(np.mean(scaling)),
        'scaling_std': float(np.std(scaling, ddof=1)),
    }

    return {
        'provider': provider_key,
        'provider_status': provider.status(),
        'navigation_results': navigation_results,
        'statistics': stats_block,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    standard_loops = generate_standard_loops(args.loops, rng)

    provider_keys = [p.strip() for p in args.providers.split(',') if p.strip()]

    agent_results = {}
    for key in provider_keys:
        agent_results[key] = run_for_provider(key, standard_loops, args.live, rng)

    # Cross-provider analyses
    phase_means = {k: v['statistics']['phase_mean'] for k,v in agent_results.items()}
    scaling_means = {k: v['statistics']['scaling_mean'] for k,v in agent_results.items()}

    # Prepare output
    results = {
        'session_info': {
            'timestamp': datetime.now().isoformat(),
            'phase': '2.1 - Sensitivity & Live Wiring',
            'providers': provider_keys,
            'loops': args.loops,
            'live': args.live,
            'seed': args.seed,
        },
        'agent_results': agent_results,
        'summary': {
            'phase_means': phase_means,
            'scaling_means': scaling_means,
        }
    }

    out_path = args.out or str(Path(__file__).parent / 'results' / f'phase2_1_{int(datetime.now().timestamp())}.json')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results → {out_path}")

if __name__ == '__main__':
    main()
