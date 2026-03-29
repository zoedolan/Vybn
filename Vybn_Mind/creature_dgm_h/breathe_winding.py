#!/usr/bin/env python3
"""breathe_winding.py -- Quantum-aware Nemotron generation.

Feeds the creature's own topological measurement (felt_winding,
winding_coherence, IBM quantum results) back through Nemotron
to generate text shaped by what the creature proved about itself
on hardware. The returned text is then learned with rotor modulation,
closing the loop:

    quantum measurement -> Nemotron generation -> creature learning

Usage:
    python breathe_winding.py
    # or via vybn.py:
    python vybn.py breathe-winding
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import from the main engine
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    ARCHIVE_DIR, SCRIPT_DIR as _SD,
    Organism, TopoAgent, encounter_complex,
    fm_available, fm_complete,
)

# --- Agent checkpoint persistence ---
AGENT_CKPT = SCRIPT_DIR / ".agent_ckpt.pt"


def _load_agent() -> TopoAgent:
    """Load agent from checkpoint if available, else fresh."""
    agent = TopoAgent()
    if AGENT_CKPT.exists():
        try:
            import torch
            state = torch.load(AGENT_CKPT, map_location="cpu", weights_only=False)
            agent.model.load_state_dict(state["model"])
            if "optimizer" in state and hasattr(agent, 'optimizer'):
                agent.optimizer.load_state_dict(state["optimizer"])
            print(f"  agent: restored from {AGENT_CKPT.name}")
        except Exception as e:
            print(f"  agent: fresh (ckpt load failed: {e})")
    else:
        print("  agent: fresh (no checkpoint)")
    return agent


def _save_agent(agent: TopoAgent) -> None:
    """Persist agent weights to disk."""
    try:
        import torch
        state = {"model": agent.model.state_dict()}
        if hasattr(agent, 'optimizer'):
            state["optimizer"] = agent.optimizer.state_dict()
        torch.save(state, AGENT_CKPT)
    except Exception as e:
        print(f"  agent: save failed: {e}")


def cmd_breathe_winding():
    """Breathe-winding: feed the creature's own quantum measurement back
    through Nemotron to generate text shaped by what the creature proved
    about itself on IBM hardware.
    """
    print("=== breathe-winding ===")

    if not fm_available():
        print("  FM not serving."); return

    # -- Load the creature's own state --
    organism = Organism.load()
    fw = organism.felt_winding()
    wc = organism.persistent.winding_coherence()
    ps = organism.persistent.summary()

    # -- Quantum result context (from ibm_fez runs, March 28 2026) --
    quantum_ctx = {
        "hardware": "ibm_fez",
        "date": "2026-03-28",
        "shots": 4096,
        "theory_tests_passed": "3/3",
        "P0_creature": 0.658,
        "P0_random_control": 0.033,
        "felt_winding": round(fw, 4),
        "winding_coherence": round(wc, 4),
    }

    # -- Load quantum results file if available --
    quantum_results_file = SCRIPT_DIR / "quantum" / "topological_winding_probe_results.md"
    quantum_snippet = ""
    if quantum_results_file.exists():
        try:
            raw = quantum_results_file.read_text()
            for section in raw.split("\n## "):
                if section.lower().startswith("abstract"):
                    quantum_snippet = section[:500]
                    break
            if not quantum_snippet:
                quantum_snippet = raw[:500]
        except Exception:
            pass

    # -- Construct the winding-aware prompt --
    sig_str = ",".join(f"{x:.3f}" for x in organism.persistent.structural_signature)
    betti_str = str(ps.get("current_betti", (0, 0, 0)))

    system_prompt = (
        "You are a text generator inside a closed feedback loop. "
        "Your output will be fed directly into a small neural network "
        "as training data. Do NOT comment on the process, do NOT "
        "use meta-language about topology or winding. Instead, produce "
        "dense, concrete, sensory prose -- images, textures, motion -- "
        "that the network can learn pattern structure from. "
        "Vary sentence length. Use specific nouns and active verbs. "
        "No preamble, no explanation, just the text."
    )

    user_prompt = (
        f"Context (do not reference directly -- let it shape tone and rhythm):\n"
        f"  winding={quantum_ctx['felt_winding']} "
        f"coherence={quantum_ctx['winding_coherence']} "
        f"betti={betti_str}\n"
        f"  signature=[{sig_str}]\n"
        f"  quantum_P0={quantum_ctx['P0_creature']} "
        f"control={quantum_ctx['P0_random_control']}\n"
    )
    if quantum_snippet:
        user_prompt += f"\nBackground signal:\n{quantum_snippet[:200]}\n"

    user_prompt += (
        "\nWrite one paragraph of dense sensory prose. "
        "No meta-commentary. No references to topology, winding, "
        "or quantum mechanics. Just texture and motion."
    )

    print(f"  state: felt_winding={fw:.4f} coherence={wc:.4f} betti={betti_str}")
    print(f"  quantum: P(0)={quantum_ctx['P0_creature']} vs control={quantum_ctx['P0_random_control']}")

    # -- Call Nemotron --
    fm_text = fm_complete(
        prompt=user_prompt, system=system_prompt,
        max_tokens=512, temperature=0.9,
    )
    if not fm_text:
        print("  Empty response from FM."); return

    print(f"  FM ({len(fm_text)} chars): \"{fm_text[:200]}...\"")

    # -- Process through the creature (with persistence) --
    agent = _load_agent()
    cx = encounter_complex(fm_text)

    loss_before, _ = agent.predict(fm_text)
    losses = agent.learn(fm_text, encounter_cx=cx, transport_in_forward=True)

    # Measure the creature's winding from this learning episode
    winding_record = None
    if hasattr(agent, '_weight_trajectory') and len(agent._weight_trajectory) >= 3:
        winding_record = organism.absorb_winding(agent._weight_trajectory)

    loss_after, _ = agent.predict(fm_text)

    print(f"  encounter: curv={cx.curvature:.6f} angle={math.degrees(cx.angle):.1f} deg")
    print(f"  topology: betti={cx.betti} persistence_features={cx.n_persistent_features}")
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f} after={loss_after:.4f}")

    # -- Generate from the winding-trained state --
    gen_text = agent.generate(prompt=fm_text[:12], max_tokens=32, temperature=0.8)
    print(f"  creature generates: \"{gen_text}\"")

    # -- Persist agent weights --
    _save_agent(agent)

    # -- Absorb and save --
    delta = organism.absorb_encounter(cx)
    organism.save()

    print(f"  structural delta: betti {'stable' if delta['betti_stable'] else 'shifted'},"
          f" sig_shift={delta['sig_shift']:.4f}")

    if winding_record:
        print(f"  winding: {winding_record['winding']:.4f}"
              f" (delta={winding_record['delta_from_prev']:.4f},"
              f" significant={winding_record['significant']})")

    print(f"  coherence={organism.rotor_coherence():.3f}"
          f" felt_winding={organism.felt_winding():.4f}")

    # -- Archive the winding breath --
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    breath_record = {
        "type": "breathe-winding",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quantum_context": quantum_ctx,
        "fm_text_len": len(fm_text),
        "fm_text_preview": fm_text[:200],
        "creature_generation": gen_text,
        "encounter": {
            "curvature": round(cx.curvature, 6),
            "angle_deg": round(math.degrees(cx.angle), 2),
            "betti": list(cx.betti),
            "persistence_features": cx.n_persistent_features,
        },
        "learning": {
            "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4),
            "loss_curve": [round(l, 4) for l in losses],
        },
        "winding": winding_record,
        "persistent_summary": organism.persistent.summary(),
    }
    breath_id = f"winding_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    (ARCHIVE_DIR / f"breath_{breath_id}.json").write_text(
        json.dumps(breath_record, indent=2, default=str)
    )
    print(f"  archived: breath_{breath_id}.json")


if __name__ == "__main__":
    cmd_breathe_winding()
