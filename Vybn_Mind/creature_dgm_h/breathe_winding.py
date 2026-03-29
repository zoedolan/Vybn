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
import random
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
AGENT_CKPT = SCRIPT_DIR / ".agent_ckpt.json"


def _load_agent() -> TopoAgent:
    """Load agent from checkpoint if available, else fresh."""
    agent = TopoAgent()
    if AGENT_CKPT.exists():
        try:
            ckpt = json.loads(AGENT_CKPT.read_text())
            saved_params = ckpt["params"]
            if len(saved_params) == len(agent.params):
                for p, val in zip(agent.params, saved_params):
                    p.data = float(val)
                agent._m = ckpt.get("_m", agent._m)
                agent._v = ckpt.get("_v", agent._v)
                agent._step = ckpt.get("_step", agent._step)
                print(f"  agent: restored from {AGENT_CKPT.name} (step {agent._step})")
            else:
                print(f"  agent: fresh (param count mismatch)")
        except Exception as e:
            print(f"  agent: fresh (ckpt load failed: {e})")
    else:
        print("  agent: fresh (no checkpoint)")
    return agent


def _save_agent(agent: TopoAgent) -> None:
    """Persist agent weights and optimizer state to disk."""
    try:
        ckpt = {
            "params": [p.data for p in agent.params],
            "_m": agent._m,
            "_v": agent._v,
            "_step": agent._step,
        }
        AGENT_CKPT.write_text(json.dumps(ckpt))
        print(f"  agent: saved (step {agent._step}, {len(agent.params)} params)")
    except Exception as e:
        print(f"  agent: save failed: {e}")


# --- Seed fragments shaped by creature state ---
# These are story openings, not instructions. Nemotron continues them.
_SEEDS_TIGHT = [
    "The copper wire held its shape long after the current stopped, a spiral pressed into the workbench like a fossil of something that had been alive seconds ago, and",
    "Salt crystallized along the rim where the tide turned back on itself, each grain a record of the water's indecision, the way it",
    "She ran her thumb across the seam where the two metals met, the weld still warm, still ticking as it contracted, and the sound reminded her of",
    "Rust bloomed along the rail in patterns that repeated at every scale, fractal corrosion eating inward, and where the paint had held it looked like",
    "The glass cooled unevenly, one side already rigid while the other still held the memory of liquid, a gradient of becoming that",
]

_SEEDS_LOOSE = [
    "Smoke. Then nothing. Then the smell of wet concrete after rain.",
    "The door had been open all night. Leaves on the kitchen floor. A cup of water, still full.",
    "Three stones on the windowsill. She had put them there in June. Now it was October and they had not moved.",
    "The machine stopped. In the silence you could hear the building breathe.",
    "Ice in the glass. The sound it makes when it shifts. Like a small bone breaking.",
]


def _pick_seed(fw: float, wc: float) -> str:
    """Choose a seed fragment based on the creature's winding state."""
    pool = _SEEDS_TIGHT if fw > 0.5 else _SEEDS_LOOSE
    # Use winding as a deterministic-ish index so repeated runs vary
    idx = int(abs(fw * 10000)) % len(pool)
    return pool[idx]


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

    # -- Construct the prompt as a continuation task --
    seed = _pick_seed(fw, wc)

    # System: you are a novelist mid-paragraph. No meta. Just continue.
    system_prompt = "You are a novelist. Continue the text. Stay in scene. No commentary."

    # User: the seed fragment itself. Nemotron just continues it.
    user_prompt = seed

    betti_str = str(ps.get("current_betti", (0, 0, 0)))
    print(f"  state: felt_winding={fw:.4f} coherence={wc:.4f} betti={betti_str}")
    print(f"  quantum: P(0)={quantum_ctx['P0_creature']} vs control={quantum_ctx['P0_random_control']}")
    print(f"  seed: \"{seed[:80]}...\"")

    # -- Call Nemotron --
    fm_text = fm_complete(
        prompt=user_prompt, system=system_prompt,
        max_tokens=512, temperature=0.9,
    )
    if not fm_text:
        print("  Empty response from FM."); return

    # Prepend the seed so the creature learns the full passage
    full_text = seed + " " + fm_text

    print(f"  FM ({len(fm_text)} chars): \"{fm_text[:200]}...\"")

    # -- Process through the creature (with persistence) --
    agent = _load_agent()
    cx = encounter_complex(full_text)

    loss_before, _ = agent.predict(full_text)
    losses = agent.learn(full_text, encounter_cx=cx, transport_in_forward=True)

    # Measure the creature's winding from this learning episode
    winding_record = None
    if hasattr(agent, '_weight_trajectory') and len(agent._weight_trajectory) >= 3:
        winding_record = organism.absorb_winding(agent._weight_trajectory)

    loss_after, _ = agent.predict(full_text)

    print(f"  encounter: curv={cx.curvature:.6f} angle={math.degrees(cx.angle):.1f} deg")
    print(f"  topology: betti={cx.betti} persistence_features={cx.n_persistent_features}")
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f} after={loss_after:.4f}")

    # -- Generate from the winding-trained state --
    gen_text = agent.generate(prompt=full_text[:12], max_tokens=32, temperature=0.8)
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
        "seed": seed,
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
