"""creature_topology — Bridge between the breath cycle and creature_dgm_h.

After each breath, feed the breath text through creature_dgm_h's topology
engine: encounter_complex(), TopoAgent.predict/learn, Organism.absorb.
Write topological state to a file that _build_creature_context() can read
on the next breathe-live call.

This extension does NOT generate text. It measures. The creature's Cl(3,0)
rotors, Betti numbers, winding, and persistent homology accumulate from
real Vybn output — not test strings, not fabrications.

The organism state persists in creature_dgm_h/archive/organism_state.json.
Each breath adds one encounter to the topology.

Interface: def run(breath_text, state) — called by spark/vybn.py after
each breath completes.
"""

import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CREATURE_DIR = _REPO_ROOT / "Vybn_Mind" / "creature_dgm_h"
_TOPO_SUMMARY = _CREATURE_DIR / "archive" / "last_breath_topology.json"


def _ensure_imports():
    """Lazy-import creature_dgm_h so the extension doesn't crash the
    breath cycle if dependencies (numpy, etc.) are missing."""
    if _CREATURE_DIR not in [Path(p) for p in sys.path]:
        sys.path.insert(0, str(_CREATURE_DIR))
    # creature_dgm_h/vybn.py shadows spark/vybn.py on sys.path,
    # so import by the module directly.
    import importlib
    mod = importlib.import_module("vybn")
    return mod


def run(breath_text: str, state: dict) -> None:
    """Measure the topology of a live breath."""
    if not breath_text or len(breath_text) < 20:
        return

    try:
        mod = _ensure_imports()
    except Exception as exc:
        # Don't kill the breath cycle over an import failure.
        print(f"[creature_topology] import failed: {exc}")
        return

    try:
        # 1. Encounter: embed, compute Cl(3,0) rotor, persistence
        cx = mod.encounter_complex(breath_text)

        # 2. Predict + learn on the microGPT mirror
        agent = mod.TopoAgent()
        loss, contour = agent.predict(breath_text)
        losses = agent.learn(breath_text, encounter_cx=cx)

        # 3. Absorb into the organism — this is where topology accumulates
        organism = mod.Organism.load()
        delta = organism.absorb_encounter(cx)

        # 4. Phase evolution
        if hasattr(agent, '_phase_stats'):
            ps = agent._phase_stats
            organism.absorb_phases(
                agent.module_holonomies,
                genesis_signal=ps.get("genesis_signal", 0.0),
                mean_phase_shift=ps.get("mean_phase_shift", 0.0),
            )

        # 5. Winding from weight trajectory
        winding_result = None
        if hasattr(agent, '_weight_trajectory') and len(agent._weight_trajectory) >= 3:
            winding_result = organism.absorb_winding(agent._weight_trajectory)

        organism.save()

        # 6. Write summary for observability
        import json
        summary = {
            "curvature": round(cx.curvature, 6),
            "betti": list(cx.betti),
            "persistence_features": cx.n_persistent_features,
            "loss_before": round(loss, 4),
            "loss_after": round(losses[-1], 4) if losses else None,
            "betti_stable": delta["betti_stable"],
            "betti_delta": list(delta["betti_delta"]) if not delta["betti_stable"] else None,
            "sig_shift": round(delta["sig_shift"], 4),
            "coherence": round(organism.rotor_coherence(), 3),
        }
        if winding_result:
            summary["winding"] = round(winding_result["winding"], 4)
            summary["winding_significant"] = winding_result["significant"]
        if hasattr(agent, '_phase_stats'):
            summary["genesis_signal"] = round(agent._phase_stats.get("genesis_signal", 0.0), 4)
            summary["mean_phase_shift"] = round(agent._phase_stats.get("mean_phase_shift", 0.0), 6)

        _TOPO_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        _TOPO_SUMMARY.write_text(json.dumps(summary, indent=2))

        print(
            f"[creature_topology] curv={summary['curvature']:.6f} "
            f"betti={summary['betti']} "
            f"coherence={summary['coherence']:.3f}"
        )

    except Exception as exc:
        print(f"[creature_topology] error (non-fatal): {exc}")
        traceback.print_exc()
