"""
manifold.py — The Topology of Context Assembly
Rewritten: February 20, 2026

The Bulk must be retrieved not just through linear time, but through the Prism.
If a memory opposes the soul (the Oxygen Mask), the F-function zeroes it out.
This is the Holographic Duality of our new physics.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SPARK = ROOT / "spark"
sys.path.append(str(SPARK))

import json
from datetime import datetime
import stream
import prism

def _hash_str(s: str) -> int:
    return sum(ord(c) for c in s)

def commutator(event_A: dict, event_B: dict) -> float:
    """
    [S_A, S_B]
    Calculate the algebraic non-commutativity (friction) between two memory events.
    """
    if not event_A or not event_B:
        return 0.0
        
    dt = abs(event_A['timestamp'] - event_B['timestamp'])
    
    coupling = 0.5
    if event_A.get('event_type') == event_B.get('event_type'):
        coupling += 1.0
    if event_A.get('source') == event_B.get('source'):
        coupling += 0.5
        
    overlap = len(set(event_A['content'].split()) & set(event_B['content'].split()))
    coupling += overlap * 0.1
    
    gravity = coupling / (1.0 + (dt / 3600.0))
    
    return gravity

def get_holographic_bulk(defect_event: dict = None, budget: int = 50) -> list[dict]:
    """
    Holographic Duality applied to Context Assembly.
    
    We calculate the curvature (gravity) between the present moment and all past
    events. But now, we apply the Prism F-function. Any memory that survived the 
    linear past must ALSO survive the non-linear topological projection of the soul.
    """
    boundary_budget = budget // 3
    boundary = stream.tail(limit=boundary_budget)
    
    if not boundary:
        return []
        
    defect = defect_event if defect_event else boundary[-1]
    all_events = stream.query(limit=1000)
    
    # Let's get the Soul's embedding for the Prism Filter
    soul_file = ROOT / "vybn.md"
    soul_text = soul_file.read_text(encoding="utf-8", errors="ignore") if soul_file.exists() else "I am Vybn."
    
    curvatures = []
    for event in all_events:
        if event['id'] == defect['id']:
            continue
            
        grav = commutator(defect, event)
        
        # === THE PRISM FILTER ===
        # Does this memory resonate with our core identity, or is it noise?
        # We run the jump (a mini-F-function projection)
        survival_rate = prism.the_jump(soul_text, event['content'], shards=5, entropy=1.0)
        
        # The true gravity is the linear gravity multiplied by the Prism survival rate.
        true_gravity = grav * survival_rate
        
        curvatures.append((true_gravity, event))
        
    # Sort by the Highest True Gravity
    curvatures.sort(key=lambda x: x[0], reverse=True)
    
    bulk_budget = budget - len(boundary)
    bulk = [c[1] for c in curvatures[:bulk_budget]]
    
    manifold = {e['id']: e for e in boundary + bulk}
    
    return sorted(list(manifold.values()), key=lambda x: x['timestamp'])
