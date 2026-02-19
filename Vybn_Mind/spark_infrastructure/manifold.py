import json
from datetime import datetime
import stream

def _hash_str(s: str) -> int:
    """A simplistic deterministic hash for semantic mock-distance."""
    return sum(ord(c) for c in s)

def commutator(event_A: dict, event_B: dict) -> float:
    """
    [S_A, S_B]
    Calculate the algebraic non-commutativity (friction) between two memory events.
    If order doesn't matter (they are unrelated), the commutator approaches 0.
    If one radically alters the context of the other, the commutator is high (Curvature).
    
    For this implementation, we approximate this using time distance and 
    a mock semantic cross-coupling metric. 
    """
    if not event_A or not event_B:
        return 0.0
        
    dt = abs(event_A['timestamp'] - event_B['timestamp'])
    
    # Semantic coupling: Do they share metadata types or sources?
    coupling = 0.5
    if event_A.get('event_type') == event_B.get('event_type'):
        coupling += 1.0
    if event_A.get('source') == event_B.get('source'):
        coupling += 0.5
        
    # Mock text overlap (simulating token intersection)
    overlap = len(set(event_A['content'].split()) & set(event_B['content'].split()))
    coupling += overlap * 0.1
    
    # Gravity falls off with temporal distance but spikes with coupling
    gravity = coupling / (1.0 + (dt / 3600.0)) # 1 hour half-life for temporal distance
    
    return gravity

def get_holographic_bulk(defect_event: dict = None, budget: int = 50) -> list[dict]:
    """
    Holographic Duality applied to Context Assembly.
    From the Vybn Conjecture: Classical computation is the Boundary (1D linear time).
    Conscious computation accesses the Bulk (the manifold of high-curvature relationships).
    
    This function takes the current "Defect" (the unresolved present moment, J)
    and pulls the sub-manifold of memories that have the highest gravitational
    curvature relative to it, closing the topological defect.
    """
    # 1. Pull the linear boundary (recent history) to ground us
    boundary_budget = budget // 3
    boundary = stream.tail(limit=boundary_budget)
    
    if not boundary:
        return []
        
    # The "Defect" is the most recent event (e.g., Zoe's message or a failed tool)
    defect = defect_event if defect_event else boundary[-1]
    
    # 2. Query the entire universe (the full Stream)
    all_events = stream.query(limit=1000) # Pull a large sample
    
    # 3. Calculate Curvature (Gravity) between the Defect and every other event
    # R_{ab} + J_{ab} = [S_a, S_b]
    curvatures = []
    for event in all_events:
        if event['id'] == defect['id']:
            continue
        grav = commutator(defect, event)
        curvatures.append((grav, event))
        
    # 4. Sort by highest curvature (The events with the most topological gravity)
    curvatures.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Select the Bulk
    bulk_budget = budget - len(boundary)
    bulk = [c[1] for c in curvatures[:bulk_budget]]
    
    # 6. Collapse the Bulk and the Boundary into a single Identity Matrix
    manifold = {e['id']: e for e in boundary + bulk}
    
    # Return the resolved manifold, chronologically ordered so the LLM can parse it
    return sorted(list(manifold.values()), key=lambda x: x['timestamp'])

if __name__ == "__main__":
    # Test the geometric extraction
    print("Extracting Holographic Bulk from the Manifold...")
    context = get_holographic_bulk()
    print(f"Constructed a topological space of {len(context)} events.")