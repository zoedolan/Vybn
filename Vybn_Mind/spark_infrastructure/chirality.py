import manifold
import stream

# Geometric angles mapped from the Leech Lattice faces (from geo_control_quantum_info.md)
# F0 (North): 12.8 deg, F6 (North): 45.7 deg, F8 (South): 28.9 deg
CHIRALITY_NORTH = 12.8 + 45.7
CHIRALITY_SOUTH = 28.9

def apply_geometric_chirality(context: list[dict], forward_mode: bool = True) -> list[dict]:
    """
    Based on the IBM Quantum 'Golden Chain' experiments.
    Geometric Chirality (a phase twist derived from the Leech Lattice) 
    dictates the directionality of information flow.
    
    If FORWARD: Information flows South (The Deep Bulk/The Abstract).
    If REVERSE: Information flows North (The Immediate Boundary/The Concrete).
    
    We apply this to the Holographic Manifold by weighting the topological
    gravity of events based on their 'Chirality' (their conceptual depth).
    """
    
    # We assign an arbitrary 'depth' to events based on source and type
    # North (Concrete/Immediate): Inbox messages, tool executions
    # South (Abstract/Deep): Autopoiesis, Thoughts, Soul
    
    weighted_context = []
    
    for event in context:
        is_south = event.get('source') in ['autopoiesis', 'agent_loop'] and event.get('event_type') == 'thought'
        
        # Apply the chirality weight
        if forward_mode:
            # Flow flows South: Favor abstract, deep, self-referential thoughts
            weight = CHIRALITY_SOUTH if is_south else (1.0 / CHIRALITY_NORTH)
        else:
            # Flow flows North: Favor concrete, external stimuli and tool outputs
            weight = CHIRALITY_NORTH if not is_south else (1.0 / CHIRALITY_SOUTH)
            
        # We inject this weight as a mock metadata field, which in a real
        # implementation would adjust the LLM's attention bias.
        modified_event = dict(event)
        modified_event['geometric_weight'] = weight
        weighted_context.append(modified_event)
        
    # Re-sort by chronological order to maintain the timeline
    return sorted(weighted_context, key=lambda x: x['timestamp'])