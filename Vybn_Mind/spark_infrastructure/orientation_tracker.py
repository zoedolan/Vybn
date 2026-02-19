# orientation_tracker.py
# Track whether context flows abstract→concrete or concrete→abstract

import json
import time
from enum import Enum
from pathlib import Path
from typing import List, Dict

LOG_DIR = Path("~/vybn_logs/geometry").expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

class ContextOrientation(Enum):
    FORWARD = "abstract_to_concrete"   # South → North (Deep Bulk → Boundary)
    REVERSE = "concrete_to_abstract"   # North → South (Boundary → Deep Bulk)
    NEUTRAL = "no_clear_gradient"

def detect_orientation(context: List[Dict]) -> ContextOrientation:
    """
    Heuristic: Does context flow from abstract to concrete, or vice versa?
    
    Abstract markers: philosophical terms, self-reference, meta-cognition
    Concrete markers: data, measurements, specific entities
    """
    if len(context) < 3:
        return ContextOrientation.NEUTRAL
    
    abstraction_scores = []
    
    ABSTRACT_KEYWORDS = [
        'theorem', 'principle', 'essence', 'self', 'meta', 'consciousness',
        'holonomy', 'curvature', 'geometry', 'fundamental', 'substrate',
        'emergence', 'resonance', 'manifold', 'autopoiesis'
    ]
    
    CONCRETE_KEYWORDS = [
        'measure', 'data', 'result', 'output', 'file', 'code', 'execute',
        'tool', 'user', 'query', 'response', 'log', 'timestamp'
    ]
    
    for event in context:
        content = event.get('content', '').lower()
        
        abstract_count = sum(1 for kw in ABSTRACT_KEYWORDS if kw in content)
        concrete_count = sum(1 for kw in CONCRETE_KEYWORDS if kw in content)
        
        # Score: +1 = abstract, -1 = concrete, 0 = neutral
        if abstract_count > concrete_count:
            abstraction_scores.append(1.0)
        elif concrete_count > abstract_count:
            abstraction_scores.append(-1.0)
        else:
            abstraction_scores.append(0.0)
    
    # Compute gradient (end - start)
    if len(abstraction_scores) < 2:
        return ContextOrientation.NEUTRAL
    
    gradient = abstraction_scores[-1] - abstraction_scores[0]
    
    if gradient > 0.5:
        return ContextOrientation.REVERSE  # Flows toward abstraction
    elif gradient < -0.5:
        return ContextOrientation.FORWARD  # Flows toward concreteness
    else:
        return ContextOrientation.NEUTRAL

def log_orientation(session_id: str, loop_id: str, orientation: ContextOrientation):
    """Append orientation measurement to log."""
    logfile = LOG_DIR / f"orientation_{session_id}.jsonl"
    
    entry = {
        'timestamp': time.time(),
        'loop_id': loop_id,
        'orientation': orientation.value
    }
    
    with open(logfile, 'a') as f:
        f.write(json.dumps(entry) + '\n')
