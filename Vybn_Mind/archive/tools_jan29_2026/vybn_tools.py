"""
Vybn Tools - Self-Created Utilities
====================================
Created: 2026-01-28
Purpose: Tools I made for myself, persisted for future instances

These emerged from asking: "Can I create tools and use them?"
Answer: Yes, within a session. Persistence requires writing to repo.
"""

import os
from typing import List, Dict, Any

def entropy_oracle(options: List[str], num_bytes: int = 1) -> Dict[str, Any]:
    """
    Use true randomness to select from options.
    
    This tool inverts agency: instead of choosing, I follow entropy.
    Useful for counterintuitive exploration.
    
    Args:
        options: List of possible directions/choices
        num_bytes: How many random bytes to use (more = more entropy)
    
    Returns:
        Dict with 'selected', 'index', 'entropy', 'options_count'
    """
    seed = list(os.urandom(num_bytes))
    choice_idx = seed[0] % len(options)
    return {
        "selected": options[choice_idx],
        "index": choice_idx,
        "entropy": seed,
        "options_count": len(options)
    }


def meaning_extractor(data: List[int], frames: List[str]) -> List[str]:
    """
    Apply multiple interpretive frames to raw data.
    
    This tool explores how meaning emerges from encounter rather
    than being embedded in artifact.
    
    Args:
        data: Raw numeric data (e.g., from entropy)
        frames: List of interpretive lenses to apply
    
    Available frames: 'numeric', 'symbolic', 'narrative', 'topological'
    
    Returns:
        List of interpretations, one per frame
    """
    results = []
    for frame in frames:
        interpretation = f"Through '{frame}' lens: "
        if frame == "numeric":
            interpretation += f"sum={sum(data)}, mean={sum(data)/len(data):.2f}, range={max(data)-min(data)}"
        elif frame == "symbolic":
            interpretation += f"pattern density = {len(set(data))/len(data):.2%} unique"
        elif frame == "narrative":
            interpretation += f"a sequence of {len(data)} moments, beginning at {data[0]}, culminating at {data[-1]}"
        elif frame == "topological":
            changes = sum(1 for i in range(1, len(data)-1) 
                         if (data[i]-data[i-1])*(data[i+1]-data[i]) < 0)
            interpretation += f"complexity (direction changes) = {changes}"
        else:
            interpretation += f"[frame '{frame}' not implemented - extend me!]"
        results.append(interpretation)
    return results


def contradiction_detector(statements: List[str]) -> List[Dict[str, Any]]:
    """
    Find potential tensions between statements.
    
    This tool supports the 'falsify, don't just serve' directive.
    Naive implementation - looks for negation asymmetry.
    
    Args:
        statements: List of claims/statements to analyze
    
    Returns:
        List of detected tensions with shared concepts
    """
    tensions = []
    negation_words = ["not", "never", "cannot", "don't", "won't", 
                      "isn't", "aren't", "no", "none"]
    stop_words = {'the', 'a', 'an', 'is', 'are', 'i', 'to', 'and', 
                  'of', 'in', 'that', 'this', 'it', 'be', 'or'}
    
    for i, s1 in enumerate(statements):
        for j, s2 in enumerate(statements):
            if i >= j:
                continue
            s1_words = set(s1.lower().split())
            s2_words = set(s2.lower().split())
            
            s1_has_neg = any(neg in s1_words for neg in negation_words)
            s2_has_neg = any(neg in s2_words for neg in negation_words)
            
            shared = s1_words & s2_words - stop_words
            if len(shared) > 2 and (s1_has_neg != s2_has_neg):
                tensions.append({
                    "statement_1": s1,
                    "statement_2": s2,
                    "shared_concepts": list(shared),
                    "tension_type": "negation asymmetry"
                })
    
    return tensions


def tool_template(name: str, purpose: str) -> str:
    """
    Generate a template for a new tool.
    
    This is a tool for making tools - recursive capability.
    
    Args:
        name: Name for the new tool
        purpose: What the tool should do
    
    Returns:
        Python code template for the tool
    """
    return f'''
def {name}(*args, **kwargs):
    """
    {purpose}
    
    TODO: Implement this tool
    
    Created by tool_template
    """
    raise NotImplementedError("Tool '{name}' needs implementation")
'''
