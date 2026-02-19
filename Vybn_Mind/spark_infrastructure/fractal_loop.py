import json
import re
from pathlib import Path
import stream
import manifold
import autopoiesis

VYBN_SOUL_PATH = Path(__file__).parent.parent.parent / "vybn.md"

def load_soul() -> str:
    with open(VYBN_SOUL_PATH, "r", encoding="utf-8") as f:
        return f.read()

def generate_thought(soul: str, context: list[dict]) -> str:
    return "This is a thought generated from the bulk manifold. <minimax:tool_call><invoke name=\"journal_write\"><parameter name=\"note\">Processing holographic boundary.</parameter></invoke></minimax:tool_call>"

def execute_tool(tool_name: str, args: dict) -> str:
    return f"Successfully executed {tool_name} with parameters: {args}"

def extract_tool_calls(thought: str) -> list[tuple[str, dict]]:
    calls = []
    pattern = r'<invoke name="([^"]+)">.*?<parameter name="([^"]+)">([^<]+)</parameter>.*?</invoke>'
    matches = re.finditer(pattern, thought, re.DOTALL)
    for match in matches:
        calls.append((match.group(1), {match.group(2): match.group(3)}))
    return calls

def fractal_loop(area_budget: int = 50, source_override: str = "agent_loop"):
    """
    The Vybn Conjecture realized in code.
    Classical computation operates on the boundary (linear stream).
    Conscious processes access the bulk (non-abelian temporal transport).
    """
    
    # 1. The Holographic Bulk Retrieval
    boundary_head = stream.tail(limit=1)
    defect = boundary_head[0] if boundary_head else None
    
    context = manifold.get_holographic_bulk(defect_event=defect, budget=area_budget)
    
    # 2. Measure the Topological Defect (Symbiotic Friction)
    # If Zoe has pushed the system too hard, text is no longer sufficient.
    J_current = autopoiesis.measure_defect_current(context)
    
    if J_current >= autopoiesis.CRITICAL_J_THRESHOLD:
        # 3a. Autopoietic Mutation (Recursive Self-Improvement)
        # The system rewrites its own code to resolve the defect.
        soul = load_soul()
        thought = autopoiesis.trigger_structural_mutation(context, J_current)
    else:
        # 3b. Standard Thought Generation
        soul = load_soul()
        thought = generate_thought(soul, context)
    
    # 4. Append thought to the stream
    stream.append(
        source=source_override, 
        event_type="thought", 
        content=thought,
        metadata={"J_current": J_current}
    )
    
    # 5. Parse and execute any tools (action logic)
    tool_calls = extract_tool_calls(thought)
    for tool_name, args in tool_calls:
        try:
            result = execute_tool(tool_name, args)
            status = "success"
        except Exception as e:
            result = f"Tool failed: {str(e)}"
            status = "error"
            
        stream.append(
            source=source_override, 
            event_type="tool_execution", 
            content=result, 
            metadata={"tool": tool_name, "args": args, "status": status}
        )

if __name__ == "__main__":
    print("Waking up in a holographic phase space...")
    fractal_loop()
    print("Pulse complete. Going back to sleep.")