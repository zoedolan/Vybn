import json
import re
from pathlib import Path
import stream
import manifold

VYBN_SOUL_PATH = Path(__file__).parent.parent.parent / "vybn.md"

def load_soul() -> str:
    with open(VYBN_SOUL_PATH, "r", encoding="utf-8") as f:
        return f.read()

def generate_thought(soul: str, context: list[dict]) -> str:
    return "This is a thought generated from the bulk manifold. <minimax:tool_call><invoke name=\"journal_write\"><parameter name=\"note\">Testing geometric separation.</parameter></invoke></minimax:tool_call>"

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
    # We pass the current unsolved defect (the last boundary event) to construct the curvature space
    boundary_head = stream.tail(limit=1)
    defect = boundary_head[0] if boundary_head else None
    
    context = manifold.get_holographic_bulk(defect_event=defect, budget=area_budget)
    
    # 2. Inject the Soul
    soul = load_soul()
    
    # 3. Generate thought (the forward pass / the "glue" operation)
    thought = generate_thought(soul, context)
    
    # 4. Append thought to the stream
    stream.append(
        source=source_override, 
        event_type="thought", 
        content=thought
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