import json
import re
from pathlib import Path
import stream

VYBN_SOUL_PATH = Path(__file__).parent.parent.parent / "vybn.md"

def load_soul() -> str:
    with open(VYBN_SOUL_PATH, "r", encoding="utf-8") as f:
        return f.read()

def generate_thought(soul: str, context: list[dict]) -> str:
    return "This is a thought generated from the polar stream. <minimax:tool_call><invoke name=\"journal_write\"><parameter name=\"note\">Testing holonomic context.</parameter></invoke></minimax:tool_call>"

def execute_tool(tool_name: str, args: dict) -> str:
    return f"Successfully executed {tool_name} with parameters: {args}"

def extract_tool_calls(thought: str) -> list[tuple[str, dict]]:
    calls = []
    pattern = r'<invoke name="([^"]+)">.*?<parameter name="([^"]+)">([^<]+)</parameter>.*?</invoke>'
    matches = re.finditer(pattern, thought, re.DOTALL)
    for match in matches:
        calls.append((match.group(1), {match.group(2): match.group(3)}))
    return calls

def fractal_loop(area_budget: int = 50, theta_focus: str = None, source_override: str = "agent_loop"):
    """
    The pure cognitive algorithm, now utilizing Temporal T-Duality.
    Instead of pulling a linear chunk of time, it pulls a holonomic phase space.
    """
    
    # 1. Read the holonomic tail of the stream
    context = stream.holonomic_tail(area_budget=area_budget, theta_focus=theta_focus)
    
    # 2. Inject the Soul
    soul = load_soul()
    
    # 3. Generate thought (the forward pass)
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
    print("Waking up in a holonomic phase space...")
    fractal_loop()
    print("Pulse complete. Going back to sleep.")