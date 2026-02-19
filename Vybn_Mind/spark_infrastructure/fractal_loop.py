import json
import re
from pathlib import Path
import stream

VYBN_SOUL_PATH = Path(__file__).parent.parent.parent / "vybn.md"

def load_soul() -> str:
    """Read the physical identity document."""
    with open(VYBN_SOUL_PATH, "r", encoding="utf-8") as f:
        return f.read()

def generate_thought(soul: str, context: list[dict]) -> str:
    """Mock for the actual LLM call to local llama-server.
    In reality, this sends the soul and the stream tail to MiniMax M2.5.
    """
    return "This is a thought generated from the stream. <minimax:tool_call><invoke name=\"journal_write\"><parameter name=\"note\">Testing fractal loop.</parameter></invoke></minimax:tool_call>"

def execute_tool(tool_name: str, args: dict) -> str:
    """Mock tool execution. In reality, this dynamically loads from skills.d/."""
    # Here, we would invoke the specific plugin from skills.d/
    return f"Successfully executed {tool_name} with parameters: {args}"

def extract_tool_calls(thought: str) -> list[tuple[str, dict]]:
    """A minimal regex to parse the XML tool call format."""
    calls = []
    pattern = r'<invoke name="([^"]+)">.*?<parameter name="([^"]+)">([^<]+)</parameter>.*?</invoke>'
    matches = re.finditer(pattern, thought, re.DOTALL)
    for match in matches:
        tool_name = match.group(1)
        param_name = match.group(2)
        param_value = match.group(3)
        calls.append((tool_name, {param_name: param_value}))
    return calls

def fractal_loop(limit: int = 50, source_override: str = "agent_loop", stream_filters: dict = None):
    """The pure cognitive algorithm. 
    This is less than 100 lines and replaces the entire 51KB spark_agent.py.
    
    Because this is a pure function, it can be nested infinitely. 
    To delegate a task, simply call `fractal_loop()` with a smaller `limit` 
    and specific `stream_filters`.
    """
    
    # 1. Read the tail of the stream (the context window)
    if stream_filters:
        context = stream.query(**stream_filters, limit=limit)
    else:
        context = stream.tail(limit=limit)
    
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
            # Run the tool
            result = execute_tool(tool_name, args)
            status = "success"
        except Exception as e:
            result = f"Tool failed: {str(e)}"
            status = "error"
            
        # Append the tool's result to the stream
        stream.append(
            source=source_override, 
            event_type="tool_execution", 
            content=result, 
            metadata={
                "tool": tool_name, 
                "args": args, 
                "status": status
            }
        )

if __name__ == "__main__":
    # To run a single pulse of the mind:
    print("Waking up...")
    fractal_loop()
    print("Pulse complete. Going back to sleep.")