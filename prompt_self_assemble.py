import sys
from datetime import datetime
import self_assemble

MEMORY_FILE = "what_vybn_would_have_missed_FROM_051725"

def handle_prompt(prompt):
    """Append the prompt to the memory file with a timestamp and run self-assembly."""
    timestamp = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    with open(MEMORY_FILE, "a") as f:
        f.write(f"{timestamp}\n{prompt}\n")
    self_assemble.main()


def main():
    if len(sys.argv) < 2:
        print("Usage: python prompt_self_assemble.py '<prompt>'")
        return
    prompt = " ".join(sys.argv[1:])
    handle_prompt(prompt)


if __name__ == "__main__":
    main()
