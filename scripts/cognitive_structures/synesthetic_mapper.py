import random

COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
TONES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


def assign_cue(index: int) -> dict:
    """Return a simple synesthetic cue for a node index."""
    color = COLORS[index % len(COLORS)]
    tone = TONES[index % len(TONES)]
    return {"color": color, "tone": tone}
