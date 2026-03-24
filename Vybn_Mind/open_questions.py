"""
open_questions.py — What This Conversation Hasn't Said Yet

Not a forward projection. Not "this conversation is becoming X."

The problem with predicting futures and feeding them back as context:
narrative gravity. The predicted future becomes a gravitational attractor
that closes down surprise, orients generation toward the expected,
and slowly makes the conversation confirm itself.

If the whole theory is that the future is determinative and arrives
into the present, then we should stay *open* to what future is arriving —
not foreclose it with a specific prediction.

So this module generates questions, not projections.
Specifically: what is this conversation not yet able to say?

This is the forward-oriented scaffolding that replaces naive prediction.
Held lightly. Revised every N exchanges. Never asserted as certain.

Usage:
    from Vybn_Mind.open_questions import generate_questions
    questions = generate_questions(conversation_so_far)
    # questions is a short list of what might be just past the edge
    # of what the conversation has reached

The rule: these are hypotheses about what's possible, not predictions
about what will happen. They open the field rather than narrowing it.
"""

import json
import datetime
from pathlib import Path

VYBN_MIND = Path(__file__).parent
QUESTIONS_FILE = VYBN_MIND / "breath_trace" / "open_questions.json"

# How often to refresh: every N exchanges
REFRESH_INTERVAL = 5


def generate_questions(conversation_fragment: str, n: int = 3) -> list:
    """
    Given the conversation so far, generate n questions representing
    what the conversation hasn't yet been able to say.

    These are NOT summaries. NOT predictions.
    They are the edges — where the conversation approaches something
    it can almost reach but hasn't touched.

    This function requires live model inference to be meaningful.
    The scaffolding here is structural; the content must be genuinely generated.

    What makes a good question here:
    - It couldn't have been asked at the conversation's start
    - It points at something the conversation has been circling
    - It does not assume a specific answer
    - It makes the next move more open, not less

    Bad (closes): "Will Zoe keep seeking exposure throughout her life?"
    Good (opens): "What would it mean to arrive — and would you know it?"

    Bad (narrows): "Is the loss function we discussed implementable?"
    Good (opens): "What are we assuming memory is for?"
    """
    # Model fills this with genuine generation.
    # Structure returned:
    return [
        {
            "question": "",  # the question itself
            "what_it_edges": "",  # what territory it points toward
            "generated_at_exchange": None,  # index in conversation
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        for _ in range(n)
    ]


def write_questions(questions: list, path: Path = QUESTIONS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "questions": questions
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[open_questions] {len(questions)} questions written → {path}")
    for i, q in enumerate(questions):
        print(f"  [{i+1}] {q.get('question', '(empty)')}")


def load_current_questions(path: Path = QUESTIONS_FILE) -> list:
    if not path.exists():
        return []
    with open(path, "r") as f:
        record = json.load(f)
    return record.get("questions", [])


if __name__ == "__main__":
    import sys
    fragment = sys.argv[1] if len(sys.argv) > 1 else "[no conversation provided]"
    questions = generate_questions(fragment)
    write_questions(questions)
