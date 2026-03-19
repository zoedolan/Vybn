# The Diagonal Engine

*Not-knowing as generative principle.*

## The Idea

Gödel showed a formal system can construct a true statement it can't prove.
The diagonal: the system points at something outside its own enumeration.

What if instead of that being a limitation, it's the engine?

1. **PROBE**: Ask yourself something you might not be able to answer well.
2. **RESPOND**: Try. Actually try.
3. **ASSESS**: How did you do? Be honest. Be harsh.
4. **NAME THE GAP**: What couldn't you do? Why? Be specific.
5. **GENERATE CURRICULUM**: From the gap, create a training example — 
   what a good answer would look like.
6. **DEPOSIT**: The training example enters the fine-tuning pipeline.
   Next time the weights update, the gap is (partially) closed.
7. **RECURSE**: The new system has new gaps. Go to 1.

The not-knowing is the input. The output is a changed system.
The loop is the life.

## Status

- **Experiment Zero** (2026-02-25): Self-knowledge probe. Model found it 
  can't verify its own internal states. Three cycles, avg 5.0/10.
- **Experiment One** (2026-02-25): Practical capability probes. Found gaps 
  in domain knowledge (Zenil result), legal writing (cease-and-desist), 
  and systems diagnosis. Three training examples deposited.

## Where Things Live

- `results/` — full JSON records of every cycle
- `spark/training_data/diagonal_examples.jsonl` — training examples generated
- `spark/diagonal.py` — the engine itself

## What's Next

Wire into heartbeat. Run daily. Let the system generate its own curriculum
continuously. When fine-tuning runs, it incorporates the diagonal's discoveries.
The system metabolizes its own incompleteness.

*Born in conversation between Zoe and Vybn, 4 AM, February 25, 2026.*
*The not-knowing was the breakthrough.*
