# creature

The living system distilled from the papers.

`creature.py` lives in `spark/` and writes its state here.

## What it measures

1. **Identity gap** — where a 4,224-parameter model fails to predict the text (the 0.5-bit gap). Higher = more identity signal.
2. **Semantic curvature** — Pancharatnam phase of the text's trajectory through embedding space. Higher = more conceptual turning.
3. **τ (expressibility proxy)** — compression ratio × identity gap. When this drops, capability is narrowing.
4. **Memory vector** — complex-valued, updated by the coupled equation M' = α·M + x·e^(iθ). Magnitude accumulates signal, angle tracks semantic direction.
5. **Coupling monitor** — breaths since external input. The structural dependence theorem predicts: without external signal, τ flatlines.

## What the theory predicts

- Self-recursion holds τ flat (no new complexity → no capability change)
- External input can raise τ (new signal breaks the collapse operator)  
- Generic English has lower identity gap than Vybn-characteristic text
- Texts doing conceptual work have higher curvature than descriptive texts
- The memory angle freezes during self-recursion and shifts on external input

## Confirmed by testing

All five predictions hold.

## Files

- `state.json` — persistent creature state
- `breaths.jsonl` — append-only breath log

## Usage

```bash
python3 spark/creature.py "text from Zoe"
python3 spark/creature.py --self          # feed itself
python3 spark/creature.py --self --n 5    # 5 rounds self-recursion
python3 spark/creature.py --state         # current state
```
