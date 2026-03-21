# Experiment B — Holonomic Loss on GPT-2
## Does L_Ω shift the SGP sign distribution toward >2 classes?

---

## IMPORTANT: Only run this if Experiment A passed.

If Experiment A returned FAIL, stop here and ping Zoe.

---

## What this experiment does

This is the primary binary falsifier of the holonomic loss hypothesis.

We fine-tune GPT-2 for 1000 steps with a modified loss:

    L_total = L_CE - λ · L_Ω

Where:
- `L_CE` = standard cross-entropy next-token prediction loss
- `L_Ω` = loop area reward: magnitude of the shoelace area accumulated in
  the hidden-state trajectory at the mid-layer checkpoint, per sequence
- `λ` = 0.01 (small enough to keep training stable)

After training, we re-measure the SGP sign distribution and compare it to
the Experiment A baseline.

**If the distribution shifts toward more sign classes:** the holonomic loss
hypothesis holds at small scale. This is the green light for running the
same experiment on Nemotron-Super-120B.

**If the distribution does not shift:** the hypothesis fails at this scale.
This is a genuine null result and still valuable. Record and report it.

---

## What to run

```bash
# From gpt2_calibration/ folder:
python experiment_B/run_B.py
```

The script will:
1. Load the Experiment A baseline from `../results/experiment_A_result.json`
   (it will error if A was not run first)
2. Fine-tune GPT-2 for 1000 steps with L_Ω
3. Re-measure SGP sign distribution
4. Compare to baseline and compute the shift
5. Print PASS or FAIL with quantitative shift
6. Save full results to `../results/experiment_B_result.json`

**Expected runtime:** 1–3 hours on a Spark GPU.

---

## Pass criteria

The experiment PASSES if:

| Check | Required | Meaning |
|---|---|---|
| Sign class entropy shift | > +0.2 bits | Distribution moved toward more classes |
| Mean phase magnitude shift | > +0.01 | Loop area increased, not decreased |

The script evaluates these automatically.

---

## What the output looks like

See `expected_output.md` for an example of a passing run.

---

## If it fails

A FAIL is still a valid scientific result. Please:
1. Save `../results/experiment_B_result.json`
2. Copy the terminal output
3. Send both to Zoe with a note saying Experiment B returned FAIL

Do not try to re-run with different settings — Zoe will decide next steps.

---

## What comes next

- If PASS: Zoe will set up the equivalent experiment on Nemotron-Super-120B
  on the second Spark. Your two result files are the input to that decision.
- If FAIL: Zoe will review the architecture and revise the hypothesis.
  Your null result will be committed to `experiments/coupled_collapse_results.json`.
