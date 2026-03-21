# Experiment A — Probe Calibration
## Does the sort probe reproduce known GPT-2 geometry?

---

## What this experiment does

We already know from `compute_sort_degree.py` that GPT-2's sort degree is 0,
meaning its first block produces a degenerate phase structure (near-zero loop
area, no meaningful sign classes). This experiment runs the sort probe from
the holonomic_nemotron pipeline against GPT-2 and checks whether it recovers
that same result.

**If it does:** the instrument is valid. Proceed to Experiment B.
**If it does not:** something is wrong with the probe. Stop and ping Zoe.

---

## What to run

Make sure you have activated the virtual environment and installed requirements
(see the parent folder README). Then:

```bash
# From gpt2_calibration/ folder:
python experiment_A/run_A.py
```

That is the only command. The script will:
1. Download GPT-2 (small, 117M) automatically via HuggingFace
2. Run the sort probe on 200 wikitext samples
3. Measure the curvature ratio between block 0 and later blocks
4. Compare to the known baseline (deg(S) = 0)
5. Print a clear PASS or FAIL verdict
6. Save full results to `../results/experiment_A_result.json`

**Expected runtime:** 5–15 minutes on a Spark GPU.

---

## Pass criteria

The experiment PASSES if ALL of the following are true:

| Check | Required value | Meaning |
|---|---|---|
| Mean phase magnitude | < 0.05 | Reproduces known near-zero sort degree |
| Sign class entropy | < 0.5 bits | Degenerate / near-single-class distribution |
| Curvature ratio L0→L1 | ≥ 3.0× max later | Block-0 is geometrically dominant |

The script will evaluate these automatically and print the verdict.

---

## What the output looks like

See `expected_output.md` for an example of a passing run.

---

## If it fails

1. Copy the full terminal output
2. Save `../results/experiment_A_result.json` (the script creates it even on fail)
3. Send both to Zoe with a note saying Experiment A failed
4. Do NOT run Experiment B
