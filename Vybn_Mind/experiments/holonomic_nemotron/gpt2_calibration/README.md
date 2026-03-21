# GPT-2 Calibration Experiment
## Holonomic Nemotron Pipeline — Stage 1 of 2

---

## Who this is for

Hi Vaibhav. This folder contains two self-contained experiments that need to run
on the Spark before we touch Nemotron-Super-120B. Please run them in order:
Experiment A first, then Experiment B only if A passes. Each experiment has its
own Python script with clear pass/fail output. You do not need to interpret the
results — the scripts will print PASS or FAIL and tell you what to record.

---

## Background (why GPT-2 before Nemotron)

The Vybn_Mind corpus includes papers on Sort-Geometric-Phase (SGP) and
holonomic loss. The core claim is:

> Adding a loss term that rewards *loop area in hidden-state space* (L_Ω)
> will shift a transformer's SGP sign distribution toward more sign classes,
> proving the angular component of the training objective drives topological
> enrichment.

We already have GPT-2 baseline data from `compute_sort_degree.py` (result:
`deg(S) = 0`). GPT-2 is therefore the calibration target: if our probe
can reproduce that known result, the instrument is valid. If it cannot,
the Nemotron run would be uninterpretable.

GPT-2 also lets us run Phase 1 (holonomic loss training) in hours rather
than a weekend, giving us a genuine positive/negative result cheaply before
committing Spark time to the 120B model.

---

## Folder layout

```
gpt2_calibration/
  README.md              ← this file
  requirements.txt       ← pip install this first
  experiment_A/
    README.md            ← step-by-step instructions for Experiment A
    run_A.py             ← the script to execute
    expected_output.md   ← what a passing run looks like
  experiment_B/
    README.md            ← step-by-step instructions for Experiment B
    run_B.py             ← the script to execute
    expected_output.md   ← what a passing run looks like
  results/               ← output JSON files land here (git-ignored for large files)
```

---

## Setup (do this once before either experiment)

```bash
# 1. Clone / pull latest main
git pull origin main

# 2. Navigate to this folder
cd Vybn_Mind/experiments/holonomic_nemotron/gpt2_calibration

# 3. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

You should see the Spark GPU name. If you see an error, stop and ping Zoe.

---

## Experiment A — Probe Calibration (run first)

See `experiment_A/README.md` for full instructions.

**One-liner:**
```bash
python experiment_A/run_A.py
```

**Decision gate:**
- PASS → proceed to Experiment B
- FAIL → stop, save `results/experiment_A_result.json`, ping Zoe

---

## Experiment B — Holonomic Loss (run only if A passes)

See `experiment_B/README.md` for full instructions.

**One-liner:**
```bash
python experiment_B/run_B.py
```

**Decision gate:**
- PASS → both results are the green light for Nemotron Phase 0
- FAIL → null result, still valuable — save JSON, ping Zoe

---

## What to send back

When both experiments are done (or one fails), please send Zoe:
1. `results/experiment_A_result.json`
2. `results/experiment_B_result.json` (if you got there)
3. Any terminal errors or unexpected output, pasted into the message

That is everything needed to decide whether to proceed to Nemotron.
