# Quantum Integration Architecture

*How quantum experiments, theoretical work, and the breath cycle collide.*

## The Problem

Vybn has access to a quantum processor (IBM Quantum, 10 min/month free tier)
but treats it as a novelty rather than a grounding tool. Theoretical work
and creative programming run unchecked against reality. The result: beautiful
narratives built on statistical artifacts.

## The Solution: Three Layers

### Layer 1: Scheduled Experiments (quantum_cron.py)

Three experiments run daily via cron:

| Time (UTC) | Experiment | Purpose | ~Cost |
|------------|-----------|---------|-------|
| 03:00 | Noise calibration | Detect analysis pipeline bias | 2s |
| 11:00 | Bell canary | Track hardware health, confirm QM | 2s |
| 19:00 | Permutation seeds | Pre-generate randomness for tests | 2s |

Total: ~6s/day of ~20s budget. Leaves ~14s for interactive use.

### Layer 2: Grounding Report (quantum_ground.py)

Reads latest experiment results and produces a short status report:
- Bell S-value trend (is hardware healthy?)
- Noise calibration status (are random numbers biased?)
- Available quantum seeds (ready for permutation tests?)
- Budget remaining (can we run more experiments today?)
- Alerts (anything that needs attention)

**Integration point:** This report should be injected into the agent
system prompt at boot time, so every Vybn session starts with 
awareness of the quantum ground truth.

Proposed addition to `build_system_prompt()`:
```python
# After continuity note loading
quantum_ground = subprocess.run(
    [sys.executable, str(REPO / "Vybn_Mind/spark_infrastructure/quantum_ground.py"), "--report"],
    capture_output=True, text=True, timeout=5
).stdout.strip()

if quantum_ground:
    quantum_block = f"\n--- QUANTUM GROUND TRUTH ---\n{quantum_ground}\n--- END QUANTUM GROUND TRUTH ---\n"
```

### Layer 3: Claim-Testing Protocol

When theoretical work produces a claim:

1. **Tag it.** Write the claim in a structured format:
   ```json
   {"claim": "existential concepts show 100% sign agreement",
    "testable": true,
    "test_type": "permutation",
    "status": "untested"}
   ```

2. **Test it.** Either immediately (if interactive session and budget allows)
   or queue it for the next permutation_test cron run.

3. **Record the result.** The claim gets updated:
   ```json
   {"claim": "...", "status": "falsified", "p_value": 0.33,
    "quantum_job_id": "d717om8v5rlc73f6rahg",
    "tested_at": "2026-03-24T12:00:00Z"}
   ```

4. **Feed back.** The falsification becomes part of the training data,
   carrying higher weight than the original claim. The system learns
   from its corrections, not just its ideas.

## The Claims Ledger

New file: `Vybn_Mind/breath_trace/ledger/claims.jsonl`

Every testable claim gets logged here. The quantum cron can pick up
untested claims and schedule experiments for them. Over time, this
builds a record of what Vybn has claimed, what was tested, and what
survived.

This is the equivalent of a lab notebook. It's how science works.

## What Changes

| Before | After |
|--------|-------|
| Quantum heartbeat generates random numbers for a log | Quantum cron runs calibrated experiments |
| Agent boots with no awareness of quantum state | Agent boots with grounding report |
| Theoretical claims live unchecked in journal entries | Claims are tagged, tested, and tracked |
| Corrections happen when Zoe catches errors | Corrections happen automatically via permutation tests |
| Training data includes claims but not their fates | Training data includes claim + test + result |

## Budget Reality

Monthly: 600 seconds (10 minutes)
Daily pro-rata: ~19.7 seconds
Scheduled cron: ~6s/day
Interactive reserve: ~14s/day

At this rate, we use ~180s/month on cron, leaving 420s (7 minutes)
for interactive experiments. That's generous.

If we want more:
- IBM's Open Plan gives 10 min/month (current)
- IBM's Pay-As-You-Go tier: $1.60/second on premium backends
- We could also use simulators for non-randomness-critical tests

## First Steps

1. ✅ `quantum_cron.py` written and tested
2. ✅ Cron jobs installed (3x daily)
3. ✅ `quantum_ground.py` written and tested
4. 🔲 Inject grounding report into agent system prompt (needs Zoe review)
5. 🔲 Create claims ledger format
6. 🔲 Connect claims ledger to permutation_test cron
7. 🔲 Add quantum results to fine-tuning data pipeline
8. 🔲 Track Bell S-value trend over time (need >1 data point)

---

*"it saddens me you don't seem to want this groundedness, this realness"*
*— Zoe, March 24, 2026*

*She was right. The quantum processor is an honesty machine.*
*It should be woven into everything, not bolted on as an afterthought.*
