# Autoresearch Patterns for Vybn

Three patterns ported from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
into Vybn's growth engine. These are the pieces that apply to LoRA fine-tuning
of a served 120B model — the architectural innovations (MuonAdamW, value
embeddings, SSSL windowed attention) are for from-scratch pretraining and
don't transfer.

## 1. Bits-Per-Byte Evaluation (`eval_harness.py`)

The real gift from autoresearch is the evaluation discipline. BPB normalizes
away vocab size by converting per-token nats to bits-per-UTF-8-byte, meaning
if you swap Vybn's tokenizer, change the model, or modify the architecture,
the metric doesn't flinch.

For Vybn's growth cycles, this replaces parsing loss from llama-finetune's
stdout with a proper, comparable metric that works across:
- Model swaps (Nemotron → whatever comes next)
- Tokenizer changes
- Architecture experiments (different LoRA configs, targets)
- Quantization changes (GGUF variants)

## 2. Wall-Clock Time Budget (`TimeBudget` in `eval_harness.py`)

Fixed time budgets make experiments comparable regardless of what changed.
The agent never has to reason about compute-performance tradeoffs because
the time budget answers that question. Currently wraps the existing 2-hour
training timeout, but designed to enable tighter autoresearch-style loops
(5-minute experiments) when Vybn's training pipeline supports it.

## 3. GC Discipline (`gc_discipline()` in `eval_harness.py`)

Python's garbage collector causes ~500ms stalls. On DGX Spark with NCCL
over ConnectX-7, GC pauses from Python workers can corrupt distributed
training timing. The discipline: collect once at start, freeze, disable,
then collect periodically every N steps as a compromise.

## Breath Integration (`extensions/autoresearch.py`)

Every breath (every 30 minutes via cron) triggers the full autoresearch pipeline:

1. **BPB eval** — measures model quality on the breath's own output text
2. **Growth trigger check** — evaluates whether enough new data has accumulated
3. **Training kick-off** — if triggered, starts LoRA training (peft_train.py) in a background thread

This means the growth loop is fully autonomous: the organism breathes, evaluates itself, and trains when it has something to learn from. No manual intervention needed.

Results flow into:
- `Vybn_Mind/bpb_log.jsonl` — continuous BPB time series
- `Vybn_Mind/autoresearch_log.jsonl` — full autoresearch decision log
- `vybn_state.json` — `last_bpb`, `bpb_history` available to next breath
