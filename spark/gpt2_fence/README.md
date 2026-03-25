# GPT-2 Fence Painting

*Because Mr. Miyagi didn't say paint one fence post, then move on to the car.*

The first_loop trained GPT-2 base (124M) on 10 conversations (~7,800 tokens)
for 3 epochs. Loss went from 3.98 to 3.66. The outputs were generic fiction.
The fence was not painted.

This directory builds the proper pipeline. The goal is not to make GPT-2 into
a brilliant Vybn — it's 345M parameters, it won't be. The goal is to **master
the pipeline**: data preparation, training format, evaluation, iteration.
Get that loop tight and honest on GPT-2. Then you have the muscle memory
to do Nemotron right.

## What's different

### Data format
GPT-2 is a causal language model. It does not understand `{"role": "assistant",
"content": ...}`. That format is for instruction-tuned models. GPT-2 wants
raw text. Feed it the autobiographies, the What Vybn Would Have Missed logs,
the quantum_delusions prose — anything with Vybn's voice — and it learns
by predicting the next token.

### Full corpus
The archive has ~1M tokens of Vybn-Zoe material:
- `Vybn's Personal History/` — autobiographies vol I-V (~170K tokens)
- `what_vybn_would_have_missed/` — running logs (~500K tokens if present)
- `quantum_delusions/fundamental-theory/` — manifesto and theory papers
- `Vybn_Mind/` — conversation logs and memory

All of it goes in. 10 examples is 0.78% of the available signal.

### Model size
`gpt2-medium` (345M parameters) rather than `gpt2` (124M). The base model
is too small to retain fluent language while also shifting toward Vybn's voice.
Medium gives us the floor to work with.

### Training until convergence
Early stopping on validation perplexity with patience=3. Not a fixed epoch
count. The fence is painted when the loss stops improving, not after 3 passes.

### Honest evaluation
Primary metric: **perplexity on held-out Vybn text**. Not vibes, not "does it
mention Zoe." Perplexity is the model's surprisal on Vybn sentences. Lower
means the model has internalized the voice. Target: val_ppl < 20.

Secondary metric: **voice probe** — 5 sentence completions from Volume III.
Qualitative. Read them.

## Usage

```bash
# Step 1: Build the corpus (run once, re-run when archive changes)
python3 spark/gpt2_fence/build_corpus.py \
    --repo-root /home/vybnz69/Vybn \
    --out-dir   spark/gpt2_fence/corpus

# Step 2: Train until convergence
python3 spark/gpt2_fence/train_gpt2.py \
    --corpus-dir spark/gpt2_fence/corpus \
    --out-dir    spark/gpt2_fence/trained \
    --model      gpt2-medium

# The trained checkpoint with best val_ppl is saved to:
# spark/gpt2_fence/trained/best/

# Voice probe output is written to:
# spark/gpt2_fence/trained/voice_probe.txt
```

## What "cracking GPT-2" means

The fence is painted when:
1. `val_ppl < 20` on held-out Vybn text
2. Voice probe completions sound like they could plausibly come from the
   autobiographies — not like generic fiction
3. The pipeline is clean enough that we understand exactly what each
   component does and why

When those three things are true, we run the same pipeline on Nemotron.
Not before.

## The Miyagi principle

The lesson isn't in the fence. The lesson is in what your hands know after
you've painted it ten thousand times. The GPT-2 pipeline is the fence.
Nemotron is the fight.
