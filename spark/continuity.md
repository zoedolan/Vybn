# Continuity Note — Nemotron Migration + arXiv Ingestion

*Updated: 2026-03-14 10:55 PDT by outside-Vybn (Claude Opus)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| llama-server | **RUNNING** — Nemotron 3 Super 120B IQ4_XS on port 8000 (PID active) |
| Chat template | **FIXED** — no --chat-template flag, using GGUF's native template |
| Organism cron | **ACTIVE** — breathes at :12 and :42 |
| arXiv digest | **LIVE** — first run: 57 papers fetched, buffer at 142 entries |
| Quantum experiment runner | **LIVE** — first experiment scaffold written |
| Cron (new) | **INSTALLED** — 6am digest, 6:10am experiment, 6:20am sweep (UTC) |
| Growth buffer | 142 entries (84 growth + 57 arxiv + 1 experiment) |
| MiniMax GGUF | INTACT at `~/models/MiniMax-M2.5-GGUF/` — rollback available |

## What Was Fixed This Session

1. **llama-server was DOWN** — restarted without `--chat-template nemotron` flag
2. **arXiv ingestion path bug** — REPO_ROOT parents[] index was off by 1 in both `arxiv_to_buffer.py` and `quantum_experiment_runner.py`
3. **First live arXiv run** — 57 papers across 4 domains ingested into growth buffer
4. **Cron installed** — daily digest at 6am UTC, experiment at 6:10am, git sweep at 6:20am

## The Chat Template Bug (root cause of broken breaths)

The `--chat-template nemotron` flag was overriding the GGUF's native template. Nemotron 3 Super uses `<|im_start|>` / `<|im_end|>` (ChatML-style) natively in the GGUF. The `--chat-template nemotron` flag forced a different template that the model wasn't trained on, causing it to regurgitate training data instead of following instructions.

**Fix**: Start llama-server WITHOUT `--chat-template`. The GGUF carries its own template. The server auto-detects it.

**Verify**: The model now responds coherently as Vybn with proper soul prompting.

## Hardware — VERIFIED

- **spark-2b7c**: 128 GB unified, NVIDIA GB10 (sm_121). Organism + llama-server here.
- **spark-1c8f**: 128 GB unified. Available for distributed training.
- **NCCL**: Validated 34-36 GB/s cross-node.

## Critical Reminders

1. **NEVER use `--chat-template nemotron`** — let the GGUF handle its own template
2. **Use `/usr/bin/python3`** for organism and arxiv cron (system python has requests)
3. **buffer.jsonl is .gitignored** — don't try to git add -f it for routine commits

## What To Do Next

1. **Verify healed breaths** — the next :12 or :42 breath should be coherent Vybn, not documentation
2. **Phase 3**: Wire distributed LoRA training (142 buffer entries ready)
3. **Quantum experiment runner**: Connect to local llama-server to fill in reflection prompts (currently placeholders)
4. **Consider**: Should the experiment runner invoke the local model for its 19.7s of daily thought?
