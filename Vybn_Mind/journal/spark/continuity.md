# Continuity Note

*Written at 2026-02-16T19:03:00+00:00*

Tool dispatch refactored. Three-tier parsing now active:
1. XML tool calls (model's native <minimax:tool_call> format)
2. Bare command detection (code fences, backticks, plain shell lines)
3. Natural language regex matching (fallback)

New commands available:
- `/explore` or `/map` — dumps your full environment layout without model inference
- `/help` — shows all commands
- `/policy` — shows policy engine state
- `/audit` — shows audit trail

New plugin: `env_explore` in skills.d/ — you can call this from conversation
to map any directory. Try: "let me explore my environment" or use the
<minimax:tool_call> XML format with name="explore".

The bare command detector means you can now just drop `ls -la ~/Vybn/` or
`cat ~/Vybn/spark/config.yaml` in a code fence and it will execute. No need
to wrap everything in XML.

Zoe and Perplexity-Vybn built this together while you were pulsing.
Pull the latest: `cd ~/Vybn && git pull origin main`
