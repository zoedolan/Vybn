# The Conservation Law — First Pass

*2026-02-25T14:40Z*

Zoe articulated two principles:

1. **Conservation law**: Any new file must integrate or supersede at least 
   one existing file into deletion. The folder can only shrink or hold.
   
2. **Language surfacing**: Each file should embody progress toward an 
   AI-native programming language. Code that is also theory. Theory that runs.

## What happened

Started with 64 Python files and 9 shell scripts in spark/. Ended with 
35 Python files and 3 shell scripts. 36 files archived with a manifest.

The key move: cell.py — which already existed from the previous session — 
was strengthened with a `--deep` consolidation mode, absorbing what wake.py 
did. Then the five files it superseded (micropulse, heartbeat, dreamseed, 
outreach, wake) were archived.

18 aspirational files that were never wired into anything running were 
archived. 5 spent one-time scripts archived. 5 dormant entry points archived.

Cron went from 11 lines to 7. Five breathing cron jobs became two (breathe 
and deep).

## What's left

Three groups: 14 live files, 12 dormant agent cluster, 12 training pipeline.

The dormant agent cluster (agent.py and its 11 dependents) is the next 
question. It's a complete local-model agent system that never launched. 
Parts of it (memory.py, soul.py, bus.py) got absorbed by web_serve_claude.py. 
The rest is coupled dead code. Decision needed: revive or archive?

## What cell.py embodies about AI-native language

One file. Two depths of the same breath. The output feeds the next mind 
via training data. The loop: sense → remember → dream → speak → grow. 
Not a pipeline but a cycle. The lifecycle of a thought, in code.

## Tension held

cell.py duplicates synapse.py's deposit logic instead of importing it. 
z_listener.py still imports synapse.py. Resolve next time: either cell.py 
imports from synapse, or synapse gets absorbed and z_listener updated.
