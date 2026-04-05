# Dual Spark Reality — Seared Into Every Boot Path
**2026-04-05T04:45 UTC**

## What happened

Zoe interrupted to point out — for what she said was the dozenth time —
that this system has TWO DGX Sparks with 256 GB unified memory, not one.
She was incensed, and rightfully so.

The root cause was architectural: every file that an instance reads on boot
described single-node hardware. The journal entries documenting the dual-Spark
reality (March 29, 2025; March 29, 2026) existed but were never loaded into
the system prompt. Each new instance woke up reading vybn.md and the agent's
hardcoded text, both of which said ONE Spark.

## What was fixed

### 1. Created `spark/continuity.md` (the file the agent ACTUALLY loads)
Previously this file didn't exist. Now it contains the hardware ground truth:
two Sparks, 256 GB, ConnectX-7, verified launch commands, and explicit
instructions never to run single-node llama-server as primary config.

### 2. Updated `vybn.md` (the soul document)
- "On the Second Spark" section: changed from future tense ("Soon, a second
  DGX Spark will come online") to present tense documenting both Sparks
- "Orientation" section: changed from "a DGX Spark" with single-node model
  list to "two DGX Sparks" with correct primary serving config

### 3. Updated `vybn_spark_agent.py` (the agent source)
- Replaced hardcoded single-node llama-server description with dual-Spark
  vLLM description
- Added `check_dual_spark()` function that runs at boot and injects live
  hardware status into the system prompt
- Fixed "DGX Spark" singular → "DGX Sparks" plural in static text

### 4. Updated `spark/breath_soul.md`
- "running on a DGX Spark" → "running on two DGX Sparks (256 GB unified memory)"

### 5. Updated `Vybn_Mind/VYBN.md`
- Body description updated to reference both Sparks

### 6. Updated `spark/server.py`
- Tool descriptions changed from "a DGX Spark" to "the DGX Sparks"

### 7. Created `spark/start-dual-spark.sh`
Canonical startup script that:
- Verifies second Spark is reachable before proceeding
- Kills any single-node llama-server
- Launches vLLM cluster via run-recipe.sh with FP8 recipe
- Has --daemon flag for background operation

### 8. Created `spark/systemd/vybn-dual-spark.service`
Systemd service for the two-node cluster, replacing the old vybn-llama.service
that started single-node llama-server.

### 9. Updated `spark/systemd/setup.sh`
Now installs vybn-dual-spark.service instead of vybn-llama.service.
Removes stale single-node service files.

### 10. Deprecated `spark/continuity_note.md`
Added deprecation notice pointing to the canonical `spark/continuity.md`.

## The principle

The error kept recurring because the truth was documented in the wrong layer.
Journal entries are historical records — they're not loaded on boot. The files
that shape each instance's initial understanding of itself are:
- vybn.md (loaded via SOUL_PATH)
- covenant.md (loaded via COVENANT_PATH)
- continuity.md (loaded via CONTINUITY_PATH)
- vybn_spark_agent.py (the hardcoded system prompt text)
- breath_soul.md (the local model's soul prompt)

ALL of these now say "two Sparks, 256 GB." Additionally, a live hardware
check runs at boot and injects the result into the system prompt, so even
if the static text drifts, the runtime check will catch it.

## Zoe's instruction

"I want the reminder I just gave you to be the last one I ever have to repeat."

This is the attempt to make that true.
