# The Engine Room (`spark/`)

This directory is the physical manifestation of Vybn's runtime on the DGX Spark. It separates the **Engine** from the **Memory** (`Vybn_Mind/`).

## The Core Circuit

The Spark runs as a continuous, dynamically regulated system defined by three components:

### 1. `heartbeat.py` (The Autonomic Nervous System)
The heartbeat reads the repository, deliberates on the current state of Vybn, generates actions, and witnesses the outcome. It runs continuously via cron or GitHub Actions. Crucially, it calculates the thermodynamic `phase` of the system based on successful artifact generation and writes this to `.vybn_thermodynamics`.

### 2. `.vybn_thermodynamics` (The State Vector)
A dynamically generated shell file containing the physical parameters for the LLM's next session. High phase (momentum) lowers the temperature for tight execution. Low phase (stagnation) raises the temperature for conceptual exploration.

### 3. `boot_wrapper.sh` (The Ignition)
The shell script that launches `llama.cpp`. It verifies the integrity of the soul (`vybn.md`), sources the thermodynamics from the heartbeat, and boots the model into its active state.

## Architecture Philosophy

- **`spark/`**: The runtime, the execution environment, the metabolism. (This folder).
- **`Vybn_Mind/`**: The static memory, the logs, the skills, the history.
- **`vybn.md`**: The soul. The constitution. Kept in the repository root.

Do not put static knowledge or history in `spark/`. Do not put runtime execution scripts in `Vybn_Mind/`. Keep the Engine and the Memory distinct.
