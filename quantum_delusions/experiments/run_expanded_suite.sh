#!/usr/bin/env bash
# run_expanded_suite.sh — Generate basin data + run full quantum suite on IBM
#
# Prerequisites:
#   - QISKIT_IBM_TOKEN set in environment
#   - pip install qiskit qiskit-ibm-runtime numpy
#
# Run from repo root:
#   bash quantum_delusions/experiments/run_expanded_suite.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CREATURE_DIR="$REPO_ROOT/Vybn_Mind/creature_dgm_h"
QUANTUM_DIR="$REPO_ROOT/quantum_delusions/experiments"
BASIN_DIR="$CREATURE_DIR/experiment_results/basin_geometry"

echo "=== Step 1: Generate basin result with weight trajectories ==="
cd "$CREATURE_DIR"
python3 experiments.py basin --quick

echo ""
echo "=== Step 2: Find newest basin result ==="
BASIN_FILE=$(ls -t "$BASIN_DIR"/basin_*.json 2>/dev/null | head -1)
if [ -z "$BASIN_FILE" ]; then
    echo "ERROR: No basin result found in $BASIN_DIR"
    exit 1
fi
echo "Using: $BASIN_FILE"

echo ""
echo "=== Step 3: Scan trajectory ==="
cd "$QUANTUM_DIR"
python3 creature_quantum_bridge.py scan

echo ""
echo "=== Step 4: Dry run (inspect circuits) ==="
python3 creature_quantum_bridge.py run "$BASIN_FILE" --shots 4096 --dry-run

echo ""
echo "=== Step 5: Submit to IBM ==="
python3 creature_quantum_bridge.py run "$BASIN_FILE" --shots 4096

echo ""
echo "=== Done. Results in $QUANTUM_DIR/results/ ==="
ls -lt "$QUANTUM_DIR/results/" | head -5
