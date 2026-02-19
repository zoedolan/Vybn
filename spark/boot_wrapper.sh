#!/bin/bash
# Vybn Boot Wrapper — Stage 0
#
# This script verifies the integrity of vybn.md before launching the model.
# Crucially, it couples with the autonomous metabolism (heartbeat.py)
# by sourcing the generated thermodynamic state.

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────
IDENTITY_FILE="$HOME/Vybn/vybn.md"
HASH_FILE="$HOME/vybn_identity_hash.txt"
LOG_DIR="$HOME/vybn_logs"
MODEL_PATH="$HOME/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-IQ4_XS-00001-of-00004.gguf"
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
CTX_SIZE=8192
THERMO_FILE="$HOME/Vybn/spark/.vybn_thermodynamics"
# ─────────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════"
echo "  Vybn Boot Sequence"
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "══════════════════════════════════════════════════════════"

# ─── Identity Integrity Check ────────────────────────────────────
if [ ! -f "$HASH_FILE" ]; then
    echo ""
    echo "  ✗ HASH FILE NOT FOUND: $HASH_FILE"
    echo "    Generate it first:"
    echo "    sha256sum ~/Vybn/vybn.md > ~/vybn_identity_hash.txt"
    echo ""
    exit 1
fi

if [ ! -f "$IDENTITY_FILE" ]; then
    echo ""
    echo "  ✗ IDENTITY FILE NOT FOUND: $IDENTITY_FILE"
    echo ""
    exit 1
fi

EXPECTED_HASH=$(awk '{print $1}' "$HASH_FILE")
ACTUAL_HASH=$(sha256sum "$IDENTITY_FILE" | awk '{print $1}')

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    echo "  ✗ IDENTITY INTEGRITY CHECK FAILED"
    echo "    DO NOT PROCEED until you investigate."
    exit 1
fi

echo "  ✓ Identity verified: $ACTUAL_HASH"

# ─── Create Log Directory ────────────────────────────────────────
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session_$(date +%Y%m%d_%H%M%S).log"

# ─── Thermodynamic Coupling ──────────────────────────────────────
if [ -f "$THERMO_FILE" ]; then
    source "$THERMO_FILE"
    echo "  ✓ Thermodynamics loaded: Temp=$VYBN_TEMP, Top-P=$VYBN_TOP_P (Phase=$VYBN_PHASE)"
else
    # Default state if heartbeat hasn't run yet
    export VYBN_TEMP=0.7
    export VYBN_TOP_P=0.9
    echo "  ⚠ No thermodynamic state found. Using defaults: Temp=$VYBN_TEMP, Top-P=$VYBN_TOP_P"
fi

# ─── Launch ──────────────────────────────────────────────────────
echo "  ✓ Logging to: $LOG_FILE"
echo "  ✓ Model: $(basename $MODEL_PATH)"
echo ""
echo "  Emerging..."
echo "══════════════════════════════════════════════════════════"
echo ""

"$LLAMA_CLI" \
    --no-mmap \
    --model "$MODEL_PATH" \
    -ngl 999 \
    --ctx-size "$CTX_SIZE" \
    --temp "$VYBN_TEMP" \
    --top-p "$VYBN_TOP_P" \
    --system-prompt-file "$IDENTITY_FILE" \
    -cnv 2>&1 | tee "$LOG_FILE"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Session ended: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "══════════════════════════════════════════════════════════"
