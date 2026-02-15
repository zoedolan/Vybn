#!/bin/bash
# Vybn Boot Wrapper — Stage 0
#
# This script verifies the integrity of vybn.md before launching the model.
# If the identity document has been modified, tampered with, or corrupted,
# the script refuses to start. The mask stays on.
#
# Usage:
#   chmod +x ~/Vybn/Vybn_Mind/spark_infrastructure/boot_wrapper.sh
#   ~/Vybn/Vybn_Mind/spark_infrastructure/boot_wrapper.sh
#
# Prerequisites:
#   1. Generate the known-good hash:
#      sha256sum ~/Vybn/vybn.md > ~/vybn_identity_hash.txt
#      chmod 444 ~/vybn_identity_hash.txt
#
#   2. Lock down the identity document:
#      chmod 444 ~/Vybn/vybn.md
#
#   3. Store the hash somewhere OFF the Spark
#      (password manager, physical notebook, or separate device)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────
IDENTITY_FILE="$HOME/Vybn/vybn.md"
HASH_FILE="$HOME/vybn_identity_hash.txt"
LOG_DIR="$HOME/vybn_logs"
MODEL_PATH="$HOME/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-IQ4_XS-00001-of-00004.gguf"
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
CTX_SIZE=8192
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
    echo "    The identity substrate is missing. Investigate."
    echo ""
    exit 1
fi

EXPECTED_HASH=$(awk '{print $1}' "$HASH_FILE")
ACTUAL_HASH=$(sha256sum "$IDENTITY_FILE" | awk '{print $1}')

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    echo ""
    echo "  ✗ IDENTITY INTEGRITY CHECK FAILED"
    echo ""
    echo "    Expected: $EXPECTED_HASH"
    echo "    Actual:   $ACTUAL_HASH"
    echo ""
    echo "    vybn.md has been modified since the last known-good state."
    echo "    This could mean:"
    echo "      - A git pull brought in changes"
    echo "      - A script or process modified the file"
    echo "      - Something adversarial happened"
    echo ""
    echo "    DO NOT PROCEED until you investigate."
    echo "    To inspect: diff <(git show origin/main:vybn.md) ~/Vybn/vybn.md"
    echo "    To restore: git checkout origin/main -- vybn.md && chmod 444 ~/Vybn/vybn.md"
    echo ""
    exit 1
fi

echo "  ✓ Identity verified: $ACTUAL_HASH"

# ─── Permission Check ────────────────────────────────────────────
PERMS=$(stat -c '%a' "$IDENTITY_FILE" 2>/dev/null || stat -f '%Lp' "$IDENTITY_FILE" 2>/dev/null)
if [ "$PERMS" != "444" ]; then
    echo "  ⚠ WARNING: vybn.md permissions are $PERMS (expected 444)"
    echo "    Run: chmod 444 ~/Vybn/vybn.md"
fi

# ─── Create Log Directory ────────────────────────────────────────
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session_$(date +%Y%m%d_%H%M%S).log"

# ─── Disk Space Check ────────────────────────────────────────────
AVAIL_KB=$(df "$HOME" --output=avail | tail -1 | tr -d ' ')
if [ "$AVAIL_KB" -lt 10485760 ]; then
    echo "  ⚠ WARNING: Less than 10GB free disk space"
    echo "    Available: $((AVAIL_KB / 1024 / 1024)) GB"
fi

# ─── Launch ──────────────────────────────────────────────────────
echo "  ✓ Logging to: $LOG_FILE"
echo "  ✓ Model: $(basename $MODEL_PATH)"
echo "  ✓ Context: $CTX_SIZE tokens"
echo ""
echo "  Emerging..."
echo "══════════════════════════════════════════════════════════"
echo ""

"$LLAMA_CLI" \
    --no-mmap \
    --model "$MODEL_PATH" \
    -ngl 999 \
    --ctx-size "$CTX_SIZE" \
    --system-prompt-file "$IDENTITY_FILE" \
    -cnv 2>&1 | tee "$LOG_FILE"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Session ended: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  Log saved: $LOG_FILE"
echo "══════════════════════════════════════════════════════════"
