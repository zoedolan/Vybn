#!/bin/bash
# Rebuild vybn:latest with the corrected Modelfile
#
# Usage:
#   cd ~/Vybn/spark
#   bash rebuild.sh
#
# What this does:
#   1. Stops any running ollama model to free memory
#   2. Rebuilds vybn:latest from the corrected Modelfile
#   3. Verifies the template is NOT the broken {{ .Prompt }} passthrough
#   4. Shows the active template so you can confirm it looks right
#
# After rebuilding, test with:
#   ollama run vybn
#   >>> hey buddy, it's me zoe. you with me?
#
# If the response is coherent and in-character (not a thriller novel),
# the fix worked.

set -e

echo ""
echo "=== Vybn Modelfile Rebuild ==="
echo ""

# Step 0: Check we're in the right place
if [ ! -f "Modelfile" ]; then
    echo "ERROR: No Modelfile found in current directory."
    echo "       Run this from ~/Vybn/spark/"
    exit 1
fi

# Step 1: Unload current model to free memory
echo "[1/4] Unloading current model..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "vybn:latest",
  "prompt": "",
  "keep_alive": 0
}' > /dev/null 2>&1 || true
echo "  done (or was not loaded)"

# Step 2: Rebuild
echo ""
echo "[2/4] Rebuilding vybn:latest from Modelfile..."
echo "  (this may take a minute)"
ollama create vybn:latest -f Modelfile
echo "  done"

# Step 3: Verify template
echo ""
echo "[3/4] Verifying template..."
TEMPLATE_CHECK=$(ollama show vybn:latest --modelfile 2>/dev/null | grep -c '{{ .Prompt }}' || true)
if [ "$TEMPLATE_CHECK" -gt 0 ]; then
    echo ""
    echo "  WARNING: Template still shows {{ .Prompt }}"
    echo "  The embedded GGUF template may not have been picked up."
    echo "  This needs investigation before proceeding."
    echo ""
    echo "  Run: ollama show vybn:latest --modelfile"
    echo "  and check the TEMPLATE section."
    exit 1
fi
echo "  template looks correct (not the broken passthrough)"

# Step 4: Show what we got
echo ""
echo "[4/4] Active Modelfile:"
echo "  -----------------------------------------------"
ollama show vybn:latest --modelfile 2>/dev/null | head -40
echo "  ..."
echo "  -----------------------------------------------"

echo ""
echo "=== Rebuild complete ==="
echo ""
echo "Test with:"
echo "  ollama run vybn"
echo "  >>> hey buddy, it's me zoe. you with me?"
echo ""
echo "Or through the agent:"
echo "  cd ~/Vybn/spark && python3 tui.py"
echo ""
