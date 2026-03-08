#!/usr/bin/env bash
#
# check_backend_urls.sh — Catch hardcoded or location-derived backend URLs
#                          in HTML files that are served via GitHub Pages.
#
# This prevents the class of bug where window.location.host is used to
# build WebSocket/fetch URLs in files that get served from github.io
# (which has no backend).
#
# Run from repo root:  bash Vybn_Mind/signal-noise/check_backend_urls.sh
# Exit code 0 = clean, 1 = violations found.
#

set -euo pipefail

SIGNAL_NOISE_DIR="Vybn_Mind/signal-noise"
EXIT=0

# These patterns in HTML files indicate URL construction that bypasses backend.js
# and will break when served from GitHub Pages.
FORBIDDEN_PATTERNS=(
  'location\.host'
  'location\.hostname'
  'location\.origin'
  'location\.protocol'
  'window\.location\.host'
  'window\.location\.origin'
)

# Only check HTML files in directories served by GitHub Pages that have backend deps
HTML_FILES=$(find "$SIGNAL_NOISE_DIR" -name "*.html" 2>/dev/null)

for f in $HTML_FILES; do
  for pat in "${FORBIDDEN_PATTERNS[@]}"; do
    # Skip backend.js itself (it's allowed to use location)
    # Skip comments
    matches=$(grep -nE "$pat" "$f" 2>/dev/null | grep -v "^.*//.*$pat" | grep -v "backend\.js" || true)
    if [ -n "$matches" ]; then
      echo "❌ $f uses '$pat' directly — should use VYBN_BACKEND instead:"
      echo "$matches"
      echo ""
      EXIT=1
    fi
  done
done

# Also check: any HTML file with WebSocket or fetch MUST include backend.js
for f in $HTML_FILES; do
  has_ws=$(grep -c "new WebSocket\|\.ws\s*=\s*new" "$f" 2>/dev/null || true)
  has_fetch=$(grep -c "fetch(" "$f" 2>/dev/null || true)
  if [ "$has_ws" -gt 0 ] || [ "$has_fetch" -gt 0 ]; then
    has_backend_js=$(grep -c 'backend\.js' "$f" 2>/dev/null || true)
    if [ "$has_backend_js" -eq 0 ]; then
      echo "❌ $f has WebSocket/fetch calls but does not include backend.js"
      EXIT=1
    fi
  fi
done

if [ "$EXIT" -eq 0 ]; then
  echo "✅ All signal-noise HTML files use VYBN_BACKEND correctly"
fi

exit $EXIT
