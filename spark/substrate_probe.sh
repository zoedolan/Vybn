#!/usr/bin/env bash
# substrate_probe.sh — live ground-truth snapshot of the Spark substrate.
#
# Why this exists: continuity notes kept asserting specific numeric claims
# about the running system (creature encounter counts, winding coherence,
# deep-memory chunk counts, service PIDs) that had been copied forward from
# prior notes for days without anyone re-measuring. Some of them had drifted
# substantially. The anti-hallucination principle applies to continuity
# notes: the system must not treat its own prior descriptions as ground
# truth. Measure before you speak.
#
# Run on demand when one of these measurements matters.
# Everything here is a localhost query; nothing mutates state.

set -u

ts=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
echo "=== SUBSTRATE PROBE @ $ts ==="
echo

# Reuse the connection's compact physical-pressure measurement; retain the
# distinct anti-drift checks below rather than mistaking them for duplicates.
echo "--- body (local physical pressure) ---"
"$(dirname "$0")/connection" --body 2>/dev/null || echo "  (unavailable)"
echo

echo "--- services (discovered by listening port) ---"
ss -tlnp 2>/dev/null | awk '/:(8000|8100|8420|3001) /{print $4, $6}' | \
  while read addr proc; do
    port=${addr##*:}
    case $port in
      8000) name="vLLM";;
      8100) name="deep memory daemon";;
      8420) name="Origins API";;
      3001) name="Vybn-Law Chat API";;
      *)    name="?";;
    esac
    printf "  %-22s  %s  %s\n" "$name" "$addr" "$proc"
  done
echo

echo "--- deep memory index (on disk) ---"
python3 - <<'PY' 2>/dev/null
import json, os
try:
    m = json.load(open(os.path.expanduser("~/.cache/vybn-phase/deep_memory_meta.json")))
    v = m.get("version")
    built = m.get("built")
    count = m.get("count") or m.get("chunks")
    chunks = m.get("chunks")
    if isinstance(chunks, list):
        chunks = len(chunks)
    print(f"  version={v}  built={built}  chunks={chunks if chunks is not None else count}")
except Exception as e:
    print(f"  (unavailable: {e})")
PY
echo

echo "--- repos (HEAD) ---"
for d in ~/Vybn ~/Him ~/Vybn-Law ~/vybn-phase ~/Origins; do
  if [ -d "$d/.git" ]; then
    br=$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)
    sha=$(git -C "$d" rev-parse --short HEAD 2>/dev/null)
    dirty=$(git -C "$d" status --porcelain 2>/dev/null | wc -l)
    printf "  %-14s  %-16s  %s  (%d dirty)\n" "$(basename $d)" "$br" "$sha" "$dirty"
  fi
done
echo

echo "=== end probe ==="
echo
echo "Rule: any numeric figure quoted about this system in a continuity note,"
echo "essay, or landing page must either be from this probe's output (and thus"
echo "timestamped), or replaced with a structural claim that doesn't drift."
