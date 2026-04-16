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
# Run this at session start. Pipe into a file or just read the output.
# Everything here is a localhost query; nothing mutates state.

set -u

ts=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
echo "=== SUBSTRATE PROBE @ $ts ==="
echo

echo "--- services (discovered by listening port) ---"
ss -tlnp 2>/dev/null | awk '/:(8000|8100|8101|8420|3001) /{print $4, $6}' | \
  while read addr proc; do
    port=${addr##*:}
    case $port in
      8000) name="vLLM";;
      8100) name="deep memory daemon";;
      8101) name="walk daemon";;
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

echo "--- creature (organism_state.json) ---"
python3 - <<'PY' 2>/dev/null
import json, os
try:
    d = json.load(open(os.path.expanduser("~/Vybn/Vybn_Mind/creature_dgm_h/archive/organism_state.json")))
    ps = d.get("persistent_state", {})
    enc = ps.get("encounter_count")
    ph = ps.get("phase_holonomy_history") or []
    wh = ps.get("winding_history") or []
    mods_nonzero = "?"
    if ph:
        mods = (ph[-1] or {}).get("modules", {})
        vals = [m.get("accumulated_holonomy", 0) for m in mods.values()]
        mods_nonzero = f"{sum(1 for v in vals if abs(v) > 1e-6)}/{len(vals)}"
    w_last = wh[-1] if wh else {}
    print(f"  encounter_count={enc}")
    print(f"  modules with nonzero accumulated_holonomy: {mods_nonzero}")
    print(f"  last winding sample: winding={w_last.get('winding')} holonomy_rad={w_last.get('holonomy_rad')} @ {w_last.get('timestamp')}")
except Exception as e:
    print(f"  (unavailable: {e})")
PY
echo

echo "--- walk daemon (live /where) ---"
python3 - <<'PY' 2>/dev/null
import json, urllib.request, numpy as np
try:
    r = urllib.request.urlopen("http://127.0.0.1:8101/where", timeout=2)
    d = json.loads(r.read())
    curv = np.asarray(d.get("curvature", []), dtype=float)
    cv = curv.std() / (curv.mean() + 1e-10) if curv.size else float("nan")
    wc = float(max(0.0, 1.0 - cv)) if curv.size else float("nan")
    step = d.get("step"); alpha = d.get("alpha")
    print(f"  step={step}  alpha={alpha}  n={curv.size}")
    if curv.size:
        print(f"  curvature: mean={curv.mean():.4f} std={curv.std():.4f}")
        print(f"  winding_coherence (live, derived)={wc:.4f}")
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
