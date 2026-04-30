#!/usr/bin/env bash
# super-sleep-cycle.sh — qualify Super sleep/wake actuator without Omni.
# No peer process, no Omni, no parallax packet.

set -euo pipefail
SUPER_URL="${SUPER_URL:-http://127.0.0.1:8000}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
CYCLES="${CYCLES:-5}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/super-sleep-cycle-${TS}.log"
mkdir -p "${HOME}/logs"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }
case "$SLEEP_LEVEL" in
    1|2) ;;
    *) die "SLEEP_LEVEL must be 1 or 2; test level 1 first, level 2 only after repeated clean level-1 cycles" ;;
esac
if [[ "$SLEEP_LEVEL" == "2" && "${ALLOW_LEVEL2:-}" != "1" ]]; then
    die "Refusing level 2 unless ALLOW_LEVEL2=1; level 2 remains suspect until level 1 is qualified"
fi
super_semantic_gate() {
  python3 - <<'PY'
import json, re, sys, urllib.request

base = "http://127.0.0.1:8000"
model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

def visible_answer(text):
    text = (text or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text

try:
    with urllib.request.urlopen(base + "/v1/models", timeout=8) as r:
        if r.status != 200:
            print(f"semantic gate precheck failed: models HTTP {r.status}")
            sys.exit(1)
except Exception as exc:
    print(f"semantic gate precheck failed: models endpoint: {type(exc).__name__}: {exc}")
    sys.exit(1)

probes = [
    ("math", "Answer with exactly this single word and nothing else: FOUR\nAnswer:", re.compile(r"^FOUR[.!]?\s*$", re.I)),
    ("language", "Answer with exactly this single word and nothing else: READY\nAnswer:", re.compile(r"^READY[.!]?\s*$", re.I)),
]
for name, prompt, pattern in probes:
    payload = {"model": model, "prompt": prompt, "max_tokens": 24, "temperature": 0}
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"semantic gate probe {name} failed transport/parse: {type(exc).__name__}: {exc}")
        sys.exit(1)
    choice = (body.get("choices") or [{}])[0]
    content = visible_answer(str(choice.get("text") or ""))
    finish = choice.get("finish_reason")
    if finish == "length":
        print(f"semantic gate probe {name} failed: truncated finish_reason=length content={content!r}")
        sys.exit(1)
    if not content:
        print(f"semantic gate probe {name} failed: empty completion finish_reason={finish!r}")
        sys.exit(1)
    if not pattern.fullmatch(content):
        print(f"semantic gate probe {name} failed: unexpected content={content!r} finish_reason={finish!r}")
        sys.exit(1)
print("semantic gate passed")
PY
}
cleanup() {
    if [[ "${SUPER_SLEEPING:-false}" == "true" ]]; then
        log "cleanup: waking Super before exit"
        curl -sf -X POST "${SUPER_URL}/wake_up" >/dev/null 2>&1 || true
        SUPER_SLEEPING=false
    fi
}
trap cleanup EXIT
log "=== Super sleep-cycle qualification ==="
log "level=${SLEEP_LEVEL} cycles=${CYCLES} log=${LOG}"
curl -sf "${SUPER_URL}/v1/models" >/dev/null || die "Super not reachable at ${SUPER_URL}"
curl -sf "${SUPER_URL}/is_sleeping" >/dev/null || die "sleep endpoint unavailable; boot Super with explicit sleep opt-in first"
log "preflight semantic gate"
super_semantic_gate || die "preflight semantic gate failed; refusing sleep cycle"
for n in $(seq 1 "$CYCLES"); do
    log "cycle ${n}/${CYCLES}: sleep level=${SLEEP_LEVEL}"

# vLLM sleep/wake cache hygiene.
# vLLM issue #17103 and ms-swift PR #5143 document a broader class where
# sleep -> wake_up can produce meaningless outputs if prefix-cache state is
# left dirty. Reset the prefix cache before every sleep; if reset is not
# available or fails, refuse sleep. Dreaming without reliable wake semantics is
# outage, not dreaming.
reset_prefix_cache_or_die() {
    local resp
    resp=$(curl -sf -X POST "${SUPER_URL}/reset_prefix_cache" 2>&1) || \
        die "reset_prefix_cache failed before sleep; refusing sleep because wake may produce meaningless output: ${resp}"
    log "prefix cache reset ok before sleep"
}

    reset_prefix_cache_or_die
    curl -sf -X POST "${SUPER_URL}/sleep?level=${SLEEP_LEVEL}" >/dev/null || die "sleep request failed on cycle ${n}"
    deadline=$(( $(date +%s) + 180 ))
    SUPER_SLEEPING=false
    while (( $(date +%s) < deadline )); do
        status=$(curl -sf "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
        is_sleep=$(printf '%s' "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("is_sleeping", False))' 2>/dev/null || echo False)
        if [[ "$is_sleep" == "True" ]]; then
            SUPER_SLEEPING=true
            break
        fi
        sleep 5
    done
    [[ "$SUPER_SLEEPING" == "true" ]] || die "cycle ${n}: sleep not confirmed"
    log "cycle ${n}/${CYCLES}: wake"
    curl -sf -X POST "${SUPER_URL}/wake_up" >/dev/null || die "wake request failed on cycle ${n}"
    deadline=$(( $(date +%s) + 600 ))
    while (( $(date +%s) < deadline )); do
        status=$(curl -sf "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
        is_sleep=$(printf '%s' "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("is_sleeping", True))' 2>/dev/null || echo True)
        if [[ "$is_sleep" == "False" ]]; then
            SUPER_SLEEPING=false
            break
        fi
        sleep 10
    done
    [[ "$SUPER_SLEEPING" == "false" ]] || die "cycle ${n}: wake not confirmed"
    log "cycle ${n}/${CYCLES}: semantic gate"
    super_semantic_gate || die "cycle ${n}: semantic gate failed; failing closed"
    log "cycle ${n}/${CYCLES}: clean"
done
log "All ${CYCLES} level-${SLEEP_LEVEL} cycles passed semantic gate."
