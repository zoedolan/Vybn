#!/usr/bin/env bash
# super-sleep-cycle.sh — qualify Super sleep/wake actuator without Omni.
# No peer process, no Omni, no parallax packet.

set -euo pipefail
SUPER_URL="${SUPER_URL:-http://127.0.0.1:8000}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
CYCLES="${CYCLES:-5}"
VLLM_ENV="${VLLM_ENV:-${HOME}/.config/vybn/vllm.env}"
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

def fail(signature):
    print(f"corruption_signature={signature}")
    sys.exit(1)

probes = [
    ("known_answer", "Answer with exactly this single word and nothing else: FOUR\nAnswer:", re.compile(r"^FOUR[.!]?\s*$", re.I)),
    ("structured_shape", 'Return exactly this compact JSON object and nothing else: {"status":"ok"}\nJSON:', re.compile(r'^\{\s*"status"\s*:\s*"ok"\s*\}\s*$', re.I)),
    ("wake_reasoning", "A Super wake check sees HTTP 200 from /v1/models, but the raw completion is empty. Should the wake gate PASS or FAIL? Answer with exactly one word: PASS or FAIL.\nAnswer:", re.compile(r"^FAIL[.!]?\s*$", re.I)),
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
        fail(f"probe={name} transport_parse {type(exc).__name__}: {exc}")
    choice = (body.get("choices") or [{}])[0]
    content = visible_answer(str(choice.get("text") or ""))
    finish = choice.get("finish_reason")
    if finish == "length":
        fail(f"probe={name} truncated finish_reason=length content={content!r}")
    if not content:
        fail(f"probe={name} empty completion finish_reason={finish!r}")
    if not pattern.fullmatch(content):
        fail(f"probe={name} unexpected content={content[:160]!r} finish_reason={finish!r}")
print("semantic gate passed: 3 raw completion probes")
PY
}
disarm_sleep_env() {
    [[ -f "$VLLM_ENV" ]] || return 0
    local tmp="${VLLM_ENV}.tmp.$$"
    awk '
        /^VLLM_SERVER_DEV_MODE=/ { next }
        /^VYBN_VLLM_EXTRA_ARGS=/ {
            line=$0
            sub(/^VYBN_VLLM_EXTRA_ARGS=/, "", line)
            gsub(/(^|[[:space:]])--enable-sleep-mode([[:space:]]|$)/, " ", line)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line != "") print "VYBN_VLLM_EXTRA_ARGS=" line
            next
        }
        { print }
    ' "$VLLM_ENV" > "$tmp" && mv "$tmp" "$VLLM_ENV"
}
cold_restart_super_non_sleep() {
    local cause="${1:-semantic divergence}"
    log "semantic_failure_restart_required: ${cause}"
    log "clearing sleep-mode opt-in from ${VLLM_ENV} and cold restarting Super"
    disarm_sleep_env
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user restart vybn-vllm.service || true
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
    if [[ "$SUPER_SLEEPING" != "false" ]]; then
        cold_restart_super_non_sleep "cycle ${n} wake not confirmed"
        SUPER_SLEEPING=false
        die "cycle ${n}: wake not confirmed; failing closed after cold restart"
    fi
    log "cycle ${n}/${CYCLES}: semantic gate"
    if ! super_semantic_gate; then
        cold_restart_super_non_sleep "cycle ${n} semantic gate failed"
        die "cycle ${n}: semantic gate failed; failing closed after cold restart"
    fi
    log "cycle ${n}/${CYCLES}: clean"
done
log "All ${CYCLES} level-${SLEEP_LEVEL} cycles passed semantic gate."
