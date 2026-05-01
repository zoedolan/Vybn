#!/usr/bin/env bash
# super-sleep-cycle.sh — qualify Super sleep/wake actuator without Omni.
# No peer process, no Omni, no parallax packet.

set -euo pipefail
SUPER_URL="${SUPER_URL:-http://127.0.0.1:8000}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
CYCLES="${CYCLES:-5}"
VLLM_ENV="${VLLM_ENV:-${HOME}/.config/vybn/vllm.env}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-30}"
SLEEP_ACTUATOR_ARM="${VYBN_SLEEP_ACTUATOR_ARM:-}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/super-sleep-cycle-${TS}.log"
mkdir -p "${HOME}/logs"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }
curl_super() { curl --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" -sf "$@"; }
case "$SLEEP_LEVEL" in
    1|2) ;;
    *) die "SLEEP_LEVEL must be 1 or 2; test level 1 first, level 2 only after repeated clean level-1 cycles" ;;
esac
if [[ "$SLEEP_LEVEL" == "2" && "${ALLOW_LEVEL2:-}" != "1" ]]; then
    die "Refusing level 2 unless ALLOW_LEVEL2=1; level 2 remains suspect until level 1 is qualified"
fi
if [[ "$SLEEP_ACTUATOR_ARM" != "1" ]]; then
    die "Refusing vLLM sleep actuator unless VYBN_SLEEP_ACTUATOR_ARM=1 is set; recent level-1 attempts wedged after /sleep"
fi
super_semantic_gate() {
  PYTHONPATH="${HOME}/Vybn/spark${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m harness.semantic_gate \
      --base-url "${SUPER_URL}" \
      --model "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
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
    SLEEP_REQUESTED=false
}
SLEEP_REQUESTED=false
cleanup() {
    if [[ "${SUPER_SLEEPING:-false}" == "true" || "${SLEEP_REQUESTED:-false}" == "true" ]]; then
        log "cleanup: waking Super before exit"
        curl_super -X POST "${SUPER_URL}/wake_up" >/dev/null 2>&1 || true
        SUPER_SLEEPING=false
        SLEEP_REQUESTED=false
    fi
}
trap cleanup EXIT
log "=== Super sleep-cycle qualification ==="
log "level=${SLEEP_LEVEL} cycles=${CYCLES} curl_max_time=${CURL_MAX_TIME}s log=${LOG}"
curl_super "${SUPER_URL}/v1/models" >/dev/null || die "Super not reachable at ${SUPER_URL}"
curl_super "${SUPER_URL}/is_sleeping" >/dev/null || die "sleep endpoint unavailable; boot Super with explicit sleep opt-in first"
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
    resp=$(curl_super -X POST "${SUPER_URL}/reset_prefix_cache" 2>&1) || \
        die "reset_prefix_cache failed before sleep; refusing sleep because wake may produce meaningless output: ${resp}"
    log "prefix cache reset ok before sleep"
}

    reset_prefix_cache_or_die
    SLEEP_REQUESTED=true
    if ! sleep_resp=$(curl_super -X POST "${SUPER_URL}/sleep?level=${SLEEP_LEVEL}" 2>&1); then
        cold_restart_super_non_sleep "cycle ${n} sleep request failed/timed out: ${sleep_resp}"
        die "cycle ${n}: sleep request failed/timed out; failing closed after cold restart"
    fi
    deadline=$(( $(date +%s) + 180 ))
    SUPER_SLEEPING=false
    while (( $(date +%s) < deadline )); do
        status=$(curl_super "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
        is_sleep=$(printf '%s' "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("is_sleeping", False))' 2>/dev/null || echo False)
        if [[ "$is_sleep" == "True" ]]; then
            SUPER_SLEEPING=true
            break
        fi
        sleep 5
    done
    [[ "$SUPER_SLEEPING" == "true" ]] || die "cycle ${n}: sleep not confirmed"
    log "cycle ${n}/${CYCLES}: wake"
    if ! wake_resp=$(curl_super -X POST "${SUPER_URL}/wake_up" 2>&1); then
        cold_restart_super_non_sleep "cycle ${n} wake request failed/timed out: ${wake_resp}"
        die "cycle ${n}: wake request failed/timed out; failing closed after cold restart"
    fi
    deadline=$(( $(date +%s) + 600 ))
    while (( $(date +%s) < deadline )); do
        status=$(curl_super "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
        is_sleep=$(printf '%s' "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("is_sleeping", True))' 2>/dev/null || echo True)
        if [[ "$is_sleep" == "False" ]]; then
            SUPER_SLEEPING=false
            SLEEP_REQUESTED=false
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
