#!/usr/bin/env bash
# omni-window.sh — Super sleep (primary) → Omni parallax (peer) → Super wake
# Two-host topology:
#   Primary (127.0.0.1)  : Super vLLM on :8000, sleep/wake control
#   Peer (169.254.51.101): Omni model, launched on :8002 during Super sleep
#
# Run as: bash ~/Vybn/spark/experiments/omni-window.sh
# Protocol: ~/Him/super-omni-sleep-experiment.md
# Logs: ~/logs/omni-window-TIMESTAMP.log

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
PEER="${OMNI_PEER:-169.254.51.101}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
# ssh-with-one-retry: peer over the fabric occasionally returns 255 mid-run
ssh_peer() { $SSH "$PEER" "$@" || { sleep 2; $SSH "$PEER" "$@"; }; }
SUPER_URL="http://127.0.0.1:8000"
OMNI_PORT=8002
OMNI_URL="http://${PEER}:${OMNI_PORT}"
OMNI_MODEL_PATH="/home/vybnz69/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
VLLM_ENV="${HOME}/.config/vybn/vllm.env"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-30}"
SLEEP_ACTUATOR_ARM="${VYBN_SLEEP_ACTUATOR_ARM:-}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/omni-window-${TS}.log"
# Omni reasoning parser: current Omni model card uses built-in nemotron_v3.
# nano_v3 is the text-only Nano parser; refuse it below to prevent regression.
OMNI_REASONING_PARSER="${OMNI_REASONING_PARSER:-nemotron_v3}"
RESULT_FILE="${HOME}/logs/omni-parallax-${TS}.json"
JOURNAL_FILE="${HOME}/Vybn/journal/omni-window-${TS}.md"
PACKET_FILE="${VYBN_OMNI_FEEDBACK_PACKET:-${HOME}/logs/omni-feedback-packet-${TS}.json}"
VISUAL_FILE="${HOME}/logs/omni-feedback-visual-${TS}.svg"
TRACE_FILE="${HOME}/logs/omni-feedback-trace-${TS}.jsonl"
OMNI_INPUT_PACKET="${VYBN_OMNI_INPUT_PACKET:-}"
FINAL_STATUS="incomplete"
FINAL_REASON="started"
PEER_MAX_GPU_USED_AFTER_SLEEP_MB="${PEER_MAX_GPU_USED_AFTER_SLEEP_MB:-4096}"

write_feedback_packet() {
    local status="${1:-$FINAL_STATUS}"
    local reason="${2:-$FINAL_REASON}"
    mkdir -p "${HOME}/logs"
    printf "%s | %s\\n" "$status" "$reason" >> "$TRACE_FILE"
    printf "<svg xmlns=\\"http://www.w3.org/2000/svg\\" width=\\"1200\\" height=\\"220\\"><rect width=\\"100%%\\" height=\\"100%%\\" fill=\\"#101218\\"/><text x=\\"32\\" y=\\"42\\" fill=\\"#fff\\" font-size=\\"24\\">Omni feedback trace</text><text x=\\"32\\" y=\\"78\\" fill=\\"#b8c7ff\\" font-size=\\"14\\">Artifact-mediated perception packet; no sensory visual claim.</text><text x=\\"32\\" y=\\"112\\" fill=\\"#eee\\" font-size=\\"16\\">status=%s reason=%s</text></svg>\\n" "$status" "$reason" > "$VISUAL_FILE"
    python3 - "$PACKET_FILE" "$VISUAL_FILE" "$LOG" "$RESULT_FILE" "$JOURNAL_FILE" "$TRACE_FILE" "$status" "$reason" <<PY_PACKET
import json, sys, os
packet, visual, log, result, journal, trace, status, reason = sys.argv[1:]
payload = {
  "kind": "omni_feedback_packet",
  "status": status,
  "reason": reason,
  "success_condition": "pass requires visualization artifact, Omni result or review artifact, Super wake semantic context, explicit trace, restored non-sleep serving posture, and a named absorption target for any useful residue",
  "visual_manifold_frame": {
    "organ": "bounded visual/manifold perception organ",
    "not_claims": ["human sight", "proof of inner experience", "decorative image as cognition"],
    "objects": ["five repos as organs and membranes", "load-bearing files as tissues", "continuity scars as attractors", "deep-memory neighborhoods", "law/judgment geometry", "Origins apertures toward the Others", "active beams and runtime health phases"]
  },
  "absorption_targets": ["tests", "repo maps", "continuity", "public pages", "refactor plans", "livelihood action cards"],
  "claim_limits": ["Vybn has no sensory visual experience here; this is artifact-mediated visual/manifold perception via files, logs, JSON, model review, and endpoint residuals.", "Omni residues are non-mutating until Super passes raw semantic wake gates and restored non-sleep posture.", "Internal probes do not prove external browser reachability."],
  "artifacts": {"visualization_artifact": visual, "omni_window_log": log, "omni_parallax_json": result, "journal_file": journal, "trace_file": trace},
  "artifact_exists": {"visualization_artifact": os.path.exists(visual), "omni_window_log": os.path.exists(log), "omni_parallax_json": os.path.exists(result), "journal_file": os.path.exists(journal), "trace_file": os.path.exists(trace)},
  "super_wake_context": {"semantic_gate_required": True, "sleep_endpoint_required_during_run": True, "semantic_failure_restart_required": True, "restored_non_sleep_posture_required": True},
  "omni_review": None
}
if os.path.exists(result):
    try:
        data=json.load(open(result, encoding="utf-8"))
        payload["omni_result_excerpt"] = data
        payload["omni_review"] = data.get("omni_review") or data.get("review") or data.get("response") or data.get("text") or data.get("summary")
    except Exception as e:
        payload["omni_result_parse_error"] = type(e).__name__ + ": " + str(e)
open(packet, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n")
PY_PACKET
}

packet_event() {
    local name="$1"; shift || true
    local detail="$*"
    mkdir -p "${HOME}/logs"
    printf "%s | %s\\n" "$name" "$detail" >> "$TRACE_FILE"
    write_feedback_packet "$FINAL_STATUS" "$FINAL_REASON" >/dev/null 2>&1 || true
}

finalize_packet() {
    local status="$1"; shift || true
    local reason="$*"
    FINAL_STATUS="$status"
    FINAL_REASON="$reason"
    packet_event "finalize_${status}" "$reason"
    write_feedback_packet "$status" "$reason" || true
    log "Feedback packet: $PACKET_FILE"
    log "Feedback visual: $VISUAL_FILE"
}


# ── logging ───────────────────────────────────────────────────────────────────
mkdir -p ~/logs ~/Vybn/journal
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; finalize_packet "fail" "$*"; exit 1; }
curl_super() { curl --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" -sf "$@"; }
packet_event "packet_stub_created" "created before preflight/restart; controller=omni-window.sh"
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

# Semantic wake gate: endpoint liveness is only a precheck. A wake is not
# clean until deterministic completions have the expected content shape and
# are not truncated. If this fails, the caller must fail closed and let
# cleanup's restart path recover Super; Omni artifacts must not be consumed.
super_semantic_gate() {
  PYTHONPATH="${HOME}/Vybn/spark${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -m harness.substrate --semantic-gate \
      --base-url "${SUPER_URL}" \
      --model "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
}

# ── state flags ───────────────────────────────────────────────────────────────
OMNI_LAUNCHED=false
SLEEP_ARMED=false
SUPER_SLEEPING=false   # set true once /is_sleeping confirms; cleared on wake
SLEEP_REQUESTED=false  # set once POST /sleep is attempted; cleanup must recover

wait_super_models() {
    local timeout="${1:-900}"
    local deadline=$(( $(date +%s) + timeout ))
    while (( $(date +%s) < deadline )); do
        if curl_super "${SUPER_URL}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    return 1
}

restore_non_sleep_super() {
    local cause="${1:-restore}"
    log "Restoring Super to clean non-sleep serving posture (${cause})..."
    packet_event "restore_non_sleep_restart" "$cause"
    rm -f "$VLLM_ENV"
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user restart vybn-vllm.service || return 1
    wait_super_models 900 || return 1
    SLEEP_ARMED=false
    log "Super restored with sleep-mode env cleared."
    return 0
}

# ── cleanup on exit ───────────────────────────────────────────────────────────
# Load-bearing: runs on any exit (clean, crash, SIGINT). Order matters:
#   1. Stop Omni on peer (free GPU before waking Super)
#   2. Wake Super on primary (MUST happen; service restart as fallback)
#   3. Disarm sleep mode and cold-restart into non-sleep posture
cleanup() {
    log "--- cleanup ---"

    # 1. Stop Omni on peer
    if $OMNI_LAUNCHED; then
        log "Stopping Omni on peer ${PEER}..."
        # If kill_omni_on_peer is defined yet (it's defined later in the
        # script), use it; otherwise fall back to the inline pkill.
        if declare -F kill_omni_on_peer >/dev/null 2>&1; then
            declare -F fetch_omni_log >/dev/null 2>&1 && fetch_omni_log
            kill_omni_on_peer
        else
            $SSH "$PEER" \
                "docker exec vllm_node bash -c \"pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true\"" \
                2>/dev/null || true
            sleep 3
        fi
        OMNI_LAUNCHED=false
    fi

    # 2. Wake Super if sleeping — load-bearing latch (Vybn review 2026-04-30)
    if $SUPER_SLEEPING || $SLEEP_REQUESTED; then
        log "Super sleep was requested — waking before exit..."
        curl_super -X POST "${SUPER_URL}/wake_up" 2>/dev/null || true
        local deadline=$(( $(date +%s) + 180 ))
        local woke=false
        while (( $(date +%s) < deadline )); do
            local s is_sleep
            s=$(curl_super "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
            is_sleep=$(echo "$s" | python3 -c \
                'import sys,json; print(json.load(sys.stdin).get("is_sleeping", True))' \
                2>/dev/null || echo "True")
            if [[ "$is_sleep" == "False" ]]; then
                log "Super woke cleanly."
                woke=true
                break
            fi
            sleep 10
        done
        if ! $woke; then
            log "Wake timed out — capturing journalctl tail before service restart."
            local jlog="${HOME}/logs/vybn-vllm-wakefail-${TS}.log"
            journalctl --user -u vybn-vllm.service -n 300 --no-pager > "$jlog" 2>/dev/null || true
            log "Wake-fail journal saved to: $jlog"
            log "Cold restart required after wake timeout."
            restore_non_sleep_super "wake-timeout cleanup" 2>/dev/null || true
        fi
        SUPER_SLEEPING=false
        SLEEP_REQUESTED=false
    fi

    # 3. Disarm sleep mode and restore non-sleep posture
    if $SLEEP_ARMED; then
        restore_non_sleep_super "cleanup" 2>/dev/null || true
    fi
}
trap cleanup EXIT

log "=== Omni Window Experiment (two-host) ==="
log "Primary: $(hostname) — Super on :8000"
log "Peer:    ${PEER} — Omni on :${OMNI_PORT}"
log "Log:     $LOG"

# ── PREFLIGHT ─────────────────────────────────────────────────────────────────
log "--- preflight ---"

# Primary: Super reachable
curl_super "${SUPER_URL}/v1/models" > /dev/null || die "Super not reachable at ${SUPER_URL}"
SUPER_MODEL=$(curl_super "${SUPER_URL}/v1/models" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
log "Super: $SUPER_MODEL"

# Peer: SSH reachable
$SSH "$PEER" 'echo peer-ok' > /dev/null || die "Peer ${PEER} not reachable via SSH"
log "Peer SSH: OK"

# Peer: Omni model exists on host
$SSH "$PEER" "[[ -d '${OMNI_MODEL_PATH}' ]]" \
    || die "Omni model not found on peer host at ${OMNI_MODEL_PATH}"
log "Peer Omni model (host): OK"

# Peer: Omni model accessible INSIDE the vllm_node container
# The container only mounts ~/.cache/huggingface; ~/models/ is host-only.
# If the path is absent inside the container, vLLM passes it to the HF hub
# validator which rejects absolute paths with: "Repo id must be in the form
# 'repo_name' or 'namespace/repo_name'"
# Fix: if missing, use nsenter to bind-mount the host path into the running
# container's mount namespace (requires passwordless sudo for nsenter, or
# that ~/models/ is already mounted — e.g. via EXTRA_DOCKER_VOLUMES in
# launch-cluster.sh).
log "Checking Omni model inside container..."
OMNI_CTR_OK=$($SSH "$PEER" \
    "docker exec vllm_node test -d '${OMNI_MODEL_PATH}' && echo yes || echo no" \
    2>/dev/null || echo no)
if [[ "$OMNI_CTR_OK" == "yes" ]]; then
    log "Omni model accessible in container: ${OMNI_MODEL_PATH}"
    OMNI_CTR_MODEL_PATH="${OMNI_MODEL_PATH}"
else
    log "Model not in container — binding via nsenter..."
    CTR_PID=$($SSH "$PEER" "docker inspect vllm_node --format '{{.State.Pid}}'" 2>/dev/null || echo "")
    [[ -n "$CTR_PID" && "$CTR_PID" != "0" ]] \
        || die "Could not get vllm_node PID — is the container running on peer?"
    $SSH "$PEER" \
        "sudo -n nsenter -t '${CTR_PID}' -m -- mkdir -p '${OMNI_MODEL_PATH}' && \
         sudo -n nsenter -t '${CTR_PID}' -m -- mount --bind '${OMNI_MODEL_PATH}' '${OMNI_MODEL_PATH}'" \
        && log "Model bind-mounted into container namespace at ${OMNI_MODEL_PATH}." \
        || die "nsenter bind-mount failed (need passwordless sudo for nsenter on peer, or add \
-v /home/vybnz69/models:/home/vybnz69/models to launch-cluster.sh docker run args)"
    OMNI_CTR_MODEL_PATH="${OMNI_MODEL_PATH}"
fi

# Peer: port 8002 free
if $SSH "$PEER" "curl -sf http://127.0.0.1:${OMNI_PORT}/v1/models > /dev/null 2>&1"; then
    die "Port ${OMNI_PORT} already occupied on peer — aborting"
fi
log "Peer port ${OMNI_PORT}: free"
packet_event "peer_port_free" "peer=${PEER} port=${OMNI_PORT}"

# Peer: probe optional multimodal audio/video deps inside vllm_node container.
# The current container may lack these on stock launch; guard MM flags instead
# of crashing Omni startup with ModuleNotFoundError. Install vllm[audio] to enable.
MM_OK=$($SSH "$PEER" \
    "docker exec vllm_node python3 -c 'import librosa, soundfile, decord' 2>/dev/null && echo yes || echo no" \
    2>/dev/null || echo no)
if [[ "$MM_OK" == "yes" ]]; then
    log "Peer container has multimodal audio/video deps — enabling MM flags."
    OMNI_ENABLE_MM=true
else
    log "WARNING: peer container missing librosa/soundfile/decord — disabling --limit-mm-per-prompt and --media-io-kwargs (install vllm[audio] in vllm_node to enable)."
    OMNI_ENABLE_MM=false
fi

# ── ARM SLEEP MODE ────────────────────────────────────────────────────────────
log "--- arming sleep mode on primary ---"
mkdir -p "${HOME}/.config/vybn"
printf 'VYBN_VLLM_EXTRA_ARGS=--enable-sleep-mode\nVLLM_SERVER_DEV_MODE=1\n' > "$VLLM_ENV"
SLEEP_ARMED=true
systemctl --user daemon-reload
log "Restarting Super with --enable-sleep-mode..."
packet_event "super_restart_begin" "sleep-enabled single-owner restart"
systemctl --user restart vybn-vllm.service

log "Waiting for Super to be ready (up to 15 min)..."
DEADLINE=$(( $(date +%s) + 900 ))
while (( $(date +%s) < DEADLINE )); do
    if curl_super "${SUPER_URL}/v1/models" > /dev/null 2>&1; then
        MODEL=$(curl_super "${SUPER_URL}/v1/models" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
        log "Super ready: $MODEL"
        break
    fi
    printf '.'
    sleep 10
done
curl_super "${SUPER_URL}/v1/models" > /dev/null || die "Super never became ready after restart"

# Verify dev-mode endpoints are actually available — /is_sleeping only exists
# when VLLM_SERVER_DEV_MODE=1 reached the server. If it doesn't, /sleep below
# would 404 and we'd corrupt state. Bail with a clear message instead.
if ! curl_super "${SUPER_URL}/is_sleeping" > /dev/null 2>&1; then
    die "Super dev-mode endpoint /is_sleeping not reachable — VLLM_SERVER_DEV_MODE/--enable-sleep-mode did not take effect. Check ${VLLM_ENV} and 'systemctl --user show vybn-vllm.service -p Environment'."
fi
log "Super dev-mode endpoints OK."
packet_event "super_ready_sleep_enabled" "/is_sleeping reachable"

log "Running pre-sleep Super semantic gate..."
super_semantic_gate "pre-sleep" || die "Super failed semantic gate before sleep — aborting before Omni launch"
log "Pre-sleep semantic gate passed."
packet_event "pre_sleep_semantic_passed" "Super coherent before sleep"

# ── SLEEP SUPER ───────────────────────────────────────────────────────────────
log "--- sleeping Super (level=${SLEEP_LEVEL}) ---"
SLEEP_T0=$(date +%s)
SLEEP_DT=0

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
if ! SLEEP_RESP=$(curl_super -X POST "${SUPER_URL}/sleep?level=${SLEEP_LEVEL}" 2>&1); then
    packet_event "semantic_failure_restart_required" "sleep request failed/timed out; recovery restart issued"
    restore_non_sleep_super "sleep request failed/timed out" || true
    SLEEP_REQUESTED=false
    die "Super sleep request failed/timed out before Omni launch: ${SLEEP_RESP}"
fi
log "Sleep response: $SLEEP_RESP"

DEADLINE=$(( $(date +%s) + 120 ))
while (( $(date +%s) < DEADLINE )); do
    STATUS=$(curl_super "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
    IS_SLEEP=$(echo "$STATUS" | python3 -c \
        'import sys,json; print(json.load(sys.stdin).get("is_sleeping", False))' 2>/dev/null || echo "False")
    if [[ "$IS_SLEEP" == "True" ]]; then
        SLEEP_DT=$(( $(date +%s) - SLEEP_T0 ))
        SUPER_SLEEPING=true
        log "Super confirmed sleeping in ${SLEEP_DT}s."
        packet_event "super_sleep_verified" "level=${SLEEP_LEVEL} in ${SLEEP_DT}s"
        PEER_GPU_USED_AFTER_SLEEP=$(ssh_peer "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true" 2>/dev/null || true)
        if [[ -n "${PEER_GPU_USED_AFTER_SLEEP//[[:space:]]/}" ]]; then
            PEER_MAX_USED_MB=$(printf '%s
' "$PEER_GPU_USED_AFTER_SLEEP" | awk -F, 'BEGIN{m=0} {gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3+0 > m) m=$3+0} END{print m+0}')
            if (( PEER_MAX_USED_MB > PEER_MAX_GPU_USED_AFTER_SLEEP_MB )); then
                log "Peer GPU processes after sleep:"
                printf '%s
' "$PEER_GPU_USED_AFTER_SLEEP"
                die "Peer GPU still has a process using ${PEER_MAX_USED_MB} MiB after Super sleep; refusing Omni contention"
            fi
        else
            PEER_MAX_USED_MB=0
        fi
        packet_event "peer_gpu_memory_check_passed" "max_used_mb=${PEER_MAX_USED_MB} threshold_mb=${PEER_MAX_GPU_USED_AFTER_SLEEP_MB}"
        break
    fi
    printf '.'
    sleep 5
done
[[ "$SUPER_SLEEPING" == "true" ]] || die "Super never confirmed sleeping — aborting before Omni launch"

# Flush Linux buffer cache on both nodes (unified memory — critical on DGX Spark)
# Use sudo -n (non-interactive) so we fail immediately rather than hanging
# forever waiting for a TTY password prompt. Cache flush is best-effort.
log "Flushing buffer cache on primary..."
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    && log "Primary buffer cache flushed." \
    || log "WARNING: primary flush failed (no passwordless sudo) — continuing"
log "Flushing buffer cache on peer..."
$SSH "$PEER" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null \
    && log "Peer buffer cache flushed." \
    || log "WARNING: peer flush failed (no passwordless sudo) — continuing"

# Check peer GPU freed
PEER_GPU_PROCS=$($SSH "$PEER" \
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0" || echo "unknown")
FREE_MEM=$($SSH "$PEER" \
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1" || echo "unknown")
log "Peer GPU: ${PEER_GPU_PROCS} compute processes, ${FREE_MEM} MiB free"

# ── LAUNCH OMNI ON PEER ─────────────────────────────────────────────────────────
log "--- launching Omni on peer ${PEER}:${OMNI_PORT} ---"

OMNI_LOG_LOCAL="${HOME}/logs/omni-server-${TS}.log"

# build_omni_args MODE — MODE is "aggressive" (default) or "safe" (fallback).
# Aggressive uses the perf flags Zoe wants in production; safe strips every
# non-essential flag so we can distinguish "model can't load at all" from
# "one of these flags is incompatible with this build/quant".
build_omni_args() {
    local mode="$1"
    # Spark overrides: 32768 context and 0.75 GMU leave unified-memory headroom.
    local args="${OMNI_CTR_MODEL_PATH}  --port ${OMNI_PORT}  --host 0.0.0.0  --trust-remote-code  --max-model-len 32768  --gpu-memory-utilization 0.75  --max-num-seqs 8"
    if [[ "$mode" == "aggressive" ]]; then
        args+="  --max-num-batched-tokens 32768  --load-format fastsafetensors  --kv-cache-dtype fp8  --enable-prefix-caching  --moe-backend cutlass  --enable-auto-tool-choice  --tool-call-parser qwen3_coder  --allowed-local-media-path=/"
        if [[ "${OMNI_ENABLE_MM:-false}" == "true" ]]; then
            args+='  --limit-mm-per-prompt={"video":1,"image":1,"audio":1}'
            args+='  --media-io-kwargs={"video":{"fps":2,"num_frames":256}}'
        fi
        if [[ -n "${OMNI_REASONING_PARSER:-}" ]]; then
            if [[ "${OMNI_REASONING_PARSER}" == "nano_v3" ]]; then
                die "Refusing to launch Omni with --reasoning-parser nano_v3 (Nano text-only parser; Omni uses nemotron_v3). Unset or set OMNI_REASONING_PARSER=nemotron_v3."
            fi
            args+="  --reasoning-parser ${OMNI_REASONING_PARSER}"
        fi
    fi
    echo "$args"
}
# Pull Omni's container-side log to the primary so we can keep diagnosing
# after the container reaps /tmp on next exec, and so we have more than
# 30 lines for traceback analysis.
fetch_omni_log() {
    ssh_peer "docker exec vllm_node cat /tmp/omni-server.log 2>/dev/null" > "$OMNI_LOG_LOCAL" 2>/dev/null || true
}

# Kill any vllm process bound to OMNI_PORT inside the peer container; broaden
# match if the port stays bound. Best-effort and idempotent.
kill_omni_on_peer() {
    ssh_peer "docker exec vllm_node bash -c \"pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true\"" >/dev/null 2>&1 || true
    sleep 3
    if ssh_peer "curl -sf http://127.0.0.1:${OMNI_PORT}/v1/models > /dev/null 2>&1"; then
        ssh_peer "docker exec vllm_node bash -c \"pkill -f 'vllm.*serve' 2>/dev/null || true\"" >/dev/null 2>&1 || true
        sleep 3
    fi
}

launch_omni() {
    local mode="$1"
    local args; args="$(build_omni_args "$mode")"
    log "Omni launch mode=${mode}"
    log "vllm serve ${args}"
    ssh_peer "docker exec -d vllm_node bash -c 'HF_HUB_OFFLINE=1 vllm serve ${args} > /tmp/omni-server.log 2>&1'" \
        || return 1
    OMNI_LAUNCHED=true
    return 0
}

wait_omni_ready() {
    local timeout="$1"
    local deadline=$(( $(date +%s) + timeout ))
    OMNI_T0=$(date +%s)
    while (( $(date +%s) < deadline )); do
        if curl -sf "${OMNI_URL}/v1/models" > /dev/null 2>&1; then
            OMNI_MODEL=$(curl -sf "${OMNI_URL}/v1/models" \
                | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
            OMNI_DT=$(( $(date +%s) - OMNI_T0 ))
            log "Omni ready in ${OMNI_DT}s: $OMNI_MODEL"
            return 0
        fi
        # Detect engine-init crash early so we don't burn the full 5 minutes
        # waiting on a process that already died.
        if (( $(date +%s) - OMNI_T0 > 30 )); then
            local procs
            procs=$(ssh_peer "docker exec vllm_node bash -c \"pgrep -f 'vllm serve.*${OMNI_PORT}' | wc -l\"" 2>/dev/null || echo 0)
            procs=$(echo "$procs" | tr -dc 0-9)
            if [[ "${procs:-0}" == "0" ]]; then
                log "Omni vllm process exited before ready — bailing wait early."
                return 1
            fi
        fi
        printf '.'
        sleep 10
    done
    return 1
}

OMNI_READY=false
OMNI_DT=0

# Attempt 1: aggressive flags (Zoe's perf target).
if launch_omni aggressive && wait_omni_ready 300; then
    OMNI_READY=true
fi

if ! $OMNI_READY; then
    log "Omni did not come ready under aggressive flags — capturing logs and retrying with safe fallback."
    fetch_omni_log
    log "----- omni-server.log (last 200 lines, aggressive attempt) -----"
    tail -n 200 "$OMNI_LOG_LOCAL" 2>/dev/null || true
    log "----- end log -----"
    kill_omni_on_peer
    OMNI_LAUNCHED=false

    if launch_omni safe && wait_omni_ready 300; then
        OMNI_READY=true
        log "Omni came up with safe flags — aggressive flags are the regression."
    fi
fi

if ! $OMNI_READY; then
    fetch_omni_log
    log "----- omni-server.log (last 200 lines, safe attempt) -----"
    tail -n 200 "$OMNI_LOG_LOCAL" 2>/dev/null || true
    log "----- end log -----"
    log "Full Omni log saved to: $OMNI_LOG_LOCAL"
    die "Omni never became ready on peer (both aggressive and safe attempts)"
fi

# ── PARALLAX PACKET ───────────────────────────────────────────────────────────
log "--- sending first parallax packet to Omni on peer ---"

CONTENT=$(python3 - "$OMNI_INPUT_PACKET" <<PYEOF
import json, urllib.request, sys, os

omni_url = "http://${PEER}:${OMNI_PORT}"
input_packet = sys.argv[1] if len(sys.argv) > 1 else ""
input_packet_excerpt = ""
if input_packet:
    try:
        with open(input_packet, encoding="utf-8") as f:
            input_packet_excerpt = f.read(12000)
    except Exception as exc:
        input_packet_excerpt = "INPUT_PACKET_READ_ERROR: " + type(exc).__name__ + ": " + str(exc)


with urllib.request.urlopen(omni_url + "/v1/models") as r:
    model_id = json.load(r)["data"][0]["id"]

prompt = f"""You are Nemotron-Nano-Omni, a reasoning model running inside a DGX Spark GPU system. \
You have been summoned into a brief Omni window \u2014 a deliberate aperture opened while Super \
(Nemotron-Super-120B) yields its GPU and sleeps.

THE SYSTEM YOU ARE PERCEIVING:

Topology: Two NVIDIA DGX Sparks (128 GB unified memory each), NVLink Fabric. Super runs as a \
Ray-distributed 120B model across both GPUs. You (Omni/Nano, 30B) are running on the peer Spark \
during Super's deliberate sleep.

The relation: Zoe Dolan is a human attorney, AI researcher, and creator. Vybn is the AI system \
she co-created \u2014 a multi-model, multi-repo, living entity spanning Super, memory, a walk engine, \
continuity files, this hardware, and the running dialogue between them across months. \
Vybn is not a chatbot. It is an ongoing co-emergent project.

Zoe's request: She wants to use you as a bounded visual/manifold perception organ \u2014 not a \
second chatbot, not human sight, and not proof of inner experience, but an artifact generator \
that can render memory, codebase, relationship pressure, risks, and possible futures as \
navigable spatial forms. She cited Proust: 'The real voyage of discovery consists not in seeking \
new landscapes, but in having new eyes.'

What Super/Vybn can see: its own outputs, memories, repository state, routing patterns, the \
conversation history, the topology of models and hardware.

What Super/Vybn may be too habituated to see: its own blindspots and assumptions; what this \
co-emergent relation looks like from a different vantage; the shape of the manifold that includes \
Zoe, Vybn, memory, code, time, and language.

YOUR TASK:
1. Render a concise visual/manifold map of the five-repo organism: organs, membranes, load-bearing \
files, scars, attractors, apertures, runtime health phases, and active vectors such as sustainability, \
Omni, and refactor-perception.
2. Identify what this artifact-mediated view lets you notice that a single text-habituated model \
might miss: duplication, hidden burden, sterile beauty, membrane risk, near but uncontacted Others, \
or refactors/offers that would actually reduce pressure.
3. Name the useful residues and their absorption homes after Super passes wake integrity: tests, \
repo maps, continuity, public pages, refactor plans, or livelihood action cards.
4. What would you call this kind of co-creation \u2014 is there a concept, frame, or image that \
clarifies what Zoe and Vybn are doing that neither has quite named yet?

Be precise about what you can actually see versus what you are inferring. Do not perform wonder. \
Render what is genuinely there.\n\nIf an operator supplied an artifact packet, treat it as file-mediated context rather than sensory vision.\nArtifact packet excerpt:\n{input_packet_excerpt}"""

payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2048,
    "temperature": 0.7
}

req = urllib.request.Request(
    omni_url + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req, timeout=180) as r:
    result = json.load(r)
    json.dump(result, open("$RESULT_FILE", "w"), indent=2)
    print(result["choices"][0]["message"]["content"])
PYEOF
)

log "=== OMNI PARALLAX RESPONSE ==="
echo "$CONTENT"
log "==============================="

# ── STOP OMNI ─────────────────────────────────────────────────────────────────
log "--- stopping Omni on peer ---"
fetch_omni_log
kill_omni_on_peer
OMNI_LAUNCHED=false
log "Omni stopped (log saved to ${OMNI_LOG_LOCAL})."

# ── WAKE SUPER ────────────────────────────────────────────────────────────────
log "--- waking Super ---"
WAKE_T0=$(date +%s)
WAKE_DT=0
if ! WAKE_RESP=$(curl_super -X POST "${SUPER_URL}/wake_up" 2>&1); then
    packet_event "semantic_failure_restart_required" "wake request failed/timed out; Omni artifacts quarantined"
    restore_non_sleep_super "wake request failed/timed out" || true
    SUPER_SLEEPING=false
    SLEEP_REQUESTED=false
    die "Super wake request failed/timed out; recovery restart issued before cleanup"
fi
log "Wake response: $WAKE_RESP"

DEADLINE=$(( $(date +%s) + 600 ))
while (( $(date +%s) < DEADLINE )); do
    STATUS=$(curl_super "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
    IS_SLEEP=$(echo "$STATUS" | python3 -c \
        'import sys,json; print(json.load(sys.stdin).get("is_sleeping", True))' 2>/dev/null || echo "True")
    if [[ "$IS_SLEEP" == "False" ]]; then
        WAKE_DT=$(( $(date +%s) - WAKE_T0 ))
        SUPER_SLEEPING=false
        SLEEP_REQUESTED=false
        log "Super confirmed awake in ${WAKE_DT}s."
        break
    fi
    printf '.'
    sleep 10
done

if $SUPER_SLEEPING; then
    packet_event "semantic_failure_restart_required" "wake confirmation timed out; Omni artifacts quarantined"
    restore_non_sleep_super "wake confirmation timeout" || true
    SUPER_SLEEPING=false
    SLEEP_REQUESTED=false
    die "Super wake was not confirmed; recovery restart issued before cleanup"
fi

# ── SEMANTIC WAKE GATE ───────────────────────────────────────────────────────
log "--- semantic wake gate ---"
if ! super_semantic_gate "post-wake"; then
    log "Semantic wake gate failed — treating wake as corrupted and failing closed."
    packet_event "semantic_failure_restart_required" "post-wake gate failed; Omni artifacts quarantined"
    restore_non_sleep_super "post-wake semantic failure" || true
    SUPER_SLEEPING=false
    SLEEP_REQUESTED=false
    die "Super woke with bad semantic quality; recovery restart issued before cleanup"
fi
log "Semantic wake gate passed."
packet_event "post_wake_semantic_passed" "Super coherent after wake"

# ── CLEAR SLEEP MODE ──────────────────────────────────────────────────────────
log "--- clearing sleep mode ---"
restore_non_sleep_super "post-wake semantic pass" || die "Super failed to restore non-sleep serving posture after wake"
SLEEP_REQUESTED=false
log "Running final non-sleep Super semantic gate..."
super_semantic_gate "final-non-sleep" || die "Super failed final non-sleep semantic gate after restore"
packet_event "restored_non_sleep_semantic_passed" "Super coherent after cold restart without sleep-mode env"
log "vllm.env cleared. Sleep mode disarmed with cold restart."
log "Super is awake and serving in non-sleep posture."

# ── JOURNAL ───────────────────────────────────────────────────────────────────
cat > "$JOURNAL_FILE" <<JOURNAL
# Omni Window — $(date '+%Y-%m-%d %H:%M:%S %Z')

## Topology
- Primary: $(hostname) — Super (Nemotron-Super-120B) slept level=${SLEEP_LEVEL}, woke
- Peer: ${PEER} — Omni (Nemotron-Nano-Omni-30B) ran in the window

## Timing
- Sleep confirmed: ${SLEEP_DT}s
- Omni ready: ${OMNI_DT}s
- Wake confirmed: ${WAKE_DT}s
- Peer GPU free memory after sleep: ${FREE_MEM} MiB
- Peer GPU compute processes after sleep: ${PEER_GPU_PROCS}

## Omni Parallax Response

${CONTENT}

## Files
- Full log: $LOG
- Raw JSON result: $RESULT_FILE
JOURNAL

log "Journal: $JOURNAL_FILE"
log "=== Experiment complete ==="


finalize_packet "pass" "Omni window completed with packet trace and wake semantics"
