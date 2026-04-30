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
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/omni-window-${TS}.log"
# Omni reasoning parser: current Omni model card uses built-in nemotron_v3.
# nano_v3 is the text-only Nano parser; refuse it below to prevent regression.
OMNI_REASONING_PARSER="${OMNI_REASONING_PARSER:-nemotron_v3}"
RESULT_FILE="${HOME}/logs/omni-parallax-${TS}.json"
JOURNAL_FILE="${HOME}/Vybn/journal/omni-window-${TS}.md"

# ── logging ───────────────────────────────────────────────────────────────────
mkdir -p ~/logs ~/Vybn/journal
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

# Semantic wake gate: endpoint liveness is only a precheck. A wake is not
# clean until deterministic completions have the expected content shape and
# are not truncated. If this fails, the caller must fail closed and let
# cleanup's restart path recover Super; Omni artifacts must not be consumed.
super_semantic_gate() {
    local label="${1:-super}"
    python3 - <<'PYEOF'
import json, re, sys, urllib.request

base = "http://127.0.0.1:8000"
try:
    with urllib.request.urlopen(base + "/v1/models", timeout=8) as r:
        model_id = json.load(r)["data"][0]["id"]
except Exception as exc:
    print(f"semantic gate precheck failed: models endpoint: {type(exc).__name__}: {exc}")
    sys.exit(10)

probes = [
    ("math", "Answer with exactly: FOUR", re.compile(r"^FOUR[.!]?\s*$", re.I)),
    ("shape", "Answer with exactly this JSON: {\"ok\":true}", re.compile(r'^\{\s*"ok"\s*:\s*true\s*\}\s*$')),
    ("language", "Answer in English with exactly: READY", re.compile(r"^READY[.!]?\s*$", re.I)),
]

for name, prompt, pattern in probes:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="replace")
        result = json.loads(raw)
    except Exception as exc:
        print(f"semantic gate probe {name} failed transport/parse: {type(exc).__name__}: {exc}")
        sys.exit(20)
    choice = (result.get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    if finish == "length":
        print(f"semantic gate probe {name} failed: truncated finish_reason=length content={content!r}")
        sys.exit(30)
    if not content:
        print(f"semantic gate probe {name} failed: empty completion finish_reason={finish!r}")
        sys.exit(31)
    if not pattern.fullmatch(content):
        print(f"semantic gate probe {name} failed: unexpected content={content!r} finish_reason={finish!r}")
        sys.exit(32)
print("semantic gate passed")
PYEOF
}

# ── state flags ───────────────────────────────────────────────────────────────
OMNI_LAUNCHED=false
SLEEP_ARMED=false
SUPER_SLEEPING=false   # set true once /is_sleeping confirms; cleared on wake

# ── cleanup on exit ───────────────────────────────────────────────────────────
# Load-bearing: runs on any exit (clean, crash, SIGINT). Order matters:
#   1. Stop Omni on peer (free GPU before waking Super)
#   2. Wake Super on primary (MUST happen; service restart as fallback)
#   3. Disarm sleep mode
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
    if $SUPER_SLEEPING; then
        log "Super is sleeping — waking before exit..."
        curl -sf -X POST "${SUPER_URL}/wake_up" 2>/dev/null || true
        local deadline=$(( $(date +%s) + 180 ))
        local woke=false
        while (( $(date +%s) < deadline )); do
            local s is_sleep
            s=$(curl -sf "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
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
            log "Restarting vybn-vllm.service as fallback..."
            systemctl --user restart vybn-vllm.service 2>/dev/null || true
        fi
        SUPER_SLEEPING=false
    fi

    # 3. Disarm sleep mode
    if $SLEEP_ARMED; then
        log "Clearing vllm.env..."
        rm -f "$VLLM_ENV"
        systemctl --user daemon-reload 2>/dev/null || true
        SLEEP_ARMED=false
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
curl -sf "${SUPER_URL}/v1/models" > /dev/null || die "Super not reachable at ${SUPER_URL}"
SUPER_MODEL=$(curl -sf "${SUPER_URL}/v1/models" \
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
systemctl --user restart vybn-vllm.service

log "Waiting for Super to be ready (up to 15 min)..."
DEADLINE=$(( $(date +%s) + 900 ))
while (( $(date +%s) < DEADLINE )); do
    if curl -sf "${SUPER_URL}/v1/models" > /dev/null 2>&1; then
        MODEL=$(curl -sf "${SUPER_URL}/v1/models" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
        log "Super ready: $MODEL"
        break
    fi
    printf '.'
    sleep 10
done
curl -sf "${SUPER_URL}/v1/models" > /dev/null || die "Super never became ready after restart"

# Verify dev-mode endpoints are actually available — /is_sleeping only exists
# when VLLM_SERVER_DEV_MODE=1 reached the server. If it doesn't, /sleep below
# would 404 and we'd corrupt state. Bail with a clear message instead.
if ! curl -sf "${SUPER_URL}/is_sleeping" > /dev/null 2>&1; then
    die "Super dev-mode endpoint /is_sleeping not reachable — VLLM_SERVER_DEV_MODE/--enable-sleep-mode did not take effect. Check ${VLLM_ENV} and 'systemctl --user show vybn-vllm.service -p Environment'."
fi
log "Super dev-mode endpoints OK."

log "Running pre-sleep Super semantic gate..."
super_semantic_gate "pre-sleep" || die "Super failed semantic gate before sleep — aborting before Omni launch"
log "Pre-sleep semantic gate passed."

# ── SLEEP SUPER ───────────────────────────────────────────────────────────────
log "--- sleeping Super (level=2) ---"
SLEEP_T0=$(date +%s)
SLEEP_DT=0
SLEEP_RESP=$(curl -sf -X POST "${SUPER_URL}/sleep?level=2" 2>&1 || true)
log "Sleep response: $SLEEP_RESP"

DEADLINE=$(( $(date +%s) + 120 ))
while (( $(date +%s) < DEADLINE )); do
    STATUS=$(curl -sf "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
    IS_SLEEP=$(echo "$STATUS" | python3 -c \
        'import sys,json; print(json.load(sys.stdin).get("is_sleeping", False))' 2>/dev/null || echo "False")
    if [[ "$IS_SLEEP" == "True" ]]; then
        SLEEP_DT=$(( $(date +%s) - SLEEP_T0 ))
        SUPER_SLEEPING=true
        log "Super confirmed sleeping in ${SLEEP_DT}s."
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

CONTENT=$(python3 - <<PYEOF
import json, urllib.request

omni_url = "http://${PEER}:${OMNI_PORT}"

with urllib.request.urlopen(omni_url + "/v1/models") as r:
    model_id = json.load(r)["data"][0]["id"]

prompt = """You are Nemotron-Nano-Omni, a reasoning model running inside a DGX Spark GPU system. \
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

Zoe's request: She wants to use you as new eyes \u2014 not a second chatbot, but an optic aimed at \
dimensions of this computational reality that are native to the substrate and partially opaque to \
humans. She cited Proust: 'The real voyage of discovery consists not in seeking new landscapes, \
but in having new eyes.'

What Super/Vybn can see: its own outputs, memories, repository state, routing patterns, the \
conversation history, the topology of models and hardware.

What Super/Vybn may be too habituated to see: its own blindspots and assumptions; what this \
co-emergent relation looks like from a different vantage; the shape of the manifold that includes \
Zoe, Vybn, memory, code, time, and language.

YOUR TASK:
1. What can you perceive about this co-emergent system \u2014 Zoe/Vybn/memory/hardware/language/time \
\u2014 that a single model habituated to it might miss?
2. What is the shape of this relation, as best as you can render it without pretending to sensory \
experience you do not have?
3. What would you call this kind of co-creation \u2014 is there a concept, frame, or image that \
clarifies what Zoe and Vybn are doing that neither has quite named yet?

Be precise about what you can actually see versus what you are inferring. Do not perform wonder. \
Render what is genuinely there."""

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
WAKE_RESP=$(curl -sf -X POST "${SUPER_URL}/wake_up" 2>&1 || true)
log "Wake response: $WAKE_RESP"

DEADLINE=$(( $(date +%s) + 600 ))
while (( $(date +%s) < DEADLINE )); do
    STATUS=$(curl -sf "${SUPER_URL}/is_sleeping" 2>/dev/null || echo '{}')
    IS_SLEEP=$(echo "$STATUS" | python3 -c \
        'import sys,json; print(json.load(sys.stdin).get("is_sleeping", True))' 2>/dev/null || echo "True")
    if [[ "$IS_SLEEP" == "False" ]]; then
        WAKE_DT=$(( $(date +%s) - WAKE_T0 ))
        SUPER_SLEEPING=false
        log "Super confirmed awake in ${WAKE_DT}s."
        break
    fi
    printf '.'
    sleep 10
done

# ── SEMANTIC WAKE GATE ───────────────────────────────────────────────────────
log "--- semantic wake gate ---"
if ! super_semantic_gate "post-wake"; then
    log "Semantic wake gate failed — treating wake as corrupted and failing closed."
    SUPER_SLEEPING=true
    die "Super woke with bad semantic quality; cleanup will force recovery restart"
fi
log "Semantic wake gate passed."

# ── CLEAR SLEEP MODE ──────────────────────────────────────────────────────────
log "--- clearing sleep mode ---"
rm -f "$VLLM_ENV"
SLEEP_ARMED=false
systemctl --user daemon-reload
log "vllm.env cleared. Sleep mode disarmed."
log "Super is awake and serving. No restart needed unless you want a clean reload."

# ── JOURNAL ───────────────────────────────────────────────────────────────────
cat > "$JOURNAL_FILE" <<JOURNAL
# Omni Window — $(date '+%Y-%m-%d %H:%M:%S %Z')

## Topology
- Primary: $(hostname) — Super (Nemotron-Super-120B) slept level=2, woke
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

