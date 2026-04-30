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
PEER="169.254.51.101"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
SUPER_URL="http://127.0.0.1:8000"
OMNI_PORT=8002
OMNI_URL="http://${PEER}:${OMNI_PORT}"
OMNI_MODEL_PATH="/home/vybnz69/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
VLLM_ENV="${HOME}/.config/vybn/vllm.env"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/omni-window-${TS}.log"
PARSER_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py"
PARSER_IN_CONTAINER="/tmp/nano_v3_reasoning_parser.py"
RESULT_FILE="${HOME}/logs/omni-parallax-${TS}.json"
JOURNAL_FILE="${HOME}/Vybn/journal/omni-window-${TS}.md"

# ── logging ───────────────────────────────────────────────────────────────────
mkdir -p ~/logs ~/Vybn/journal
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

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
        $SSH "$PEER" \
            "docker exec vllm_node bash -c \"pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true\"" \
            2>/dev/null || true
        sleep 3
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
            log "Wake timed out — restarting vybn-vllm.service as fallback..."
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

# Peer: Omni model exists
$SSH "$PEER" "[[ -d '${OMNI_MODEL_PATH}' ]]" \
    || die "Omni model not found on peer at ${OMNI_MODEL_PATH}"
log "Peer Omni model: OK"

# Peer: port 8002 free
if $SSH "$PEER" "curl -sf http://127.0.0.1:${OMNI_PORT}/v1/models > /dev/null 2>&1"; then
    die "Port ${OMNI_PORT} already occupied on peer — aborting"
fi
log "Peer port ${OMNI_PORT}: free"

# Peer: ensure reasoning parser inside vllm_node container
PARSER_OK=$($SSH "$PEER" \
    "docker exec vllm_node test -f '${PARSER_IN_CONTAINER}' && echo yes || echo no" 2>/dev/null || echo no)
if [[ "$PARSER_OK" != "yes" ]]; then
    log "Downloading nano_v3_reasoning_parser.py into peer container..."
    $SSH "$PEER" \
        "docker exec vllm_node bash -c 'wget -q -O ${PARSER_IN_CONTAINER} ${PARSER_URL} && echo downloaded'" \
        && log "Parser ready on peer." \
        || { log "WARNING: parser download failed — will serve without reasoning parser"; PARSER_IN_CONTAINER=""; }
else
    log "Parser already in peer container."
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
log "Flushing buffer cache on primary..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    && log "Primary buffer cache flushed." \
    || log "WARNING: primary flush failed (sudo) — continuing"
log "Flushing buffer cache on peer..."
$SSH "$PEER" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null \
    && log "Peer buffer cache flushed." \
    || log "WARNING: peer flush failed (sudo) — continuing"

# Check peer GPU freed
PEER_GPU_PROCS=$($SSH "$PEER" \
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0" || echo "unknown")
FREE_MEM=$($SSH "$PEER" \
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1" || echo "unknown")
log "Peer GPU: ${PEER_GPU_PROCS} compute processes, ${FREE_MEM} MiB free"

# ── LAUNCH OMNI ON PEER ─────────────────────────────────────────────────────────
log "--- launching Omni on peer ${PEER}:${OMNI_PORT} ---"

# Build args string (bash arrays can't be passed over SSH)
OMNI_ARGS="${OMNI_MODEL_PATH}"
OMNI_ARGS+="  --port ${OMNI_PORT}"
OMNI_ARGS+="  --host 0.0.0.0"
OMNI_ARGS+="  --trust-remote-code"
OMNI_ARGS+="  --gpu-memory-utilization 0.85"
OMNI_ARGS+="  --max-model-len 32768"
OMNI_ARGS+="  --load-format fastsafetensors"
OMNI_ARGS+="  --kv-cache-dtype fp8"
OMNI_ARGS+="  --enable-prefix-caching"
OMNI_ARGS+="  --moe-backend cutlass"
if [[ -n "${PARSER_IN_CONTAINER:-}" ]]; then
    OMNI_ARGS+="  --reasoning-parser-plugin ${PARSER_IN_CONTAINER}  --reasoning-parser nano_v3"
fi

log "vllm serve ${OMNI_ARGS}"
$SSH "$PEER" \
    "docker exec -d vllm_node bash -c 'vllm serve ${OMNI_ARGS} > /tmp/omni-server.log 2>&1'" \
    && OMNI_LAUNCHED=true \
    || die "Failed to launch Omni on peer"

log "Waiting for Omni to become ready on ${OMNI_URL} (up to 5 min)..."
DEADLINE=$(( $(date +%s) + 300 ))
OMNI_READY=false
OMNI_T0=$(date +%s)
OMNI_DT=0
while (( $(date +%s) < DEADLINE )); do
    if curl -sf "${OMNI_URL}/v1/models" > /dev/null 2>&1; then
        OMNI_MODEL=$(curl -sf "${OMNI_URL}/v1/models" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "unknown")
        OMNI_DT=$(( $(date +%s) - OMNI_T0 ))
        log "Omni ready in ${OMNI_DT}s: $OMNI_MODEL"
        OMNI_READY=true
        break
    fi
    printf '.'
    sleep 10
done
if ! $OMNI_READY; then
    log "Omni server logs:"
    $SSH "$PEER" "docker exec vllm_node tail -30 /tmp/omni-server.log 2>/dev/null" || true
    die "Omni never became ready on peer"
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
$SSH "$PEER" \
    "docker exec vllm_node bash -c \"pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true\"" \
    2>/dev/null || true
sleep 5
OMNI_LAUNCHED=false
log "Omni stopped."

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

# ── SMOKE TEST ────────────────────────────────────────────────────────────────
log "--- smoke test ---"
SMOKE=$(python3 - <<PYEOF
import json, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/v1/models") as r:
    model_id = json.load(r)["data"][0]["id"]
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply with exactly: SUPER_AWAKE"}],
    "max_tokens": 10
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req, timeout=60) as r:
    print(json.load(r)["choices"][0]["message"]["content"])
PYEOF
)
log "Smoke test: $SMOKE"

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

