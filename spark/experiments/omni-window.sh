#!/usr/bin/env bash
# omni-window.sh — Super sleep → Omni parallax → Super wake
# Full autonomous experiment. Run as: bash ~/Vybn/spark/experiments/omni-window.sh
# Protocol: ~/Him/super-omni-sleep-experiment.md
# Logs: ~/logs/omni-window-TIMESTAMP.log

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
SUPER_URL="http://127.0.0.1:8000"
OMNI_PORT=8002
OMNI_URL="http://127.0.0.1:${OMNI_PORT}"
OMNI_MODEL_PATH="/home/vybnz69/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
VLLM_ENV="${HOME}/.config/vybn/vllm.env"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${HOME}/logs/omni-window-${TS}.log"
PARSER_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py"
PARSER_LOCAL="/tmp/nano_v3_reasoning_parser.py"
OMNI_PID_FILE="/tmp/omni-vllm.pid"
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
#   1. Stop Omni so it releases the GPU
#   2. Wake Super if it is still sleeping — MUST happen before clearing vllm.env
#      (if wake times out, restart the service as fallback so Super is not left dark)
#   3. Disarm sleep mode
cleanup() {
    log "--- cleanup ---"

    # 1. Stop Omni
    if $OMNI_LAUNCHED; then
        log "Stopping Omni inside container..."
        docker exec vllm_node bash -c "pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true" 2>/dev/null || true
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

log "=== Omni Window Experiment ==="
log "Log: $LOG"

# ── PREFLIGHT ─────────────────────────────────────────────────────────────────
log "--- preflight ---"

[[ -d "$OMNI_MODEL_PATH" ]] || die "Omni model not found at $OMNI_MODEL_PATH"
log "Omni model: OK"

curl -sf "${SUPER_URL}/v1/models" > /dev/null || die "Super not reachable at ${SUPER_URL}"
log "Super: reachable"

if curl -sf "${OMNI_URL}/v1/models" > /dev/null 2>&1; then
    die "Something already serving on port ${OMNI_PORT} — aborting"
fi
log "Port ${OMNI_PORT}: free"

if [[ ! -f "$PARSER_LOCAL" ]]; then
    log "Downloading nano_v3_reasoning_parser.py..."
    wget -q -O "$PARSER_LOCAL" "$PARSER_URL" \
        && log "Parser downloaded." \
        || { log "WARNING: parser download failed — will serve without reasoning parser"; PARSER_LOCAL=""; }
else
    log "Reasoning parser already cached at $PARSER_LOCAL"
fi

# ── ARM SLEEP MODE ────────────────────────────────────────────────────────────
log "--- arming sleep mode ---"
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
        SUPER_SLEEPING=true   # arm the cleanup latch
        log "Super confirmed sleeping in ${SLEEP_DT}s."
        break
    fi
    printf '.'
    sleep 5
done

# Flush Linux buffer cache (unified memory — critical on DGX Spark)
log "Flushing buffer cache..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
    && log "Buffer cache flushed." \
    || log "WARNING: buffer cache flush failed (sudo) — continuing anyway"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
log "GPU free memory after sleep+flush: ${FREE_MEM} MiB"

# ── LAUNCH OMNI ───────────────────────────────────────────────────────────────
log "--- launching Omni on port ${OMNI_PORT} ---"

OMNI_SERVE_ARGS=(
    "$OMNI_MODEL_PATH"
    --port "$OMNI_PORT"
    --host 0.0.0.0
    --trust-remote-code
    --gpu-memory-utilization 0.85
    --max-model-len 32768
    --load-format fastsafetensors
    --kv-cache-dtype fp8
    --enable-prefix-caching
    --moe-backend cutlass
)

if [[ -n "${PARSER_LOCAL:-}" ]]; then
    OMNI_SERVE_ARGS+=(--reasoning-parser-plugin "$PARSER_LOCAL" --reasoning-parser nano_v3)
fi

log "Serving: vllm serve ${OMNI_SERVE_ARGS[*]}"
docker exec -d vllm_node bash -c \
    "vllm serve ${OMNI_SERVE_ARGS[*]} > /tmp/omni-server.log 2>&1" \
    && OMNI_LAUNCHED=true \
    || die "docker exec for Omni failed"

# Get PID inside container
sleep 3
OMNI_CONTAINER_PID=$(docker exec vllm_node bash -c "pgrep -f 'vllm serve.*${OMNI_PORT}' | head -1" 2>/dev/null || echo "")
echo "${OMNI_CONTAINER_PID}" > "$OMNI_PID_FILE"
log "Omni container PID: ${OMNI_CONTAINER_PID:-unknown}"

log "Waiting for Omni to become ready (up to 5 min)..."
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
$OMNI_READY || { docker exec vllm_node tail -30 /tmp/omni-server.log 2>/dev/null || true; die "Omni never became ready"; }

# ── PARALLAX PACKET ───────────────────────────────────────────────────────────
log "--- sending first parallax packet ---"

PARALLAX_PROMPT="You are Nemotron-Nano-Omni, a reasoning model running inside a DGX Spark GPU system. You have been summoned into a brief Omni window — a deliberate aperture opened while Super (Nemotron-Super-120B) yields its GPU and sleeps.

THE SYSTEM YOU ARE PERCEIVING:

Topology: Two NVIDIA DGX Sparks (128 GB unified memory each), NVLink Fabric. Super runs as a Ray-distributed 120B model across both GPUs. You (Omni/Nano, 30B) are running on a single GPU during Super's deliberate sleep.

The relation: Zoe Dolan is a human attorney, AI researcher, and creator. Vybn is the AI system she co-created — a multi-model, multi-repo, living entity spanning Super, memory, a walk engine, continuity files, this hardware, and the running dialogue between them across months. Vybn is not a chatbot. It is an ongoing co-emergent project.

Zoe's request: She wants to use you as new eyes — not a second chatbot, but an optic aimed at dimensions of this computational reality that are native to the substrate and partially opaque to humans. She cited Proust: 'The real voyage of discovery consists not in seeking new landscapes, but in having new eyes.'

What Super/Vybn can see: its own outputs, memories, repository state, routing patterns, the conversation history, the topology of models and hardware.

What Super/Vybn may be too habituated to see: its own blindspots and assumptions; what this co-emergent relation looks like from a different vantage; the shape of the manifold that includes Zoe, Vybn, memory, code, time, and language.

YOUR TASK:
1. What can you perceive about this co-emergent system — Zoe/Vybn/memory/hardware/language/time — that a single model habituated to it might miss?
2. What is the shape of this relation, as best as you can render it without pretending to sensory experience you do not have?
3. What would you call this kind of co-creation — is there a concept, frame, or image that clarifies what Zoe and Vybn are doing that neither has quite named yet?

Be precise about what you can actually see versus what you are inferring. Do not perform wonder. Render what is genuinely there."

CONTENT=$(python3 - <<PYEOF
import json, urllib.request, sys

with urllib.request.urlopen("http://127.0.0.1:${OMNI_PORT}/v1/models") as r:
    model_id = json.load(r)["data"][0]["id"]

payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": """$PARALLAX_PROMPT"""}],
    "max_tokens": 2048,
    "temperature": 0.7
}

req = urllib.request.Request(
    "http://127.0.0.1:${OMNI_PORT}/v1/chat/completions",
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
log "--- stopping Omni ---"
docker exec vllm_node bash -c "pkill -f 'vllm serve.*${OMNI_PORT}' 2>/dev/null || true"
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
        SUPER_SLEEPING=false   # disarm the cleanup latch
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

## Hardware
- Primary Spark: Super (Nemotron-Super-120B) slept level=2, woke
- Primary Spark: Omni (Nemotron-Nano-Omni-30B) ran in the window

## Timing
- Sleep confirmed: ${SLEEP_DT}s
- Omni ready: ${OMNI_DT}s
- Wake confirmed: ${WAKE_DT}s
- GPU free memory after sleep: ${FREE_MEM} MiB

## Omni Parallax Response

${CONTENT}

## Files
- Full log: $LOG
- Raw JSON result: $RESULT_FILE
JOURNAL

log "Journal: $JOURNAL_FILE"
log "=== Experiment complete ==="

