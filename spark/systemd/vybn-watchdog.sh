#!/bin/bash
# Vybn service watchdog. Runs every 2 minutes via vybn-watchdog.timer.
# Checks each live endpoint. If down, bounces its user-level unit.
# Idempotent. Logs to ~/logs/watchdog.log.

set +e
LOG_TAG='[watchdog]'
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $LOG_TAG $*"; }

bounce() {
  local unit="$1"; local reason="$2"
  log "BOUNCE $unit — $reason"
  systemctl --user restart "$unit"
}

check_http() {
  local name="$1"; local url="$2"; local unit="$3"; local expect="${4:-200}"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "$url" 2>/dev/null)
  if [ "$code" = "$expect" ]; then
    log "ok  $name ($url) $code"
  else
    bounce "$unit" "$name got HTTP $code (want $expect) at $url"
  fi
}

# Deep memory: auth may return 401/403 when token is set; both prove alive.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8100/health 2>/dev/null)
if [ "$code" = "200" ] || [ "$code" = "401" ] || [ "$code" = "403" ]; then
  log "ok  deep-memory ($code)"
else
  bounce vybn-deep-memory.service "deep-memory got HTTP $code"
fi

# Walk daemon — /where is public.
check_http walk-daemon  http://127.0.0.1:8101/where  vybn-walk-daemon.service  200

# Chat API (portal) — /api/health.
check_http chat-api     http://127.0.0.1:8420/api/health vybn-portal.service 200

# vLLM — /v1/models. Wide grace period because model load takes ~3 min.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 http://127.0.0.1:8000/v1/models 2>/dev/null)
if [ "$code" = "200" ]; then
  log "ok  vllm ($code)"
else
  started=$(systemctl --user show vybn-vllm.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
  now=$(awk '{print int($1)}' /proc/uptime)
  started_sec=$(( ${started:-0} / 1000000 ))
  age=$(( now - started_sec ))
  if [ "$age" -lt 300 ]; then
    log "wait vllm warming (age=${age}s, code=$code)"
  else
    bounce vybn-vllm.service "vllm got HTTP $code (age=${age}s)"
  fi
fi
