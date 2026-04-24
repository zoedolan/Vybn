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

# vLLM — /v1/models. Wide grace period because a cold Nemotron 120B load
# across 2 Sparks is ~10-13 min (26 safetensor shards at ~25s/shard on the
# slower node, plus cudagraph capture). Anything under 900s is warming,
# not failing. Bouncing mid-load would reset the checkpoint read and
# cause the service to crash-loop forever under StartLimitBurst.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 http://127.0.0.1:8000/v1/models 2>/dev/null)
if [ "$code" = "200" ]; then
  log "ok  vllm ($code)"
else
  started=$(systemctl --user show vybn-vllm.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
  active=$(systemctl --user is-active vybn-vllm.service 2>/dev/null)
  if [ -z "$started" ] || [ "$started" = "0" ]; then
    # Unit never activated (or inactive). Only start if there's no container;
    # if containers are up the load was started manually and still warming.
    if docker ps --format '{{.Names}}' | grep -q '^vllm_node$'; then
      log "wait vllm containers present, no systemd record — leaving manual load alone"
    else
      bounce vybn-vllm.service "vllm code=$code, no containers, unit inactive"
    fi
  else
    now=$(awk '{print int($1 * 1000000)}' /proc/uptime)
    age_us=$(( now - started ))
    age=$(( age_us / 1000000 ))
    if [ "$age" -lt 900 ]; then
      log "wait vllm warming (age=${age}s, code=$code, active=$active)"
    else
      bounce vybn-vllm.service "vllm got HTTP $code (age=${age}s, active=$active)"
    fi
  fi
fi
