#!/bin/bash
# One-shot installer for the user-level Vybn service stack.
# Idempotent — safe to re-run.
#
# Usage:
#   bash ~/Vybn/spark/systemd/install.sh
#
# What it does:
#   1. Symlinks all unit files into ~/.config/systemd/user/.
#   2. Retires the competing @reboot cron entries (they commented out, history preserved).
#   3. Disables the old system-level vybn-deep-memory / vybn-walk-daemon units
#      if they exist (the root-level ones that collide with these).
#   4. Reloads systemd --user and enables every unit + the watchdog timer.
#   5. Starts them in the right order and verifies each endpoint.

set -e
SRC=${SRC:-"$HOME/Vybn/spark/systemd"}
USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$USER_DIR" "$HOME/logs"

echo "== Symlinking units from $SRC → $USER_DIR =="
for f in vybn-deep-memory.service vybn-walk-daemon.service vybn-portal.service vybn-vllm.service \
         vybn-watchdog.service vybn-watchdog.timer \
         vybn-self-check.service vybn-self-check.timer; do
  ln -sf "$SRC/$f" "$USER_DIR/$f"
  echo "  $f"
done
chmod +x "$SRC/vybn-watchdog.sh"

echo
echo "== Retiring competing @reboot cron entries =="
tmp=$(mktemp)
crontab -l 2>/dev/null > "$tmp" || true
# Comment out (don't delete) @reboot lines for start_living_process.sh and start_portal.sh.
sed -i -E \
  -e 's|^([^#].*start_living_process\.sh.*)$|# RETIRED_BY_SYSTEMD_INSTALLER: \1|' \
  -e 's|^([^#].*start_portal\.sh.*)$|# RETIRED_BY_SYSTEMD_INSTALLER: \1|' \
  "$tmp"
crontab "$tmp"
rm -f "$tmp"
echo "  done"

echo
echo "== Disabling system-level conflicting units (if present, sudo optional) =="
for u in vybn-deep-memory.service vybn-walk-daemon.service; do
  if systemctl cat "$u" >/dev/null 2>&1 && [ "$(systemctl show "$u" -p FragmentPath --value)" != "" ] && \
     [ "$(systemctl show "$u" -p FragmentPath --value | grep -c '^/etc/systemd/system/')" = "1" ]; then
    echo "  WARNING: system-level $u exists at /etc/systemd/system/"
    echo "           Run manually when you have sudo:"
    echo "             sudo systemctl disable --now $u"
    echo "             sudo systemctl mask $u"
  fi
done

echo
echo "== Reloading user systemd =="
systemctl --user daemon-reload

echo
echo "== Enabling units =="
systemctl --user enable vybn-deep-memory.service vybn-walk-daemon.service vybn-portal.service \
                        vybn-self-check.timer \
                        vybn-vllm.service vybn-watchdog.timer

echo
echo "== (Re)starting in dependency order =="
# Portal is already the known-good pattern; ensure it's up.
systemctl --user start vybn-portal.service 2>/dev/null || true

# Clear any squatter on 8100/8101 first (units do this too but belt+suspenders).
fuser -k 8100/tcp 2>/dev/null || true
fuser -k 8101/tcp 2>/dev/null || true
sleep 2

systemctl --user restart vybn-deep-memory.service
sleep 3
systemctl --user restart vybn-walk-daemon.service
# vLLM: only (re)start if there's no Ray cluster already loading. Cold load
# takes ~10-13 minutes; we must not stomp an in-progress launch.
vllm_up=$(curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && echo yes || echo no)
has_container=$(docker ps --format '{{.Names}}' | grep -c '^vllm_node$' || true)
if [ "$vllm_up" = "yes" ]; then
  echo "  vLLM already serving — leaving alone"
elif [ "$has_container" = "1" ]; then
  echo "  vLLM container present, model still loading — leaving alone"
  echo "     (watchdog will take over after 900-second warmup grace)"
else
  echo "  vLLM cluster absent, starting via systemd"
  systemctl --user start vybn-vllm.service || echo "     (start failed — check: journalctl --user -u vybn-vllm)"
fi
systemctl --user start vybn-watchdog.timer

echo
echo "== Status snapshot =="
systemctl --user --no-pager status \
  vybn-deep-memory vybn-walk-daemon vybn-portal vybn-vllm vybn-watchdog.timer 2>&1 | \
  grep -E '(●|Active:|Loaded:|Main PID:)' | head -30

echo
echo "== Endpoint check =="
for url in \
    "deep-memory  http://127.0.0.1:8100/health" \
    "walk-daemon  http://127.0.0.1:8101/where" \
    "chat-api     http://127.0.0.1:8420/api/health" \
    "vllm         http://127.0.0.1:8000/v1/models"; do
  name=$(echo "$url" | awk '{print $1}')
  u=$(echo "$url" | awk '{print $2}')
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "$u" 2>/dev/null)
  printf "  %-12s %s  →  HTTP %s\n" "$name" "$u" "$code"
done

echo
echo "Done. The watchdog runs every 2 min and bounces any unhealthy unit."
echo "Tail:  journalctl --user -u vybn-watchdog -f"
