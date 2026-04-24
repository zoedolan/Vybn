# Vybn Spark — User-Level Service Stack

The chats died on 2026-04-24 because the Spark's service layer had three structural defects:

1. **Dual ownership**. The same service (deep-memory on :8100) was owned by both a system-level unit and an `@reboot` cron running `start_living_process.sh`. Whichever started first held the port; the other crash-looped until `StartLimitBurst` tripped. The system-level unit then required `sudo systemctl reset-failed` to recover.
2. **No pre-flight port cleanup**. Neither the unit nor the cron cleared squatters before binding.
3. **vLLM had no supervisor at all**. `vllm.service` was masked; `launch-cluster.sh` was invoked ad hoc. A reboot left the stack chatless because nothing brought vLLM back up.

This directory consolidates everything into **user-level** systemd units (linger is on, same pattern as the already-healthy `vybn-portal.service`), adds pre-flight port cleanup, adds a 2-minute watchdog timer that bounces anything unhealthy, and retires the conflicting cron paths.

## Install

```bash
bash ~/Vybn/spark/systemd/install.sh
```

Idempotent. Re-run any time to resynchronize the stack with the repo.

## Layout

| File | Purpose |
|---|---|
| `vybn-deep-memory.service` | Deep memory API on :8100. Pre-flight `fuser -k 8100/tcp`. |
| `vybn-walk-daemon.service` | Walk daemon on :8101. Ordered `After=vybn-deep-memory`. |
| `vybn-vllm.service` | Supervises the 2-node Ray cluster serving Nemotron 120B on :8000. `ExecStartPre` clean-stops; `ExecStart` runs launcher in foreground so systemd supervises; `--distributed-executor-backend ray` is required for 1 GPU per node. |
| `vybn-watchdog.sh` | Health-check script: curls each endpoint, bounces its unit if unhealthy. |
| `vybn-watchdog.service` | Oneshot that runs the script. |
| `vybn-watchdog.timer` | Every 2 minutes, 90s after boot. |
| `install.sh` | Installer: symlinks units into `~/.config/systemd/user/`, retires conflicting cron, reloads, enables, starts, verifies. |

## What the Watchdog Catches

Every 2 minutes (after a 90s boot grace) it checks:

- `http://127.0.0.1:8100/health` → 200/401/403 (any of those proves alive; 401/403 just means auth is set).
- `http://127.0.0.1:8101/where` → 200.
- `http://127.0.0.1:8420/api/health` → 200 (bounces `vybn-portal.service`).
- `http://127.0.0.1:8000/v1/models` → 200. A 5-minute grace applies from the last `ActiveEnter` of `vybn-vllm`, because cold model load takes ~3 minutes and we don't want a restart loop while it's still warming.

If a check fails, the watchdog runs `systemctl --user restart <unit>`. The unit's own `Restart=always` handles transient crashes; the watchdog handles the case where systemd thinks it's running but the endpoint is hung.

## Observing

```bash
# Live watchdog log
journalctl --user -u vybn-watchdog -f

# Any unit's status
systemctl --user status vybn-vllm

# Full endpoint check right now
bash ~/Vybn/spark/systemd/vybn-watchdog.sh
```

## The Two Things Only sudo Can Do

The installer writes user-level units without `sudo`. But if the old system-level `vybn-deep-memory.service` and `vybn-walk-daemon.service` are still enabled in `/etc/systemd/system/`, they'll race us after reboot. The installer warns if it detects them; to finish the cutover run:

```bash
sudo systemctl disable --now vybn-deep-memory.service vybn-walk-daemon.service
sudo systemctl mask vybn-deep-memory.service vybn-walk-daemon.service
```

After that, the user-level units are the only path.

## The Pattern

User-level systemd with linger is the default for anything the partnership owns. Writing to `/etc/systemd/system/` requires sudo, locks us out of sudo-free recovery, and duplicates the cron layer. Everything we add from here on should go in this directory as a `.service` / `.timer` pair.
