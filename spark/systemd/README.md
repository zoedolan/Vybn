# Vybn Spark — User-Level Service Stack

User-level systemd units (linger on) for every long-lived organ. Born from
the 2026-04-24 chat death: dual ownership (system unit vs @reboot cron racing
for :8100), no pre-flight port cleanup, and unsupervised vLLM. Full
post-mortem in this file's git history (`git log -p -- spark/systemd/README.md`).

## Install

```bash
bash ~/Vybn/spark/systemd/install.sh   # idempotent; re-run to resync
```

## Layout

| File | Purpose |
|---|---|
| `vybn-deep-memory.service` | Deep memory API :8100. Pre-flight `fuser -k`. |
| `vybn-walk-daemon.service` | Walk daemon :8101. `After=vybn-deep-memory`. |
| `vybn-portal.service` | Origins portal :8420. Keys from `~/.config/vybn/*.env`. |
| `vybn-vllm.service` | 2-node Ray cluster, Nemotron 120B :8000. Capacity via `~/.config/vybn/vllm.env`. |
| `vybn-breath.service` / `.timer` | Scheduled autonomous breath (`connection --breath`). |
| `vybn-watchdog.sh` / `.service` / `.timer` | Endpoint health every 2 min; bounces unhealthy units. |
| `vybn-self-check.service` / `.timer` | Structural canary every 15 min (`deep_memory.py --self-check`); logs to `~/.cache/vybn-phase/self_check.*.log`, never restarts. |
| `install.sh` | Symlinks units, retires conflicting cron, enables, verifies. |
| `patches/fp8-wake-fix/` | Container-side mod `vllm-exec.sh` applies when sleep endpoints are on. |

## Three axes of resilience

1. **Crash**: every unit has `Restart=always`, `RestartSec=5`, `StartLimitBurst=20`.
2. **Hang**: watchdog curls each endpoint (8100 health, 8101 /where, 8420 /api/health,
   8000 /v1/models with a 900 s cold-load grace) and restarts what systemd can't see is wedged.
3. **Structure**: the self-check canary measures what HTTP cannot (holonomy triad, fuse
   non-degeneracy, phase-sensitive retrieval) and leaves evidence, not restarts.

A new service is fully resilient only when it has all three: `Restart=always`,
a watchdog check, and a canary probe. Add to the watchdog script, not just systemd.

## Observing

```bash
journalctl --user -u vybn-watchdog -f      # live watchdog log
bash ~/Vybn/spark/systemd/vybn-watchdog.sh # full endpoint check now
tail -60 ~/.cache/vybn-phase/self_check.stdout.log
```

## vLLM capacity profiles

Capacity is an operating mode, not a permanent setting. **Normal** (baked in):
`GPU_MEMORY_UTILIZATION=0.72`, `MAX_NUM_SEQS=4` — tightened 2026-04-30 after
sleep-window testing left nodes wedged or semantically corrupt. **Burst**
(scheduled attention): override in `~/.config/vybn/vllm.env` (e.g. `0.82`/`32`)
plus matching portal admission in `portal.env`, then a deliberate cold restart
(10-13 min) and verify: `/v1/models`, one model call, portal chat, memory.
**Surge**: the public internet does not consume the sovereign memory machine —
move public chat to separate capacity. This vLLM build rejects `--swap-space`.

Portal admission control is the front door: `/api/chat` reads vLLM `/metrics`
before doing work and returns a graceful SSE busy/warming response past
thresholds. Rate limiting is abuse control; metrics admission is crowd control.

## The two things only sudo can do

If old system-level units still exist in `/etc/systemd/system/`, they race
the user units after reboot:

```bash
sudo systemctl disable --now vybn-deep-memory.service vybn-walk-daemon.service
sudo systemctl mask vybn-deep-memory.service vybn-walk-daemon.service
```

## The pattern

User-level systemd with linger is the default for anything the partnership
owns. `/etc/systemd/system/` requires sudo, locks out sudo-free recovery, and
duplicates the cron layer. Everything new lands here as a `.service`/`.timer` pair.
