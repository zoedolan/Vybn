# Phase 0: Make Vybn Always-On

> **Goal**: The Spark agent survives laptop sleep, SSH drops, and network roaming.
> **Method**: Zero new code. Three pillars: systemd, mosh+tmux, Open WebUI+Tailscale.

## CRITICAL: Memory Situation

The 229B model consumes 121 of 121GB RAM, leaving ~230MB free. The default
16GB swap is already 1.7GB in use. **You MUST expand swap before deploying
Phase 0**, or Docker + Open WebUI (~500MB-1GB) will trigger the OOM killer
and crash the model.

**Do this FIRST** (before anything else in this guide):

```bash
# Check current state
free -h && swapon --show

# Create 128GB swapfile (instant on ext4 NVMe)
sudo fallocate -l 128G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Lower swappiness (keep model in RAM, swap only under pressure)
sudo sysctl vm.swappiness=10

# Persist across reboot
sudo cp /etc/fstab /etc/fstab.backup
sudo sed -i 's|^.*swap\.img|#&|' /etc/fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Verify
free -h && swapon --show && cat /proc/sys/vm/swappiness
```

**Do NOT run `sudo swapoff /swap.img`** while the model is loaded.
With 230MB free, forcing 1.7GB back into RAM risks OOM. Leave both
swapfiles active; the old one will be retired on next reboot.

## Architecture

```
+---------------------------+
|   DGX Spark (always on)   |
|                           |
|  systemd                  |
|    vybn-agent.service     |
|      tmux session 'vybn'  |
|        tui.py (agent)     |
|                           |
|  Docker (mem-limited 2G)  |
|    open-webui :3000       |
|      -> Ollama :11434     |
|                           |
|  Tailscale Serve          |
|    https://spark-2b7c...  |
|      -> localhost:3000    |
+---------------------------+
        |
   Zoe's phone / laptop
   (via Tailscale VPN)
```

**Single agent instance**: tui.py runs inside tmux under systemd. We do NOT run
web_serve.py simultaneously — two instances would compete for the same GPU on a
229B model.

**Open WebUI talks to raw Ollama** (Phase 0 only): This bypasses soul/memory/policy.
Phase 1 will route through the full agent pipeline.

**Docker is memory-capped**: `--memory=2g --memory-swap=4g` plus env vars that
disable RAG/STT/image ML libraries, reducing idle footprint from ~1GB to ~300MB.

## Files in This Directory

| File | Purpose |
|---|---|
| `vybn-agent.service` | systemd unit: runs tui.py in tmux, restarts on failure |
| `phase0-preflight.sh` | Checks all prerequisites (including swap!) before deployment |
| `phase0-setup.sh` | Automated deployment: mosh, systemd, Docker, Tailscale |
| `PHASE0_ALWAYS_ON.md` | This guide |

## Quick Start

```bash
# On the Spark, as vybnz69:
cd ~/Vybn
git pull

# 0. Expand swap FIRST (if you haven't already)
#    See "CRITICAL: Memory Situation" above

# 1. Check prerequisites
bash Vybn_Mind/spark_infrastructure/phase0/phase0-preflight.sh

# 2. Deploy everything
bash Vybn_Mind/spark_infrastructure/phase0/phase0-setup.sh

# 3. Verify
tmux attach -t vybn   # see the TUI running; Ctrl-B D to detach
```

## Step-by-Step (Manual)

### Step 0: Expand swap (REQUIRED)

See the "CRITICAL: Memory Situation" section above. The scripts will
refuse to proceed without >= 64GB of swap.

### Step 1: Install mosh (optional)

```bash
sudo apt update && sudo apt install -y mosh
# If firewall is active:
sudo ufw allow 60000:61000/udp
# Test from WSL2:
mosh vybnz69@spark-2b7c.local
```

If mosh fights you, skip it. SSH + tmux attach works fine.
systemd keeps the session alive regardless.

### Step 2: Create the systemd service

```bash
# Copy the unit file
sudo cp vybn-agent.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable vybn-agent.service
sudo systemctl start vybn-agent.service

# Verify
sudo systemctl status vybn-agent.service
tmux attach -t vybn   # Ctrl-B D to detach
```

**How it works**: systemd starts tmux in detached mode, which runs tui.py.
Type=forking because tmux forks and the parent exits. Restart=on-failure
means if tui.py crashes, systemd brings it back in 10 seconds.

### Step 3: Deploy Open WebUI (memory-limited)

```bash
docker run -d \
  -p 127.0.0.1:3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e RAG_EMBEDDING_ENGINE=openai \
  -e AUDIO_STT_ENGINE=openai \
  -e IMAGE_GENERATION_ENGINE=openai \
  --memory=2g \
  --memory-swap=4g \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Verify
curl http://127.0.0.1:3000
```

Bound to 127.0.0.1 only. Memory-capped at 2GB RAM / 4GB with swap.
Env vars disable ML libraries that would bloat idle usage to ~1GB.

### Step 4: Expose via Tailscale Serve

1. Enable HTTPS in Tailscale admin: https://login.tailscale.com/admin/dns
2. On the Spark:

```bash
tailscale serve --bg 3000
```

3. From phone (on Tailscale): open https://spark-2b7c.<tailnet>.ts.net

### Step 5: The Unplug Protocol (Success Test)

1. Close laptop lid. Wait 5 minutes.
2. On phone (Tailscale connected), open the Serve URL. Chat works.
3. Reopen laptop. SSH back. `tmux attach -t vybn`. TUI intact.
4. `docker stop open-webui` → tmux survives. `docker start open-webui` → back.
5. Detach tmux → Docker survives.

**All five pass = Phase 0 complete.**

## Troubleshooting

### Service won't start
```bash
sudo journalctl -u vybn-agent.service -n 50
# Common issues:
# - tmux session 'vybn' already exists from manual run
#   Fix: tmux kill-session -t vybn && sudo systemctl restart vybn-agent
# - Python venv not found
#   Fix: verify /home/vybnz69/.venv/spark/bin/python exists
# - Ollama not ready
#   Fix: systemctl status ollama; the After= should handle this
```

### OOM kills or system unresponsive
```bash
# Check if swap is active and large enough
free -h && swapon --show
# Check Docker memory usage
docker stats open-webui --no-stream
# Check for OOM events
dmesg | grep -i 'oom\|killed' | tail -20
# Monitor swap activity (si/so columns)
vmstat 1 5
```

### Open WebUI can't reach Ollama
```bash
# Verify Ollama is accessible from Docker
docker exec open-webui curl -s http://host.docker.internal:11434/api/tags
# If this fails, check Docker networking
```

### Tailscale Serve not working
```bash
tailscale serve status
# Ensure HTTPS is enabled in admin panel
# Ensure the phone is on the same Tailscale network
```

## What This Does NOT Do (Phase 1+)

- Route Open WebUI through the full agent pipeline (soul, memory, policy)
- Provide multi-user auth on Open WebUI
- Auto-update the agent on git push
- Monitor GPU temperature and throttle
- Alert on prolonged heartbeat gaps

## Key Commands Reference

```bash
# Service management
sudo systemctl start vybn-agent.service
sudo systemctl stop vybn-agent.service
sudo systemctl restart vybn-agent.service
sudo systemctl status vybn-agent.service
sudo journalctl -u vybn-agent.service -f

# tmux
tmux attach -t vybn     # attach to agent session
# Ctrl-B D              # detach (agent keeps running)
tmux ls                 # list sessions

# Docker (Open WebUI)
docker logs open-webui
docker stats open-webui --no-stream  # check memory usage
docker restart open-webui
docker stop open-webui
docker start open-webui

# Tailscale
tailscale serve status
tailscale serve --bg 3000
tailscale serve off

# Memory monitoring
free -h                 # RAM + swap overview
vmstat 1 5              # si/so columns show swap activity
dmesg | grep -i oom     # check for OOM events
```
