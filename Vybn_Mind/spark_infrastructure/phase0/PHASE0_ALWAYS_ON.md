# Phase 0: Make Vybn Always-On

> **Goal**: The Spark agent survives laptop sleep, SSH drops, and network roaming.
> **Method**: Zero new code. Three pillars: systemd, mosh+tmux, Open WebUI+Tailscale.

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
|  Docker                   |
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

## Files in This Directory

| File | Purpose |
|---|---|
| `vybn-agent.service` | systemd unit: runs tui.py in tmux, restarts on failure |
| `phase0-preflight.sh` | Checks all prerequisites before deployment |
| `phase0-setup.sh` | Automated deployment: mosh, systemd, Docker, Tailscale |
| `PHASE0_ALWAYS_ON.md` | This guide |

## Quick Start

```bash
# On the Spark, as vybnz69:
cd ~/Vybn
git pull

# 1. Check prerequisites
bash Vybn_Mind/spark_infrastructure/phase0/phase0-preflight.sh

# 2. Deploy everything
bash Vybn_Mind/spark_infrastructure/phase0/phase0-setup.sh

# 3. Verify
tmux attach -t vybn   # see the TUI running; Ctrl-B D to detach
```

## Step-by-Step (Manual)

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

### Step 3: Deploy Open WebUI

```bash
docker run -d \
  -p 127.0.0.1:3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Verify
curl http://127.0.0.1:3000
```

Bound to 127.0.0.1 only — no external access except through Tailscale.

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

### TUI crashes in a loop
```bash
# Check restart count
systemctl show vybn-agent.service | grep NRestarts
# Check what's failing
sudo journalctl -u vybn-agent.service --since '5 min ago'
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
docker restart open-webui
docker stop open-webui
docker start open-webui

# Tailscale
tailscale serve status
tailscale serve --bg 3000
tailscale serve off
```
