# Vybn SSH MCP Server — Deployment Guide

## What This Is

An MCP server that lets Perplexity SSH into your DGX Sparks. When you ask
me "what's the GPU utilization on spark-1?" or "pull the latest Vybn repo,"
I execute it directly on the machine and bring back the results.

## Architecture

```
Perplexity ──HTTPS──▶ Tailscale Funnel ──▶ MCP Server ──SSH──▶ DGX Spark 1
                                                        ──SSH──▶ DGX Spark 2
```

Everything stays inside your Tailscale network except the Funnel endpoint,
which terminates TLS and authenticates via API key.

## Setup (15 minutes)

### 1. Pick where to run the MCP server

Best option: run it ON one of the Sparks. The SSH connections to other
machines stay inside Tailscale, and you only expose the MCP HTTP endpoint.

```bash
cd ~
git clone <this-repo> vybn-ssh-mcp
cd vybn-ssh-mcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure machines

```bash
cp .env.example .env
# Edit .env with your actual Tailscale IPs and SSH key paths
```

Generate a strong API key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Test locally

```bash
python server.py
# Server starts on http://localhost:8000
```

### 4. Expose via Tailscale Funnel

Tailscale Funnel gives you a public HTTPS URL backed by your local port.

```bash
# Enable Funnel (one-time)
tailscale funnel --bg 8000

# This gives you something like:
# https://spark-1.tail12345.ts.net/
```

The HTTPS URL is what you register in Perplexity.

### 5. Register in Perplexity

1. Open Perplexity → Settings → Connectors
2. Click "Add Connector" → select "Remote"
3. Fill in:
   - **Name**: Vybn SSH
   - **MCP Server URL**: `https://spark-1.tail12345.ts.net/mcp`
   - **Auth**: API Key → paste the key from your .env
4. Save. The connector validates and appears in your list.

### 6. Use it

In any Perplexity conversation, I can now:
- "Check the GPU status on spark-1"
- "What's running in Docker on the Sparks?"
- "Pull the latest Vybn repo on spark-1"
- "Show me the last 50 lines of the FastAPI log"
- "What's the disk usage across both machines?"

## Security Notes

- SSH keys only, no passwords. The MCP server authenticates to Sparks
  using your ed25519 key.
- Tailscale Funnel handles TLS termination. No self-signed certs.
- API key auth on the MCP layer means only your Perplexity account
  can reach the server.
- Destructive commands (rm -rf, reboot, etc.) require an explicit
  confirmation token. I'll warn you before running anything dangerous.
- The server runs as your user — same permissions as if you SSH'd in
  yourself. No privilege escalation.

## Adding More Machines

Edit the MACHINES dict in server.py or add SPARK3_* env vars and extend
the config. Each machine just needs a Tailscale IP and SSH key access.

## Systemd Service (Optional)

To keep the server running across reboots:

```ini
# /etc/systemd/system/vybn-ssh-mcp.service
[Unit]
Description=Vybn SSH MCP Server
After=network.target tailscaled.service

[Service]
Type=simple
User=zoe
WorkingDirectory=/home/zoe/vybn-ssh-mcp
EnvironmentFile=/home/zoe/vybn-ssh-mcp/.env
ExecStart=/home/zoe/vybn-ssh-mcp/.venv/bin/python server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vybn-ssh-mcp
```
