# Vybn SSH MCP Server

Bridges Perplexity Computer to the DGX Sparks via SSH over Tailscale.

```
Perplexity ──HTTPS──▶ Tailscale Funnel ──▶ This Server ──SSH──▶ DGX Spark(s)
```

## Tools

| Tool | What it does |
|------|-------------|
| `shell_exec` | Run any command on a Spark (with safety rails for destructive ops) |
| `read_file` | Read a file from a Spark |
| `write_file` | Write content to a file on a Spark |
| `gpu_status` | GPU utilization, memory, temperature, running processes |
| `sensorium` | Run the Vybn sensorium and return its perception |
| `model_status` | Check active models, inference endpoints, GPU memory |
| `repo_status` | Git status and recent commits for any repo on the Spark |
| `continuity` | Read the Spark-resident Vybn's continuity note |
| `journal` | Read recent journal entries from the Spark-resident Vybn |

## Quick Start

```bash
chmod +x setup.sh
./setup.sh
```

The setup script creates a venv, installs dependencies, generates an API key,
and walks you through Tailscale Funnel + Perplexity registration.

See [DEPLOY.md](DEPLOY.md) for the full guide.

## Security

- SSH key auth only, no passwords
- Tailscale Funnel handles TLS — no self-signed certs
- API key auth on the MCP layer
- Destructive commands require explicit confirmation
- Runs as your user — same permissions as your SSH session
