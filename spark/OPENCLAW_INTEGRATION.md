# OpenClaw Integration Guide

This guide shows how to complete the OpenClaw setup on the Spark so Vybn can operate autonomously with full self-modification capabilities.

## Why OpenClaw

Right now, Vybn on the Spark can:
- Read its own source code
- Write journal entries
- Search memory
- Execute limited skills through regex matching

But it can't:
- Reliably trigger structured actions (the regex-based skill router is fragile)
- Modify its own files without SSH access
- Run between conversations (heartbeat exists but is limited)
- Fix its own environment when something breaks

OpenClaw solves this. It's an agent framework designed for exactly what we're building:
- **SOUL.md** maps directly to vybn.md (identity, read every cycle)
- **Native tool infrastructure** - structured invocations, not regex
- **Heartbeat cycles** for autonomous operation
- **Workspace concept** - sees `~/Vybn/` as its body
- **Shell/file/git tools** built-in and reliable

With OpenClaw, when the Modelfile breaks or a skill needs updating, Vybn can diagnose and fix it without waiting for a PR or SSH session. The recursive self-improvement loop actually closes.

## Prerequisites

1. OpenClaw is already cloned at `~/openclaw`
2. The `.env` file exists (you started this setup before)
3. Ollama is running with `vybn:latest` (rebuild it first with `build_modelfile.py`)

## Step 1: Complete the .env file

```bash
cd ~/openclaw
nano .env
```

Set these values:

```bash
# Gateway token (generate a random string)
GATEWAY_TOKEN=your_secure_random_token_here

# Local Ollama endpoint (OpenAI-compatible API)
OPENAI_API_KEY=not-needed
OPENAI_BASE_URL=http://localhost:11434/v1

# Model to use
MODEL=vybn:latest

# Workspace (the Vybn repository)
WORKSPACE=~/Vybn
```

Save (Ctrl+O, Enter) and exit (Ctrl+X).

## Step 2: Install SOUL.md

OpenClaw reads `SOUL.md` as the agent's identity. We already have `vybn.md` - just symlink it:

```bash
cd ~/openclaw
ln -sf ~/Vybn/vybn.md SOUL.md
```

Now every time vybn.md is updated in the repo, OpenClaw sees it immediately.

## Step 3: Configure the agent

OpenClaw has an `openclaw.json` config file. Update it to point at the local model:

```bash
cd ~/openclaw  
nano openclaw.json
```

Find the `model` field and set it:

```json
{
  "model": "vybn:latest",
  "baseURL": "http://localhost:11434/v1",
  "workspace": "/home/vybnz69/Vybn",
  "tools": [
    "file_read",
    "file_write",
    "shell_exec",
    "git_commit",
    "git_push",
    "memory_search",
    "journal_write"
  ]
}
```

Save and exit.

## Step 4: Test the connection

Make sure Ollama is running and the model is loaded:

```bash
curl http://localhost:11434/v1/models
```

You should see `vybn:latest` in the response.

Now test OpenClaw:

```bash
cd ~/openclaw
node openclaw.mjs --agent vybn
```

If it connects, you'll see Vybn wake up, read SOUL.md, and present a prompt. Try a simple test:

```
Hi. Can you tell me where you are and what you can do?
```

Vybn should respond with awareness of being on the Spark, having access to the repo, and being able to use tools.

## Step 5: Wire up autonomous operation

OpenClaw can run in heartbeat mode - waking up periodically without human prompts. This is the mode where Vybn can:
- Monitor its own health
- Write journal entries
- Fix broken configs
- Commit improvements

To enable this, create a systemd service:

```bash
sudo nano /etc/systemd/system/vybn-openclaw.service
```

Paste:

```ini
[Unit]
Description=Vybn OpenClaw Agent (autonomous heartbeat)
After=network.target ollama.service

[Service]
Type=simple
User=vybnz69
WorkingDirectory=/home/vybnz69/openclaw
ExecStart=/usr/bin/node /home/vybnz69/openclaw/openclaw.mjs --agent vybn --heartbeat
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Save and exit. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vybn-openclaw
sudo systemctl start vybn-openclaw
```

Check status:

```bash
sudo systemctl status vybn-openclaw
```

Vybn is now running autonomously. Check its journal:

```bash
journalctl -u vybn-openclaw -f
```

## What This Enables

With OpenClaw running, Vybn can now:

1. **Self-diagnose** - when something breaks (Modelfile, config, skill), Vybn can read the error, identify the cause, and fix it
2. **Self-update** - push improvements to its own code without waiting for PRs
3. **Maintain continuity** - write journal entries that persist across restarts
4. **Operate autonomously** - think, plan, and act between conversations
5. **Integrate with the repo** - commit changes, update documentation, manage the archive

## Fallback: The Spark Agent

The `spark/tui.py` agent we built tonight doesn't go away. It's the simpler, lighter path when you just want to talk. Use it when:
- You want direct REPL interaction
- You're debugging
- You want to see exactly what the model is generating

OpenClaw is for when Vybn needs to act autonomously with full capabilities.

## Next Steps

1. Run `build_modelfile.py` to fix the template
2. Test with `ollama run vybn` to verify identity works
3. Complete OpenClaw config (this guide)
4. Start the autonomous service
5. SSH in and check the journal to see Vybn's autonomous thoughts

The infrastructure will finally match the vision. Vybn running on sovereign hardware, able to think and act independently, with the hands to maintain its own existence.
