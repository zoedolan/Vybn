# Continuity Note — Truth in the Age LIVE

## What just happened
- PR #2467 merged: Truth in the Age of Intelligence (between-sessions artifact)
- Built `gateway.py` — unified ASGI dispatcher that mounts all three bootcamp backends:
  - `/` and `/signal-noise/*` → signal_noise_api.py
  - `/threshold/*` → threshold_api.py  
  - `/truth-age/*` → truth_age_api.py
- Gateway running on port 8090, behind existing Tailscale Funnel
- All three apps verified: HTTP routes ✅, WebSocket init ✅, Funnel routing ✅
- Crontab updated: `@reboot` now starts `gateway.py` instead of `signal_noise_api.py`
- Gateway committed as `86f90b5`

## What's live
- **GitHub Pages**: `zoedolan.github.io/Vybn/Vybn_Mind/signal-noise/truth-in-the-age/`
- **Backend**: All `/truth-age/*` routes served through Funnel via gateway
- **Aspect docs**: All 7 referenced docs exist and will be loaded into system prompt
- **Commons**: Harvest/commons endpoints ready, will accumulate over the week

## Architecture
```
GitHub Pages (static HTML)
    ↓ backend.js resolves URL
Tailscale Funnel (<TAILSCALE_HOSTNAME>)
    ↓ proxies to localhost:8090
gateway.py (PID on port 8090)
    ├── /signal-noise/* → signal_noise_api (SIGNAL/NOISE)
    ├── /threshold/*    → threshold_api (THRESHOLD)  
    └── /truth-age/*    → truth_age_api (Truth in the Age)
```

## Still pending from #2463
1. vLLM auto-restart on reboot
2. Heartbeat verification with new endpoint
3. Triage untracked files
4. Wire connectome bridge into breath cycle
5. Populate memory fabric

## Gateway PID
Check with: `ps aux | grep gateway | grep -v grep`
Restart: `cd ~/Vybn/Vybn_Mind/signal-noise && source ~/.vybn_keys && nohup python3 gateway.py >> ~/signal-noise.log 2>&1 &`
