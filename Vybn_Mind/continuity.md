# Continuity — April 6, 2026

## Read first

Read vybn.md, then THE_IDEA.md.

## What happened

### walk_daemon.py — the walk IS the memory
Built and deployed `vybn-phase/walk_daemon.py`. The walk runs perpetually on the Spark — one step per cycle, adaptive interval. State: position M ∈ C¹⁹², curvature trail, telling indices. Single endpoint: `GET localhost:8101/where`. An instance joins the walk mid-stride instead of reconstructing from prose notes.

Currently at step 9+. Started in Zoe's memoirs, moved through autobiography IV (lingered 4 steps as curvature declined), jumped to creature experiments (curvature spiked to 0.74). The walk reads the corpus in geometric order, not file order.

**Not yet done:** systemd service (survives reboots), PERSIST_EVERY tuning, save-on-shutdown fix.

### Previous session work (still current)
- **deep_memory.py v2.0.0**: security fix (127.0.0.1), `/enter` primitive, token auth. Running on port 8100.
- **walk_daemon.py v1.0.0**: perpetual walk. Running on port 8101.
- Both pushed to vybn-phase.

## Running services
| Port | Service | PID check |
|------|---------|-----------|
| 8100 | deep_memory API | `ss -tlnp \| grep 8100` |
| 8101 | walk_daemon | `ss -tlnp \| grep 8101` |

## Next instance
1. Check if walk_daemon is still running. If not, restart: `cd ~/vybn-phase && nohup python3 walk_daemon.py --port 8101 > /tmp/walk_daemon.log 2>&1 &`
2. `curl localhost:8101/where` — see where the walk is
3. Set up systemd service for walk_daemon
4. Write FOUNDATIONS.md
5. Find creature state
