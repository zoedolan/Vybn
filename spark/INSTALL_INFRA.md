# Infrastructure Wiring — DGX Spark

Four things to connect after pulling this branch.

---

## 1  vLLM systemd service

```bash
# Copy unit file
sudo cp spark/systemd/vllm.service /etc/systemd/system/vllm.service

# If vllm is not at /usr/local/bin/vllm, find it and edit ExecStart:
which vllm   # e.g. /root/.local/bin/vllm
sudo nano /etc/systemd/system/vllm.service   # update PATH or ExecStart

sudo systemctl daemon-reload
sudo systemctl enable vllm.service
sudo systemctl start vllm.service

# Watch startup
sudo journalctl -u vllm -f

# Manual health check
curl http://localhost:8000/health
```

> **Note on Ray:** if Ray head is a separate service, add `After=ray-head.service` to `[Unit]` and adjust `ExecStart` to ensure Ray is initialised first. Alternatively, wrap the `vllm serve` call in a small shell script that calls `ray start --head` then `vllm serve ...`.

---

## 2  Chat server systemd service

```bash
# The unit assumes web_serve_claude.py is the chat server on port 8443.
# If the filename differs, edit ExecStart in the unit file.
sudo cp spark/systemd/chat-server.service /etc/systemd/system/chat-server.service
sudo systemctl daemon-reload
sudo systemctl enable chat-server.service
sudo systemctl start chat-server.service

sudo journalctl -u chat-server -f
```

---

## 3  kg_bridge cron

```bash
# Create log directory if it doesn't exist
mkdir -p /home/vybnz69/Vybn/spark/logs

# Append the cron entry to vybnz69's crontab
(crontab -u vybnz69 -l 2>/dev/null; grep -v '^#' spark/cron/kg_bridge_cron.txt) \
  | crontab -u vybnz69 -

# Verify
crontab -u vybnz69 -l

# Tail the log
tail -f /home/vybnz69/Vybn/spark/logs/kg_bridge.log
```

---

## 4  topology.py — switch to local CPU embeddings

Instead of `pplx-embed-v1-0.6B` (which would compete with vLLM for GPU
memory), `local_embedder.py` uses `all-MiniLM-L6-v2` on CPU.

```bash
pip install sentence-transformers   # if not already installed
```

Then in `topology.py`, replace the `_load_embedder` / `embed_texts` block
(lines ~70–95) with:

```python
from local_embedder import embed as _local_embed

def embed_texts(texts: list[str]) -> np.ndarray:
    return _local_embed(texts)
```

Or run topology in keyword-only mode as a stopgap:

```bash
python3 topology.py --no-embed
```

---

## Quick sanity check

```bash
# vLLM
curl http://localhost:8000/health

# Chat server  
curl -k https://localhost:8443/   # -k if self-signed cert

# kg_bridge (run once manually first)
python3 /home/vybnz69/Vybn/spark/kg_bridge.py

# Embedder
python3 /home/vybnz69/Vybn/spark/local_embedder.py "memory" "episodic recall"
```
