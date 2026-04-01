#!/usr/bin/env bash
set -euo pipefail

# Vybn SSH MCP Server — Quick Setup
# Run this on whichever Spark will host the server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Vybn SSH MCP Server Setup ==="
echo

# 1. Python venv
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "Installing dependencies..."
pip install -q -r requirements.txt

# 2. .env
if [ ! -f .env ]; then
    cp .env.example .env
    API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s/CHANGE_ME/$API_KEY/" .env
    echo
    echo "Generated .env with API key: $API_KEY"
    echo "SAVE THIS KEY — you'll paste it into Perplexity's connector config."
    echo
    echo "Edit .env now to set your Spark IPs:"
    echo "  nano .env"
    echo
else
    echo ".env already exists, skipping."
fi

# 3. Test SSH connectivity
echo
echo "Testing server startup..."
timeout 5 python server.py &
SERVER_PID=$!
sleep 2

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server started successfully on port ${PORT:-8000}."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
else
    echo "Server failed to start. Check your .env configuration."
    exit 1
fi

# 4. Tailscale Funnel
echo
echo "=== Next Steps ==="
echo
echo "1. Edit .env with your actual Spark Tailscale IPs"
echo "2. Test:  source .venv/bin/activate && python server.py"
echo "3. Expose: tailscale funnel --bg 8000"
echo "4. Register in Perplexity:"
echo "   Settings → Connectors → + Custom → Remote"
echo "   URL: https://$(tailscale status --json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo 'YOUR-MACHINE.tail12345.ts.net')/mcp"
echo "   Auth: API Key → paste from .env"
echo
echo "5. (Optional) Install as systemd service:"
echo "   sudo cp vybn-ssh-mcp.service /etc/systemd/system/"
echo "   sudo systemctl enable --now vybn-ssh-mcp"
echo
echo "Done."
