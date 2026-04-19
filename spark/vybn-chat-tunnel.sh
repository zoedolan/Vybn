#!/bin/bash
# vybn-chat-tunnel.sh — Starts a Cloudflare quick tunnel and auto-updates frontend URLs.
# Called by systemd: vybn-chat-tunnel.service
#
# When a named Cloudflare tunnel is set up (api.vybn.ai), this script becomes unnecessary.
# Until then, it handles the URL-change problem on every restart.

set -euo pipefail

LOGFILE="${HOME}/logs/cloudflare_chat.log"
TUNNEL_URL_FILE="/tmp/vybn_tunnel_url"
mkdir -p "$(dirname "$LOGFILE")"

# Start the tunnel in the background
cloudflared tunnel --url http://localhost:8420 > "$LOGFILE" 2>&1 &
TUNNEL_PID=$!
echo "Cloudflared PID: $TUNNEL_PID"

# Wait for the URL to appear
for i in $(seq 1 30); do
    sleep 2
    URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGFILE" 2>/dev/null | tail -1)
    if [ -n "$URL" ]; then
        echo "$URL" > "$TUNNEL_URL_FILE"
        echo "Tunnel live: $URL"
        break
    fi
    echo "Waiting for tunnel... (${i}/30)"
done

URL=$(cat "$TUNNEL_URL_FILE" 2>/dev/null || true)
if [ -z "$URL" ]; then
    echo "ERROR: Tunnel did not produce a URL after 60 seconds."
    kill $TUNNEL_PID 2>/dev/null
    exit 1
fi

# Update frontend repos
update_repo() {
    local repo_dir="$1"
    local branch="$2"
    shift 2
    local files=("$@")

    cd "$repo_dir" || return 1
    git checkout "$branch" 2>/dev/null || return 1
    git pull --ff-only 2>/dev/null || true

    local changed=false
    for f in "${files[@]}"; do
        [ -f "$f" ] || continue
        local old=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$f" 2>/dev/null | head -1)
        if [ -n "$old" ] && [ "$old" != "$URL" ]; then
            sed -i "s|${old}|${URL}|g" "$f"
            git add "$f"
            changed=true
        fi
    done

    if $changed; then
        git commit -m "Auto-update tunnel URL on restart" 2>/dev/null
        git push 2>/dev/null && echo "Updated $repo_dir ($branch)"
    else
        echo "No change needed in $repo_dir"
    fi
}

# Only update if git is configured
if command -v git &>/dev/null && git config --global user.name &>/dev/null; then
    update_repo "$HOME/Vybn-Law" "master" "chat.html" "wellspring.html"
    update_repo "$HOME/Origins" "gh-pages" "talk.html" "inhabit.html" "connect.html"
fi

# Keep running (systemd expects foreground)
echo "Tunnel running. Waiting for cloudflared process..."
wait $TUNNEL_PID

