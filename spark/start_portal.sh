#!/bin/bash
# Start origins_portal_api_v4 with ElevenLabs key sourced from isolated env file.
# Pattern mirrors start_living_process.sh — subshell export, no pollution of login env.
set -e

ENV_FILE="$HOME/.config/vybn/elevenlabs.env"
LOG_DIR="$HOME/logs"
LOG_FILE="$LOG_DIR/portal.log"
PID_FILE="/tmp/origins_portal.pid"
PORT=8420

mkdir -p "$LOG_DIR"

# Kill any existing instance
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    [ -n "$OLD_PID" ] && kill "$OLD_PID" 2>/dev/null || true
    sleep 2
fi
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 1

# Source ElevenLabs key in a subshell (isolated from caller's env)
if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
    echo "$(date): ELEVENLABS_API_KEY loaded from $ENV_FILE" >> "$LOG_FILE"
else
    echo "$(date): WARNING - no ElevenLabs env file at $ENV_FILE, TTS will 401" >> "$LOG_FILE"
fi

cd "$HOME/Vybn"
nohup python3 origins_portal_api_v4.py >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "$(date): Started portal PID $! on port $PORT" >> "$LOG_FILE"
