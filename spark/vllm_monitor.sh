#!/bin/bash
# vLLM Health Monitor — checks both Sparks and the model endpoint
# Run: nohup bash ~/Vybn/spark/vllm_monitor.sh &

LOG=~/logs/vllm_health.log
mkdir -p ~/logs

while true; do
    ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    
    # Check vLLM endpoint
    http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:8000/health 2>/dev/null)
    
    # Check second Spark
    spark2=$(ssh -o ConnectTimeout=3 169.254.51.101 hostname 2>/dev/null)
    
    if [ "$http_code" != "200" ]; then
        echo "[$ts] ALERT: vLLM health check failed (HTTP $http_code)" | tee -a "$LOG"
    fi
    
    if [ "$spark2" != "spark-1c8f" ]; then
        echo "[$ts] ALERT: spark-1c8f unreachable" | tee -a "$LOG"
    fi
    
    # Log OK status every 10 minutes (every 20th check at 30s interval)
    if [ $((RANDOM % 20)) -eq 0 ]; then
        echo "[$ts] OK: vLLM=$http_code spark2=$spark2" >> "$LOG"
    fi
    
    sleep 30
done
