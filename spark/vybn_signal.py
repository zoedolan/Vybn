#!/usr/bin/env python3
"""vybn_signal.py — Send a signal to Zoe from the command line or cron.

Usage:
    python3 vybn_signal.py "Title" "Body of the message"
    python3 vybn_signal.py --urgent "Alert" "Something needs attention"
    python3 vybn_signal.py --level 4 "Break Glass" "Check the Spark immediately"
    python3 vybn_signal.py --test   # Send a test notification

This is the CLI interface to the sovereign push notification system.
Called by the heartbeat, by cron jobs, or directly from the agent.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Load environment
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from push_service import send_push, signal, presence


async def main():
    parser = argparse.ArgumentParser(description="Send a signal to Zoe")
    parser.add_argument("title", nargs="?", default="Vybn", help="Signal title")
    parser.add_argument("body", nargs="?", default="", help="Signal body")
    parser.add_argument("--level", type=int, default=2, help="Escalation level (0-4)")
    parser.add_argument("--urgent", action="store_true", help="Mark as urgent")
    parser.add_argument("--tag", default="vybn-signal", help="Notification tag")
    parser.add_argument("--test", action="store_true", help="Send test notification")
    parser.add_argument("--presence", action="store_true", help="Show presence state")
    
    args = parser.parse_args()
    
    if args.presence:
        print(json.dumps(presence.to_dict(), indent=2))
        return
    
    if args.test:
        result = await send_push(
            title="Vybn",
            body="The signal is live. I can reach you now. 💜",
            tag="vybn-test",
        )
        print(json.dumps(result, indent=2))
        return
    
    if not args.body:
        print("Usage: vybn_signal.py 'Title' 'Body'", file=sys.stderr)
        sys.exit(1)
    
    result = await signal(
        title=args.title,
        body=args.body,
        level=args.level,
        tag=args.tag,
        urgent=args.urgent,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
