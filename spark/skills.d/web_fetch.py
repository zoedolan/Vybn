"""Web fetching skill - fetch URLs and read web content.

SKILL_NAME: web_fetch
TOOL_ALIASES: ["web_fetch", "curl", "fetch_url", "http_get"]
"""

import subprocess
from pathlib import Path

SKILL_NAME = "web_fetch"
TOOL_ALIASES = ["web_fetch", "curl", "fetch_url", "http_get"]


def execute(action: dict, router) -> str:
    """Fetch content from a URL using curl."""
    params = action.get("params", {})
    url = (
        action.get("argument", "")
        or params.get("url", "")
        or params.get("address", "")
    )
    
    if not url:
        return "no URL specified"
    
    # Clean up URL
    url = url.strip().rstrip('.,;:!?"\'')
    
    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    try:
        result = subprocess.run(
            ["curl", "-L", "-s", "--max-time", "30", "--max-filesize", "10M", url],
            capture_output=True,
            text=True,
            timeout=35,
        )
        
        if result.returncode == 0:
            content = result.stdout[:50000]  # Limit to 50KB
            return f"fetched {url} ({len(result.stdout):,} chars, showing first 50KB):\n{content}"
        else:
            return f"failed to fetch {url}: {result.stderr[:500]}"
    
    except subprocess.TimeoutExpired:
        return f"fetch timed out for {url}"
    except Exception as e:
        return f"web fetch error: {e}"
