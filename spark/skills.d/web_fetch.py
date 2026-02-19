"""Web fetching skill - fetch URLs and read web content.

SKILL_NAME: web_fetch
TOOL_ALIASES: ["web_fetch", "curl", "fetch_url", "http_get"]
"""
import ipaddress
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

SKILL_NAME = "web_fetch"
TOOL_ALIASES = ["web_fetch", "curl", "fetch_url", "http_get"]

# SECURITY: Block requests to internal/private network addresses.
# Prevents SSRF attacks where an attacker tricks Vybn into fetching
# cloud metadata, LAN admin panels, or localhost services.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud metadata
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

def _is_safe_url(url: str) -> bool:
    """Check that a URL does not resolve to a private/internal IP."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP(s)
        addrs = socket.getaddrinfo(hostname, parsed.port or 443)
        for family, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            for net in _BLOCKED_NETWORKS:
                if ip in net:
                    return False
        return True
    except (socket.gaierror, ValueError, OSError):
        return False


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

    # SECURITY: Block internal/private network requests (SSRF protection)
    if not _is_safe_url(url):
        return (
            f"blocked: {url} resolves to a private/internal IP address. "
            f"Web fetch is restricted to public internet addresses."
        )

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
