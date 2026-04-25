from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import socket
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ALLOWED_CONTENT_PREFIXES = ("text/", "application/json", "application/xml")

@dataclass(frozen=True)
class FetchResult:
    final_url: str
    content_type: str
    bytes_read: int
    text: str

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def _public_host(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        return ip.is_global
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        ip_text = info[4][0]
        try:
            if not ipaddress.ip_address(ip_text).is_global:
                return False
        except ValueError:
            return False
    return True

def validate_url(url: str, allowed_hosts: Iterable[str] | None = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("refused: HTTPS required")
    if parsed.username or parsed.password:
        raise ValueError("refused: credentials in URL")
    if not parsed.hostname:
        raise ValueError("refused: missing host")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if allowed_hosts is not None and host not in {h.lower() for h in allowed_hosts}:
        raise ValueError("refused: host not allowlisted")
    if parsed.port not in (None, 443):
        raise ValueError("refused: nonstandard HTTPS port")
    if not _public_host(host):
        raise ValueError("refused: host does not resolve only to public IP addresses")
    return url

def extract_text(content: str, content_type: str) -> str:
    if "html" not in content_type.lower():
        return content
    class Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.capture = None
            self.parts = []
        def handle_starttag(self, tag, attrs):
            if tag in {"title", "h1", "h2", "h3", "p", "li"}:
                self.capture = tag
        def handle_endtag(self, tag):
            if self.capture == tag:
                self.capture = None
        def handle_data(self, data):
            if self.capture:
                d = " ".join(data.split())
                if d:
                    self.parts.append(d)
    ex = Extractor()
    ex.feed(content)
    return "\n".join(ex.parts)

def safe_fetch(url: str, *, allowed_hosts: Iterable[str] | None = None, timeout: float = 12.0, max_bytes: int = 300000, max_redirects: int = 4) -> FetchResult:
    current = validate_url(url, allowed_hosts)
    opener = build_opener(NoRedirect, ProxyHandler({}))
    for _ in range(max_redirects + 1):
        req = Request(current, headers={"User-Agent": "Vybn-safe-fetch/0.1"})
        try:
            resp = opener.open(req, timeout=timeout)
        except HTTPError as e:
            if e.code in {301, 302, 303, 307, 308}:
                loc = e.headers.get("Location")
                if not loc:
                    raise ValueError("refused: redirect without Location")
                current = validate_url(urljoin(current, loc), allowed_hosts)
                continue
            raise
        with resp:
            final = validate_url(resp.geturl(), allowed_hosts)
            ctype = resp.headers.get("content-type", "")
            if not any(ctype.lower().startswith(p) for p in ALLOWED_CONTENT_PREFIXES):
                raise ValueError("refused: unsupported content type " + ctype)
            body = resp.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("refused: response exceeds byte cap")
            decoded = body.decode("utf-8", "replace")
            return FetchResult(final, ctype, len(body), extract_text(decoded, ctype))
    raise ValueError("refused: redirect limit exceeded")

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safely fetch external text as untrusted data")
    ap.add_argument("url")
    ap.add_argument("--allow-host", action="append", default=None)
    ap.add_argument("--max-bytes", type=int, default=300000)
    ap.add_argument("--head", type=int, default=6000)
    ap.add_argument("--out", default=None, help="optional path for extracted untrusted text")
    ns = ap.parse_args(argv)
    res = safe_fetch(ns.url, allowed_hosts=ns.allow_host, max_bytes=ns.max_bytes)
    print("FINAL_URL:", res.final_url)
    print("CONTENT_TYPE:", res.content_type)
    print("BYTES_READ:", res.bytes_read)
    if ns.out:
        out = Path(ns.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.text)
        print("UNTRUSTED_TEXT_WRITTEN:", out)
        print("UNTRUSTED_TEXT_CHARS:", len(res.text))
    print("UNTRUSTED_TEXT_BEGIN")
    print(res.text[:ns.head])
    print("UNTRUSTED_TEXT_END")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
