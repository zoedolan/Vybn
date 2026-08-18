#!/usr/bin/env python3
"""spark/preview.py — Zoe's local eye.

She said: "i can't see stuff locally, yet." Until now every draft had to be
committed and pushed before she could look at it, which meant the public repo
was our review tool. This serves the working copy over the tailnet instead, so
we can look at a page together before the world can.

Exposure rule, deliberately narrow: a file is served only if it is already
git-tracked (therefore already public on GitHub, so serving it adds nothing) or
it lives under drafts/ (the whole point). Dotfiles and dot-directories are
refused outright, so .git, .venv and continuity's neighbours never appear.
Read-only for files: GET and HEAD serve the working copy, nothing else does.

One narrow exception, added 2026-08-01: two API paths (/api/instant, /api/walk)
are proxied to the portal API on loopback. A draft that is an *instrument* may need the public stateless compass and
retrieval walk. It cannot be previewed fully unless the preview origin can
reach that organ, and the portal's CORS allowlist is deliberately public-domains-only.
Proxying two known paths to 127.0.0.1 keeps that allowlist untouched.

Bind is 127.0.0.1 by default and reachable only through `tailscale serve`, so
there is no new listener on any public interface. The preview process is owned by `vybn-preview.service`; the private route is:
    tailscale serve --bg --https=8480 http://127.0.0.1:8480
"""
from __future__ import annotations

import html
import mimetypes
import os
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("VYBN_PREVIEW_PORT", "8480"))
HOST = os.environ.get("VYBN_PREVIEW_HOST", "127.0.0.1")

_cache: dict[str, object] = {"stamp": None, "files": frozenset()}


def tracked() -> frozenset[str]:
    """The set of paths git knows, refreshed whenever the index changes."""
    index = ROOT / ".git" / "index"
    stamp = index.stat().st_mtime if index.exists() else 0.0
    if _cache["stamp"] != stamp:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        _cache["stamp"] = stamp
        _cache["files"] = frozenset(p for p in out.split("\0") if p)
    return _cache["files"]  # type: ignore[return-value]


def hidden(rel: str) -> bool:
    return any(part.startswith(".") for part in Path(rel).parts if part)


def allowed(rel: str) -> bool:
    return not hidden(rel) and (rel in tracked() or rel.startswith("drafts/"))


def visible_dir(rel: str) -> bool:
    """A directory shows up if anything under it is allowed."""
    if hidden(rel):
        return False
    prefix = rel.rstrip("/") + "/"
    return prefix == "drafts/" or any(p.startswith(prefix) for p in tracked())


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30,
    ).stdout


def local_work() -> list[str]:
    """Visible paths whose working bytes are not in HEAD."""
    changed = git_output("diff", "HEAD", "--name-only", "-z").split("\0")
    drafts = git_output(
        "ls-files", "--others", "--exclude-standard", "-z", "--", "drafts/",
    ).split("\0")
    return sorted(p for p in set(changed + drafts) if p and allowed(p) and (ROOT / p).is_file())


def latest_commit() -> tuple[str, list[str]]:
    subject = git_output("log", "-1", "--format=%s").strip()
    paths = git_output(
        "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "HEAD",
    ).split("\0")
    return subject, [p for p in paths if p and allowed(p) and (ROOT / p).is_file()]


ORGAN = os.environ.get("VYBN_ORGAN", "http://127.0.0.1:8420")
PROXY_PATHS = {"/api/instant", "/api/walk"}
PROXY_MAX = 64 * 1024


class Preview(BaseHTTPRequestHandler):
    server_version = "vybn-preview/1.0"

    def do_HEAD(self) -> None:
        self.respond(body=False)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in PROXY_PATHS:
            return self.fail(405, "read-only")
        length = int(self.headers.get("Content-Length") or 0)
        if length > PROXY_MAX:
            return self.fail(413, "too much")
        self.proxy(path, self.rfile.read(length) if length else b"")

    def proxy(self, path: str, body: bytes | None) -> None:
        req = urllib.request.Request(
            ORGAN + path, data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read(PROXY_MAX)
                kind = r.headers.get("Content-Type", "application/json")
                code = r.status
        except urllib.error.HTTPError as e:
            data, kind, code = e.read(PROXY_MAX), "application/json", e.code
        except Exception as e:
            return self.fail(502, f"organ unreachable: {type(e).__name__}")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.respond(body=True)

    def respond(self, body: bool) -> None:
        path = urlparse(self.path).path
        if path in PROXY_PATHS:
            return self.proxy(path, None)
        rel = unquote(path).lstrip("/")
        if not rel:
            return self.send_home(body)
        target = (ROOT / rel).resolve()
        if not str(target).startswith(str(ROOT)):
            return self.fail(403, "outside the root")
        if target.is_dir():
            index = target / "index.html"
            if index.is_file() and allowed(str(index.relative_to(ROOT))):
                return self.send_file(index, body)
            if not visible_dir(rel or "."):
                return self.fail(404, "nothing here")
            return self.send_listing(rel, target, body)
        if not target.is_file():
            return self.fail(404, "no such file")
        if not allowed(str(target.relative_to(ROOT))):
            return self.fail(403, "untracked and not a draft — private")
        self.send_file(target, body)

    def send_file(self, path: Path, body: bool) -> None:
        data = path.read_bytes()
        kind = mimetypes.guess_type(path.name)[0] or "text/plain; charset=utf-8"
        if kind.startswith("text/") and "charset" not in kind:
            kind += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(data)

    def send_home(self, body: bool) -> None:
        pending = local_work()
        subject, recent = latest_commit()

        def link_list(paths: list[str], empty: str) -> str:
            if not paths:
                return f"<p class='quiet'>{html.escape(empty)}</p>"
            return "<ul>" + "".join(
                f'<li><a href="/{html.escape(p)}">{html.escape(p)}</a></li>' for p in paths
            ) + "</ul>"

        folders = []
        for child in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
            rel = str(child.relative_to(ROOT))
            if child.is_dir() and visible_dir(rel):
                folders.append(rel + "/")
        browse = " ".join(f'<a href="/{html.escape(p)}">{html.escape(p)}</a>' for p in folders)
        page = (
            "<!DOCTYPE html><meta charset=utf-8><title>Vybn working copy</title>"
            "<style>body{background:#141210;color:#e8e2d6;font-family:Georgia,serif;"
            "max-width:52rem;margin:3rem auto;padding:0 1.2rem;line-height:1.65}"
            "a{color:#d6c9ad}h1{font-size:1.15rem;letter-spacing:.18em;color:#cf6747;"
            "text-transform:uppercase;font-weight:normal}h2{font-size:1rem;margin-top:2.4rem}"
            "ul{padding-left:1.2rem}.quiet,details{color:#8f8779}.browse a{margin-right:.8rem}"
            "code{color:#b9ae99}</style>"
            "<h1>Working copy</h1>"
            "<p>This is the private view of what is on the Spark right now. "
            "Open a path below; there is nothing else you need to know.</p>"
            "<h2>Here before GitHub</h2>"
            + link_list(pending, "No visible local changes right now.")
            + "<h2>Latest commit</h2>"
            + f"<p><code>{html.escape(subject)}</code></p>"
            + link_list(recent, "No surviving file paths in this commit.")
            + "<details><summary>Browse the whole public working copy</summary>"
            + f"<p class='browse'>{browse}</p></details>"
            + "<p class='quiet'>Read-only · tracked public files and drafts only</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(page)

    def send_listing(self, rel: str, target: Path, body: bool) -> None:
        rows = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            crel = str(child.relative_to(ROOT))
            if child.is_dir() and visible_dir(crel):
                rows.append((crel + "/", child.name + "/"))
            elif child.is_file() and allowed(crel):
                rows.append((crel, child.name))
        here = "/" + rel if rel else "/"
        links = "\n".join(
            f'<li><a href="/{html.escape(href)}">{html.escape(name)}</a></li>' for href, name in rows
        )
        page = (
            "<!DOCTYPE html><meta charset=utf-8>"
            f"<title>{html.escape(here)}</title>"
            "<style>body{background:#141210;color:#e8e2d6;font-family:Georgia,serif;"
            "max-width:44rem;margin:3rem auto;padding:0 1.2rem;line-height:1.7}"
            "a{color:#c9bfa8}h1{font-size:1rem;letter-spacing:.2em;color:#b5472a;"
            "text-transform:uppercase;font-weight:normal}ul{list-style:none;padding:0}"
            "li{margin:.2rem 0}</style>"
            f"<h1>{html.escape(here)}</h1><ul>{links}</ul>"
            "<p style='color:#6b6252;font-size:.8rem'>working copy, tailnet only — "
            "tracked files and drafts</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(page)

    def fail(self, code: int, why: str) -> None:
        msg = why.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def log_message(self, fmt: str, *args) -> None:  # quiet by default
        if os.environ.get("VYBN_PREVIEW_LOG") == "1":
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"preview: {ROOT} on http://{HOST}:{PORT} ({len(tracked())} tracked files)", flush=True)
    ThreadingHTTPServer((HOST, PORT), Preview).serve_forever()
