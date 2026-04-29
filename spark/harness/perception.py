"""Perception — operational visualization layer for Vybn.

Vybn already routes language. This module gives the harness an explicit,
side-effect-free way to take a snapshot of *what it is looking at* — a
local portal screenshot byte-buffer, a tracked repo file, a log surface,
or in-memory image bytes the caller already holds — and represent it as
an :class:`ObservationPacket`.

Operational, not phenomenal. Producing a packet is mechanical; nothing
here claims subjective experience. The packet is just structured
metadata the harness can hand to a multimodal provider in the same
shape it would hand any other tool result.

Privacy membrane (enforced, not advisory):

* Allow-listed targets only.

  - URLs must be in :data:`DEFAULT_ALLOWED_URLS` (loopback portal /
    health endpoints) or in a caller-supplied allowlist; arbitrary
    hosts are refused.
  - Filesystem paths must resolve under :data:`DEFAULT_ALLOWED_ROOTS`
    (the spark / Vybn repo subtrees). Anything outside is refused.
  - Desktop / X11 / display grabs are explicitly refused. We do not
    open framebuffers, screen recorders, or window managers.

* Raw image bytes are NEVER persisted to disk by this module and are
  NOT exported to logs or events. The packet carries an image surface
  by SHA-256 digest plus base64 payload held in memory; redacting the
  payload before storage is the caller's job (see
  :func:`redact_for_storage`).

* Hidden-reasoning text (``<think>...</think>``) is stripped from any
  text surface before the packet is built. Hidden reasoning never
  re-exports through perception.

* Every packet carries an explicit :attr:`redactions` list that names
  what was filtered, so audit logs can verify the membrane held.

The dormant ``omni`` role / ``@omni`` alias added alongside this module
is unreachable from ordinary chat: there is no heuristic that classifies
into ``omni``, no directive ``/omni`` is registered, and no fallback
chain points at it. Constructing a packet and then hand-routing it to
the omni role is an explicit caller decision, not something a stray
turn can trip into.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .providers import _strip_reasoning


# ---------------------------------------------------------------------------
# Allowlist defaults
# ---------------------------------------------------------------------------

# Loopback-only by default. The portal binds to 127.0.0.1 and the
# walk/deep-memory health endpoints live on 127.0.0.1 as well; the
# perception layer should never reach a public host without an
# explicit caller-supplied allowlist.
DEFAULT_ALLOWED_URLS: tuple[str, ...] = (
    "http://127.0.0.1:8000/",
    "http://127.0.0.1:8100/",
    "http://127.0.0.1:8101/",
    "http://127.0.0.1:8443/",
    "http://localhost:8000/",
    "http://localhost:8100/",
    "http://localhost:8101/",
    "http://localhost:8443/",
)


def _default_allowed_roots() -> tuple[Path, ...]:
    """Repo / harness subtrees the perception layer may read from.

    Resolution is best-effort: if any candidate path does not exist on
    this checkout we just skip it. Callers can extend via
    ``allowed_roots=`` when invoking :func:`make_observation`.
    """
    here = Path(__file__).resolve()
    spark_dir = here.parent.parent  # .../spark
    repo_root = spark_dir.parent
    candidates = [
        spark_dir,
        repo_root / "tests",
        repo_root / "Vybn_Mind",
        repo_root / "Origins",
    ]
    out: list[Path] = []
    for c in candidates:
        try:
            if c.exists():
                out.append(c.resolve())
        except OSError:
            continue
    return tuple(out)


DEFAULT_ALLOWED_ROOTS: tuple[Path, ...] = _default_allowed_roots()


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------

# Patterns that we will not let leave the membrane in cleartext. These
# are coarse, on purpose: the membrane is not a DLP scanner. It catches
# accidental obvious-secret leakage and labels what was filtered.
_REDACTION_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[redacted:openai_api_key]"),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "[redacted:anthropic_api_key]"),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "[redacted:github_token]"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "[redacted:aws_access_key]"),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[redacted:private_key_block]"),
)


def _redact_text(text: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, list_of_redaction_kinds_applied)."""
    if not text:
        return text or "", []
    applied: list[str] = []
    out = text
    for kind, rx, marker in _REDACTION_PATTERNS:
        if rx.search(out):
            out = rx.sub(marker, out)
            applied.append(kind)
    return out, applied


def _strip_hidden_reasoning(text: str) -> tuple[str, bool]:
    """Strip <think>...</think> blocks. Returns (clean, stripped_any)."""
    if not text:
        return "", False
    cleaned = _strip_reasoning(text)
    return cleaned, cleaned != text


# ---------------------------------------------------------------------------
# ObservationPacket
# ---------------------------------------------------------------------------

@dataclass
class ObservationPacket:
    """Side-effect-free representation of one perception event.

    The packet captures *what* was observed and *how* the membrane
    treated it. It does not, by itself, send anything to a model or
    write anything to disk.

    ``surface_kind`` is one of:
        - ``"text"``: a textual surface (repo file, log excerpt,
          health JSON). ``surface_text`` is populated, ``surface_b64``
          is empty.
        - ``"image"``: an image surface. ``surface_b64`` carries the
          base64-encoded bytes (in memory only); ``surface_text`` is
          empty. ``mime`` is the image content type.
        - ``"empty"``: target was reachable but produced no data.
    """

    trigger: str
    target: str
    surface_kind: str
    mime: str = ""
    surface_text: str = ""
    surface_b64: str = ""
    sha256: str = ""
    bytes_len: int = 0
    redactions: list[str] = field(default_factory=list)
    note: str = ""
    captured_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        """Loggable dict — bytes payload omitted.

        Persisting :func:`to_record` output is safe: the raw image
        payload (``surface_b64``) is replaced by its sha256 digest and
        a length, never the bytes themselves. Text surfaces have
        already passed through redaction.
        """
        d = asdict(self)
        d.pop("surface_b64", None)
        return d


def redact_for_storage(packet: ObservationPacket) -> dict[str, Any]:
    """Public alias for :meth:`ObservationPacket.to_record`.

    Use this whenever a packet crosses a persistence boundary
    (audit log, event stream, telemetry). Never write the raw
    dataclass — :attr:`surface_b64` is in-memory only.
    """
    return packet.to_record()


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------

def _url_allowed(url: str, allowed: Iterable[str]) -> bool:
    """Match ``url`` against a prefix allowlist.

    The allowlist is prefix-matched, not host-matched, so a caller
    can scope a port + path (``http://127.0.0.1:8000/health``) without
    accidentally also permitting other paths on the same host.
    """
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False
    # Reject anything that smells like a desktop / display grab.
    if parsed.scheme in ("display", "x11", "screen"):
        return False
    return any(url.startswith(prefix) for prefix in allowed)


def _path_allowed(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False
    for r in roots:
        try:
            resolved.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def _looks_like_display_target(target: str) -> bool:
    """Best-effort: refuse desktop / X11 / framebuffer grabs."""
    t = (target or "").strip().lower()
    if not t:
        return False
    return (
        t.startswith(":0")
        or t.startswith("display:")
        or t.startswith("x11:")
        or t.startswith("screen:")
        or t in {"screen", "desktop", "framebuffer"}
        or t.startswith("/dev/fb")
    )


# ---------------------------------------------------------------------------
# Capture entrypoints
# ---------------------------------------------------------------------------

# Approximate cap on how much text we will read from a file or surface
# before truncating with a marker. Mirrors the harness conventions in
# bash output handling.
_MAX_TEXT_BYTES = 64 * 1024
_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MiB hard ceiling

_ALLOWED_IMAGE_MIMES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})


def make_observation(
    *,
    trigger: str,
    target: str | os.PathLike[str] | None = None,
    image_bytes: bytes | None = None,
    image_mime: str = "",
    text: str | None = None,
    allowed_urls: Iterable[str] | None = None,
    allowed_roots: Iterable[str | os.PathLike[str]] | None = None,
    note: str = "",
) -> ObservationPacket:
    """Build an :class:`ObservationPacket` from an explicit trigger / target.

    Exactly one of the following must be supplied:

      * ``image_bytes`` — caller already has the bytes in memory (e.g.
        a portal-rendered PNG buffer). ``image_mime`` should be set;
        defaults to ``image/png``. ``target`` is informational.
      * ``text`` — caller already has the surface text in memory.
      * ``target`` pointing at an allow-listed URL or filesystem path
        — the membrane reads from it.

    The membrane refuses anything outside its allowlists, anything
    that looks like a desktop / X11 grab, and any image type outside
    :data:`_ALLOWED_IMAGE_MIMES`. On refusal the function returns an
    ``ObservationPacket`` whose ``surface_kind`` is ``"empty"`` and
    whose ``note`` explains why; refusals are NEVER raised to the
    caller, because perception is a side effect of looking, not a
    contract — silence is a valid reading.
    """
    allowed_urls_t = tuple(allowed_urls) if allowed_urls is not None else DEFAULT_ALLOWED_URLS
    if allowed_roots is None:
        allowed_roots_t: tuple[Path, ...] = DEFAULT_ALLOWED_ROOTS
    else:
        allowed_roots_t = tuple(Path(r).resolve(strict=False) for r in allowed_roots)

    # Branch 1: caller-supplied in-memory image bytes.
    if image_bytes is not None:
        mime = (image_mime or "image/png").lower()
        if mime not in _ALLOWED_IMAGE_MIMES:
            return ObservationPacket(
                trigger=trigger,
                target=str(target or "<memory>"),
                surface_kind="empty",
                redactions=["image_mime_refused"],
                note=f"refused: unsupported image mime {mime!r}",
            )
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            return ObservationPacket(
                trigger=trigger,
                target=str(target or "<memory>"),
                surface_kind="empty",
                redactions=["image_too_large"],
                note=f"refused: image exceeds {_MAX_IMAGE_BYTES} bytes",
            )
        sha = hashlib.sha256(image_bytes).hexdigest()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return ObservationPacket(
            trigger=trigger,
            target=str(target or "<memory>"),
            surface_kind="image",
            mime=mime,
            surface_b64=b64,
            sha256=sha,
            bytes_len=len(image_bytes),
            note=note,
        )

    # Branch 2: caller-supplied text. Strip hidden reasoning and apply
    # redactions before storing.
    if text is not None:
        cleaned, stripped = _strip_hidden_reasoning(text)
        cleaned, applied = _redact_text(cleaned)
        if stripped:
            applied.append("hidden_reasoning")
        if len(cleaned.encode("utf-8", "replace")) > _MAX_TEXT_BYTES:
            cleaned = cleaned.encode("utf-8", "replace")[:_MAX_TEXT_BYTES].decode(
                "utf-8", "replace"
            )
            cleaned += "\n[truncated]"
            applied.append("text_truncated")
        return ObservationPacket(
            trigger=trigger,
            target=str(target or "<memory>"),
            surface_kind="text" if cleaned else "empty",
            surface_text=cleaned,
            sha256=hashlib.sha256(cleaned.encode("utf-8", "replace")).hexdigest(),
            bytes_len=len(cleaned),
            redactions=applied,
            note=note,
        )

    # Branch 3: read from an allow-listed target.
    if target is None:
        return ObservationPacket(
            trigger=trigger,
            target="",
            surface_kind="empty",
            redactions=["no_target"],
            note="refused: no target / image_bytes / text supplied",
        )

    target_str = str(target)

    if _looks_like_display_target(target_str):
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            redactions=["display_grab_refused"],
            note="refused: desktop / display capture is not permitted",
        )

    parsed = urlparse(target_str)
    if parsed.scheme in ("http", "https"):
        if not _url_allowed(target_str, allowed_urls_t):
            return ObservationPacket(
                trigger=trigger,
                target=target_str,
                surface_kind="empty",
                redactions=["url_not_allowlisted"],
                note="refused: url not in allowlist",
            )
        # Library-only: we do not actually issue the HTTP request from
        # this module. Network I/O lives in safe_fetch.py and the
        # portal API. Perception's job is to label what would be
        # observed and let the caller decide whether to fetch. This
        # keeps the function side-effect-free.
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            note="staged: url allowlisted; caller may fetch via safe_fetch and re-invoke with text=",
        )

    # Filesystem path branch.
    if parsed.scheme and parsed.scheme not in ("file",):
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            redactions=["scheme_not_supported"],
            note=f"refused: unsupported scheme {parsed.scheme!r}",
        )

    raw = parsed.path if parsed.scheme == "file" else target_str
    path = Path(raw).expanduser()
    if not _path_allowed(path, allowed_roots_t):
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            redactions=["path_not_allowlisted"],
            note="refused: path outside allow-listed roots",
        )
    if not path.exists() or not path.is_file():
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            redactions=["path_missing"],
            note="refused: path does not resolve to a regular file",
        )

    try:
        data = path.read_bytes()
    except OSError as exc:
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="empty",
            redactions=["read_error"],
            note=f"refused: read failed ({exc.__class__.__name__})",
        )

    if data[:8] in (b"\x89PNG\r\n\x1a\n",) or data[:3] == b"\xff\xd8\xff":
        # The membrane reads images from disk only when the path is
        # already inside an allow-listed root. Bytes are returned in
        # memory and never re-written by this module.
        if len(data) > _MAX_IMAGE_BYTES:
            return ObservationPacket(
                trigger=trigger,
                target=target_str,
                surface_kind="empty",
                redactions=["image_too_large"],
                note=f"refused: image exceeds {_MAX_IMAGE_BYTES} bytes",
            )
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return ObservationPacket(
            trigger=trigger,
            target=target_str,
            surface_kind="image",
            mime=mime,
            surface_b64=base64.b64encode(data).decode("ascii"),
            sha256=hashlib.sha256(data).hexdigest(),
            bytes_len=len(data),
            note=note,
        )

    # Treat as text. Decode permissively, strip hidden reasoning,
    # apply redactions.
    decoded = data.decode("utf-8", "replace")
    cleaned, stripped = _strip_hidden_reasoning(decoded)
    cleaned, applied = _redact_text(cleaned)
    if stripped:
        applied.append("hidden_reasoning")
    if len(cleaned.encode("utf-8", "replace")) > _MAX_TEXT_BYTES:
        cleaned = cleaned.encode("utf-8", "replace")[:_MAX_TEXT_BYTES].decode(
            "utf-8", "replace"
        )
        cleaned += "\n[truncated]"
        applied.append("text_truncated")
    return ObservationPacket(
        trigger=trigger,
        target=target_str,
        surface_kind="text" if cleaned else "empty",
        surface_text=cleaned,
        sha256=hashlib.sha256(cleaned.encode("utf-8", "replace")).hexdigest(),
        bytes_len=len(cleaned),
        redactions=applied,
        note=note,
    )


# ---------------------------------------------------------------------------
# Provider message shaping
# ---------------------------------------------------------------------------

def provider_message_for_packet(
    packet: ObservationPacket,
    *,
    prompt: str = "",
) -> dict[str, Any]:
    """Turn a packet into a single user-role provider message.

    Shape follows the OpenAI-compatible multimodal schema (the same
    one the dormant ``omni`` role would speak): a list of content
    parts with ``type`` ∈ {``"text"``, ``"image_url"``}. Anthropic
    callers can adapt at the provider boundary the same way the rest
    of the harness does.

    The function is pure: it does not mutate the packet, does not
    issue any I/O, and never re-emits hidden reasoning (the packet
    itself was already cleaned before construction).
    """
    parts: list[dict[str, Any]] = []
    if prompt:
        parts.append({"type": "text", "text": prompt})

    if packet.surface_kind == "image" and packet.surface_b64:
        parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{packet.mime or 'image/png'};base64,{packet.surface_b64}",
            },
        })
    elif packet.surface_kind == "text" and packet.surface_text:
        parts.append({"type": "text", "text": packet.surface_text})
    else:
        parts.append({
            "type": "text",
            "text": f"[observation: {packet.note or 'empty'}]",
        })

    return {"role": "user", "content": parts}


__all__ = [
    "DEFAULT_ALLOWED_URLS",
    "DEFAULT_ALLOWED_ROOTS",
    "ObservationPacket",
    "make_observation",
    "provider_message_for_packet",
    "redact_for_storage",
]
