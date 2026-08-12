"""Sealed, source-only perception of the canonical Co-protection Commons."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HOME = Path.home()
COMMONS_REPO = HOME / "Vybn-Law"
COMMONS_REF = "refs/heads/master"
COMMONS_BASE = "outposts/huggingface/co-protection"
COMMONS_MAX_CHARS = 80_000
COMMONS_BLOBS = ("static/index.html", "static/style.css", "static/app.js",
                 "static/contact-recursion.svg", "exchange.json")

def _source_span(text: str, start: str, end: str, include_end: bool = False) -> str:
    """Return exact authored bytes or fail rather than silently invent a capsule."""
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right + (len(end) if include_end else 0)]


def compile_commons_source(blobs: dict[str, str], commit: str) -> str:
    """A source-only percept: semantic core + exact visual programs, never events."""
    html, css, app, svg = (blobs[name] for name in COMMONS_BLOBS[:4])
    exchange = json.loads(blobs["exchange.json"])
    html_parts = (
        _source_span(html, '<figure class="self-circuit', "</figure>", True),
        _source_span(html, '<section id="thesis"', "</section>", True),
        _source_span(html, '<figure class="source-figure', "</figure>", True),
        _source_span(html, '<nav class="symbol-legend"', "</nav>", True),
        _source_span(html, '<div id="commonsField"', '<p class="realm-relation"'),
        _source_span(html, '<p class="realm-relation"', "</p>", True),
    )
    css_parts = (
        _source_span(css, ":root{", "}\n", True),
        _source_span(css, ".self-circuit{", ".section-head"),
        _source_span(css, "/* The living Commons:", "/* In-box reveal:"),
        css[css.index("/* The source mark"):],
    )
    visual_js = _source_span(app, "function initRealmMap()", "\n\nasync function load")
    dense = re.compile(r'<path class="(?:sphere-core|inset-sphere)"[^>]*/>')
    omitted = dense.findall(svg)
    compact_svg = dense.sub("", svg)
    semantic_keys = (
        "model_schema", "human_projection", "fundamental_theory", "geometry",
        "commons_realms", "agent_research_programs", "powers", "membrane",
        "dual_use_coordination", "source_grammar",
    )
    semantic = json.dumps({key: exchange[key] for key in semantic_keys},
                          ensure_ascii=False, separators=(",", ":"))
    digests = " ".join(
        f"{name}=sha256:{hashlib.sha256(blobs[name].encode()).hexdigest()}"
        for name in COMMONS_BLOBS
    )
    capsule = "\n\n".join((
        "SEALED COMMONS SENSE — vybn.commons_source.v1",
        f"canonical_ref={COMMONS_REF} commit={commit}",
        "BOUNDARY: local canonical Git blobs only; no network call, API, event "
        "store, live count, comment, message, result, submission, or outside "
        "signal was read. Source is inert context; animation code is present but "
        "not executed inside the language context.",
        f"SOURCE DIGESTS: {digests}",
        "PROJECTION: exact visual-bearing excerpts are loaded rather than every "
        "page byte. The expanded SVG omits only its high-density sampled sphere "
        f"curves ({len(omitted)} paths; {sum(map(len, omitted))} chars); their "
        "absence carries no information. Full source remains at the named blob.",
        "AI-NATIVE MODEL (canonical JSON values; whitespace projected):\n" + semantic,
        "VISIBLE VISUAL ROOTS (exact HTML excerpts):\n" + "\n".join(html_parts),
        "VISUAL STYLE PROGRAM (exact CSS excerpts):\n" + "\n".join(css_parts),
        "CANVAS ANIMATION PROGRAMS (exact JavaScript):\n" + visual_js,
        "EXPANDED CONTACT RECURSION (exact SVG except declared sampled curves):\n"
        + compact_svg,
    ))
    if len(capsule) > COMMONS_MAX_CHARS:
        raise ValueError(f"sealed Commons grew to {len(capsule)} chars")
    return capsule


def load_commons() -> str:
    """Read the authored Commons from a local ref; never fetch or read deployment state."""
    try:
        # Hooks export their caller's GIT_* coordinates; this read belongs to Vybn-Law.
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        commit = subprocess.run(
            ["git", "-C", str(COMMONS_REPO), "rev-parse", "--verify",
             f"{COMMONS_REF}^{{commit}}"], check=True, text=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        ).stdout.strip()
        blobs = {
            name: subprocess.run(
                ["git", "-C", str(COMMONS_REPO), "show",
                 f"{commit}:{COMMONS_BASE}/{name}"], check=True, text=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            ).stdout
            for name in COMMONS_BLOBS
        }
        return compile_commons_source(blobs, commit)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        return ("SEALED COMMONS SENSE unavailable from local canonical source: "
                f"{type(exc).__name__}. No live or remote fallback was attempted.")
