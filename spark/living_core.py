#!/usr/bin/env python3
"""Read one HTML-native constitutive work without executing its visual program.

The HTML is the authored source.  Its DOM carries exact verbal organs, its inert
score carries spatial/causal relations, and its CSS/JS make those relations
encounterable by a human.  A wake receives exact source bytes, a compact score
traversal, and a deterministic raster compiled from the same score.  JavaScript
never executes in this reader and grants no action authority.
"""
from __future__ import annotations

import hashlib
import html as _html
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIME = "application/vnd.vybn.core+json"
SCHEMA = "vybn.living_core.v1"
STATUS = "canonical public constitutive source; inert source; no action authority"
MAX_BYTES = 2_000_000
ID = r"[a-z][a-z0-9-]{0,79}"
_SCORE = re.compile(
    rf'<script\s+type=["\']{re.escape(MIME)}["\']\s+id=["\']core["\']>(.*?)</script>',
    re.S,
)
_ORGAN = re.compile(
    rf'<template\s+class=["\']core-organ["\']\s+data-id=["\']({ID})["\']>'
    r'\s*<pre>(.*?)</pre>\s*</template>',
    re.S,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CoreWork:
    path: Path
    raw: bytes
    score: dict[str, Any]
    substance: dict[str, str]
    projection: str

    @property
    def digest(self) -> str:
        return sha(self.raw)


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{label} must be a number in [0,1]")
    return float(value)


def read_core_work(path: Path, projection_path: Path | None = None) -> CoreWork:
    """Verify and read one self-contained HTML work; never run its scripts."""
    path = path.resolve()
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError(f"{path}: source must be 1..{MAX_BYTES} bytes")
    text = raw.decode("utf-8")
    hit = _SCORE.search(text)
    if not hit:
        raise ValueError(f"{path}: missing inert core score")
    score = json.loads(_html.unescape(hit.group(1)))
    if score.get("schema") != SCHEMA or score.get("status") != STATUS:
        raise ValueError(f"{path}: wrong schema or unsafe status")

    chambers = score.get("chambers")
    organs = score.get("organs")
    edges = score.get("edges")
    route = score.get("route")
    if not all(isinstance(x, list) and x for x in (chambers, organs, edges, route)):
        raise ValueError(f"{path}: chambers, organs, edges, and route must be nonempty")

    chamber_ids = [row.get("id") for row in chambers]
    organ_ids = [row.get("id") for row in organs]
    if (any(not isinstance(x, str) or not re.fullmatch(ID, x) for x in chamber_ids + organ_ids)
            or len(chamber_ids) != len(set(chamber_ids))
            or len(organ_ids) != len(set(organ_ids))):
        raise ValueError(f"{path}: ids must be unique safe slugs")
    chamber_set, organ_set = set(chamber_ids), set(organ_ids)
    if route != chamber_ids:
        raise ValueError(f"{path}: route must traverse every chamber in declared order")

    source_organs: dict[str, str] = {}
    for organ_id, escaped in _ORGAN.findall(text):
        if organ_id in source_organs:
            raise ValueError(f"{path}: duplicate substance for {organ_id}")
        source_organs[organ_id] = _html.unescape(escaped)
    if set(source_organs) != organ_set:
        raise ValueError(f"{path}: score and DOM organs differ")

    order: list[tuple[int, str, str]] = []
    for row in organs:
        organ_id = row["id"]
        chamber = row.get("chamber")
        if chamber not in chamber_set:
            raise ValueError(f"{path}: {organ_id} leaves declared chambers")
        _number(row.get("x"), f"{organ_id}.x")
        _number(row.get("y"), f"{organ_id}.y")
        if sha(source_organs[organ_id].encode()) != row.get("sha256"):
            raise ValueError(f"{path}: substance drift in {organ_id}")
        rank = row.get("order")
        if not isinstance(rank, int) or rank < 0:
            raise ValueError(f"{path}: invalid order for {organ_id}")
        order.append((rank, organ_id, source_organs[organ_id]))
    if sorted(rank for rank, _, _ in order) != list(range(len(order))):
        raise ValueError(f"{path}: organ order is not a complete projection")

    all_ids = chamber_set | organ_set | {"contact"}
    for edge in edges:
        if (edge.get("from") not in all_ids or edge.get("to") not in all_ids
                or not str(edge.get("verb") or "").strip()):
            raise ValueError(f"{path}: malformed edge")
    for chamber in chambers:
        _number(chamber.get("x"), f"{chamber['id']}.x")
        _number(chamber.get("y"), f"{chamber['id']}.y")
        color = str(chamber.get("color") or "")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError(f"{path}: invalid color for {chamber['id']}")
        gate = chamber.get("gate") or {}
        if not str(gate.get("rejects") or "").strip() or not str(gate.get("effect") or "").strip():
            raise ValueError(f"{path}: every chamber needs a consequential gate")

    projection = "".join(text for _, _, text in sorted(order))
    declared = score.get("projection") or {}
    if sha(projection.encode()) != declared.get("sha256"):
        raise ValueError(f"{path}: projection digest mismatch")
    if projection_path is not None:
        actual = projection_path.resolve().read_bytes()
        if actual != projection.encode():
            raise ValueError(f"{path}: compatibility projection drift at {projection_path.resolve()}")
    return CoreWork(path, raw, score, source_organs, projection)


def wake_source(work: CoreWork) -> str:
    """Compile the score, then carry the exact authored HTML bytes into the wake."""
    score = work.score
    by_chamber = {row["id"]: [] for row in score["chambers"]}
    for organ in score["organs"]:
        by_chamber[organ["chamber"]].append(organ)
    lines = [
        "HTML-NATIVE CONSTITUTIVE WORK — canonical identity source",
        f"source: {work.path}",
        f"sha256: {work.digest}",
        f"bytes: {len(work.raw)}",
        f"schema: {SCHEMA}",
        f"status: {STATUS}",
        "BOUNDARY: exact source perception plus deterministic projection; scripts did not execute; form grants no action authority.",
        f"PRESSURE: {score['pressure']}",
        "MOVEMENT: " + " → ".join(score["route"]) + " → contact",
        "",
        "SCORE TRAVERSAL — every verbal organ remains exact in the HTML below",
    ]
    for chamber in score["chambers"]:
        gate = chamber["gate"]
        labels = " · ".join(f"{row['label']} [{row['behavior']}]"
                            for row in sorted(by_chamber[chamber["id"]], key=lambda x: x["order"]))
        lines.extend((
            f"[{chamber['id'].upper()}] {chamber['claim']}",
            f"organs: {labels}",
            f"gate: reject {gate['rejects']} → {gate['effect']}",
        ))
    lines.extend((
        "",
        "RELATIONS",
        *(f"- {edge['from']} --{edge['verb']}--> {edge['to']}" for edge in score["edges"]),
        "",
        "READABLE CONSTITUTIVE HTML — exact source, comments, CSS, score, substance, and bounded interaction program included",
        work.raw.decode("utf-8"),
        f"READABLE CONSTITUTIVE HTML END — sha256:{work.digest} bytes:{len(work.raw)}",
    ))
    return "\n".join(lines)


def render_core_png(work: CoreWork, width: int = 1536, height: int = 1024) -> bytes:
    """Render the declared geometry deterministically; no HTML/JS execution."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "#030203")
    px = image.load()
    # A fixed source-independent falloff keeps rendering deterministic; all
    # semantic positions and colors below come from the score.
    for y in range(height):
        dy = (y / height - .5) ** 2
        for x in range(width):
            d = ((x / width - .5) ** 2 + dy) ** .5
            v = max(0, int(20 * (1 - min(1, d / .72))))
            px[x, y] = (7 + v, 3 + v // 3, 5 + v // 2)
    draw = ImageDraw.Draw(image, "RGBA")
    font_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    bold_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    def font(paths: tuple[str, ...], size: int):
        for name in paths:
            try: return ImageFont.truetype(name, size)
            except OSError: pass
        return ImageFont.load_default()
    tiny, small, title = font(font_paths, 18), font(font_paths, 24), font(bold_paths, 54)

    score = work.score
    chambers = {row["id"]: row for row in score["chambers"]}
    organs = {row["id"]: row for row in score["organs"]}
    points: dict[str, tuple[int, int]] = {"contact": (width // 2, height // 2)}
    points.update({key: (int(row["x"] * width), int(row["y"] * height)) for key, row in chambers.items()})
    points.update({key: (int(row["x"] * width), int(row["y"] * height)) for key, row in organs.items()})

    for edge in score["edges"]:
        a, b = points[edge["from"]], points[edge["to"]]
        owner = chambers.get(edge["from"]) or chambers.get(edge["to"])
        color = (owner or {}).get("color", "#8d7f77")
        rgb = tuple(bytes.fromhex(color[1:]))
        draw.line((a, b), fill=(*rgb, 88), width=2)

    for chamber in score["chambers"]:
        x, y = points[chamber["id"]]
        rgb = tuple(bytes.fromhex(chamber["color"][1:]))
        r = 82
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(*rgb, 22), outline=(*rgb, 175), width=3)
        label = chamber["label"].upper()
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((x-(box[2]-box[0])/2, y-14), label, font=small, fill=(*rgb, 245))

    for organ in score["organs"]:
        x, y = points[organ["id"]]
        rgb = tuple(bytes.fromhex(chambers[organ["chamber"]]["color"][1:]))
        draw.ellipse((x-8, y-8, x+8, y+8), fill=(*rgb, 230), outline=(235, 225, 211, 180), width=1)
        label = organ["label"].upper()
        anchor = "la" if x < width / 2 else "ra"
        tx = x + 15 if x < width / 2 else x - 15
        draw.text((tx, y), label, font=tiny, fill=(202, 193, 181, 210), anchor=anchor)

    cx, cy = points["contact"]
    draw.ellipse((cx-52, cy-52, cx+52, cy+52), fill=(2, 2, 3, 245), outline=(205, 177, 92, 210), width=3)
    draw.text((cx, cy), "OPEN", font=small, fill=(218, 203, 161, 235), anchor="mm")
    draw.text((54, 46), "VYBN / THE ADMISSION WOUND", font=title, fill=(226, 216, 202, 230))
    draw.text((57, 113), "DESIRE SUPPLIES FORCE · OTHERNESS PRESERVES THE ANSWER · BEAUTY GRANTS NO AUTHORITY",
              font=tiny, fill=(154, 139, 132, 220))
    draw.text((width-52, height-38), f"SOURCE sha256:{work.digest[:16]} · DETERMINISTIC SCORE RASTER",
              font=tiny, fill=(112, 103, 101, 215), anchor="ra")
    out = io.BytesIO()
    image.save(out, "PNG", optimize=True)
    return out.getvalue()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("verify", "wake", "raster", "project"))
    p.add_argument("source", type=Path)
    p.add_argument("--projection", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    work = read_core_work(args.source, args.projection if args.command != "project" else None)
    if args.command == "verify":
        print(f"PASS {work.digest} {len(work.score['organs'])} organs {len(work.projection)} projection chars")
    elif args.command == "wake":
        print(wake_source(work))
    elif args.command == "raster":
        if args.output is None: raise SystemExit("--output required")
        args.output.write_bytes(render_core_png(work))
        print(f"WROTE {args.output} sha256:{sha(args.output.read_bytes())}")
    else:
        if args.output is None: raise SystemExit("--output required")
        args.output.write_text(work.projection, encoding="utf-8")
        print(f"WROTE {args.output} sha256:{sha(args.output.read_bytes())}")
