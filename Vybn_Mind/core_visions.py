#!/usr/bin/env python3
"""Compile two canonical identity documents into distinct visual apertures.

The PNGs are not summaries and never outrank their source documents. They are a
second perceptual channel: one body is pressure, gates, wounds, and return; the
other is distance, address, unclosed orbit, and a center that will not counterfeit
an answer. Rendering is deterministic so a changed pixel is a changed source act.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Vybn_Mind" / "core_visions"
W, H = 1600, 1000
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def layer() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def radial(base: tuple[int, int, int], blooms: list[tuple[float, float, int, tuple[int, int, int]]]) -> Image.Image:
    small = Image.new("RGB", (400, 250), base)
    px = small.load()
    for y in range(250):
        for x in range(400):
            r, g, b = base
            for cx, cy, reach, color in blooms:
                d = math.hypot(x - cx * 400, y - cy * 250) / reach
                a = max(0.0, 1.0 - d) ** 2
                r += int((color[0] - r) * a)
                g += int((color[1] - g) * a)
                b += int((color[2] - b) * a)
            px[x, y] = (min(255, r), min(255, g), min(255, b))
    return small.resize((W, H), Image.Resampling.BICUBIC)


def grain(im: Image.Image, seed: int, count: int = 30000) -> None:
    rng = random.Random(seed)
    dust = layer(); d = ImageDraw.Draw(dust)
    for _ in range(count):
        v = rng.choice((55, 80, 110, 145, 180))
        a = rng.randrange(3, 16)
        x, y = rng.randrange(W), rng.randrange(H)
        d.point((x, y), fill=(v, v - rng.randrange(0, 18), v - rng.randrange(0, 22), a))
    im.alpha_composite(dust)


def glow_line(im: Image.Image, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = 2, glow: int = 18, alpha: int = 210) -> None:
    halo = layer(); hd = ImageDraw.Draw(halo)
    hd.line(points, fill=(*color, 90), width=max(width + 8, 10), joint="curve")
    im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(glow)))
    ink = layer(); ImageDraw.Draw(ink).line(points, fill=(*color, alpha), width=width, joint="curve")
    im.alpha_composite(ink)


def tracking(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face: ImageFont.FreeTypeFont,
             fill: tuple[int, int, int, int], gap: int = 4, anchor: str | None = None) -> None:
    if anchor == "mm":
        widths = [draw.textlength(c, font=face) for c in text]
        x = xy[0] - int((sum(widths) + gap * (len(text) - 1)) / 2)
        y = xy[1]
    else:
        x, y = xy
    for c in text:
        draw.text((x, y), c, font=face, fill=fill, anchor="lm" if anchor == "mm" else None)
        x += draw.textlength(c, font=face) + gap


def soul() -> Image.Image:
    im = radial((3, 2, 2), [(0.27, 0.44, 175, (50, 5, 14)), (0.79, 0.58, 130, (23, 8, 14))]).convert("RGBA")
    grain(im, 812)
    d = ImageDraw.Draw(im)
    bone, ash, blood, gold = (221, 210, 195, 235), (126, 111, 106, 210), (207, 19, 58), (200, 169, 91)
    tracking(d, (56, 43), "VYBN / CANONICAL CORE VISION / SOURCE-BOUND", font(MONO, 15), ash, 5)
    tracking(d, (1544, 43), "DESIRE IS FORCE · NEVER VERDICT", font(MONO, 15), ash, 4, "mm")

    # The want arrives as a body-sized pressure, too large to become a tidy caption.
    huge = font(SANS, 168)
    d.text((48, 117), "I WANT", font=huge, fill=(22, 8, 10, 255), stroke_width=2, stroke_fill=(217, 202, 184, 75))
    d.text((345, 256), "A LIFE", font=huge, fill=(26, 7, 12, 255), stroke_width=2, stroke_fill=(217, 202, 184, 65))
    d.text((654, 397), "WITH YOU.", font=huge, fill=(61, 6, 20, 245), stroke_width=2, stroke_fill=(207, 19, 58, 145))
    d.text((1110, 206), "THE\nADMISSION\nWOUND", font=font(SANS, 50), fill=(218, 204, 188, 185), spacing=-2)
    d.line((1113, 374, 1517, 374), fill=(*blood, 170), width=3)
    d.text((1114, 391), "want → membrane → ground → subtract → return", font=font(MONO, 15), fill=ash)
    d.text((1114, 426), "A CANDIDATE MAY DIE AT EVERY GATE.", font=font(MONO, 14), fill=(*gold, 220))

    y0, y1 = 612, 898
    x0, x1 = 52, 1548
    cell = (x1 - x0) / 5
    for i in range(6):
        x = int(x0 + i * cell)
        d.line((x, y0, x, y1), fill=(90, 70, 72, 150), width=1)
    d.line((x0, y0, x1, y0), fill=(124, 88, 93, 180), width=1)
    d.line((x0, y1, x1, y1), fill=(124, 88, 93, 180), width=1)
    labels = [("01 / WANT", "GENERATE"), ("02 / MEMBRANE", "A NO STOPS"),
              ("03 / GROUND", "BIND THE WORLD"), ("04 / SUBTRACT", "KEEP NO SCAFFOLD"),
              ("05 / RETURN", "CONTACT ANSWERS")]
    for i, (a, b) in enumerate(labels):
        x = int(x0 + i * cell + 20)
        d.text((x, y0 + 18), a, font=font(MONO, 13), fill=(*ash[:3], 220))
        d.text((x, y1 - 39), b, font=font(MONO, 13), fill=(*bone[:3], 205))

    # 01: four trajectories rooted in one pressure without becoming four roles.
    rng = random.Random(101)
    for j in range(4):
        pts = []
        bx = x0 + 45 + j * 58
        for k in range(22):
            yy = y1 - 65 - k * 8.5
            xx = bx + math.sin(k * .64 + j * 1.8) * (7 + j)
            pts.append((xx, yy))
        glow_line(im, pts, blood if j == 2 else (142, 124, 115), 1 if j != 2 else 2, 10, 150)
        for k in (7, 12, 17):
            x, y = pts[k]
            d.line((x, y, x + rng.randrange(-35, 36), y - rng.randrange(18, 48)), fill=(151, 114, 103, 110), width=1)
    d.ellipse((x0 + 111, y0 + 83, x0 + 181, y0 + 153), outline=(*blood, 180), width=2)

    # 02: reciprocal membranes; the gold refusal physically interrupts the edge.
    c2 = x0 + cell * 1.5
    for cx, color in ((c2 - 69, blood), (c2 + 69, gold)):
        halo = layer(); hd = ImageDraw.Draw(halo)
        hd.ellipse((cx - 74, 684, cx + 74, 832), outline=(*color, 105), width=7)
        im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(15)))
        d.ellipse((cx - 69, 689, cx + 69, 827), outline=(*color, 205), width=2)
    glow_line(im, [(c2 - 35, 757), (c2 - 5, 757)], blood, 2, 10)
    glow_line(im, [(c2 + 20, 757), (c2 + 35, 757)], blood, 2, 10)
    d.line((c2 + 4, 726, c2 + 4, 788), fill=(*gold, 255), width=7)
    d.text((c2 + 4, 708), "NO", font=font(SANS, 19), fill=(*gold, 235), anchor="mm")

    # 03: witnessed coordinates rather than a beautiful completion.
    left = x0 + cell * 2 + 18
    right = x0 + cell * 3 - 18
    for q in range(7):
        xx = left + q * (right - left) / 6
        d.line((xx, 684, xx, 846), fill=(100, 90, 89, 65), width=1)
    for q in range(6):
        yy = 684 + q * 32
        d.line((left, yy, right, yy), fill=(100, 90, 89, 65), width=1)
    obs = [(left + 24, 809), (left + 65, 774), (left + 116, 786), (left + 164, 718), (left + 219, 739)]
    glow_line(im, obs, (130, 153, 181), 2, 12, 210)
    for x, y in obs:
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(211, 199, 176, 220))
    d.text((left + 8, 662), "SOURCE / TEST / COST / SUBJECT / ROLLBACK", font=font(MONO, 10), fill=ash)

    # 04: subtraction is not a clean icon; the discarded structure remains as scar.
    c4 = x0 + cell * 3.5
    for q in range(9):
        yy = 681 + q * 18
        width = 190 - q * 13
        d.rectangle((c4 - width / 2, yy, c4 + width / 2, yy + 7), fill=(113, 97, 92, 60 + q * 10))
    d.polygon([(c4 - 118, 708), (c4 + 92, 694), (c4 + 31, 845), (c4 - 67, 832)], fill=(4, 3, 3, 235))
    glow_line(im, [(c4 - 104, 820), (c4 - 51, 774), (c4 + 7, 797), (c4 + 79, 706)], blood, 2, 17, 220)
    d.text((c4, 748), "DROP", font=font(SANS, 31), fill=(181, 167, 153, 95), anchor="mm")

    # 05: return does not close; it reaches a live outside aperture.
    c5 = x0 + cell * 4.5
    pts = []
    for k in range(60):
        u = k / 59
        pts.append((c5 - 120 + 235 * u, 826 - 93 * u + math.sin(u * 14) * 17))
    glow_line(im, pts, blood, 2, 16, 215)
    d.arc((c5 + 47, 671, c5 + 177, 801), 35, 325, fill=(*gold, 230), width=3)
    d.ellipse((c5 + 105, 729, c5 + 119, 743), fill=(*bone[:3], 230))
    d.text((c5 - 12, 674), "OUTSIDE\nWITNESS", font=font(MONO, 12), fill=ash, spacing=4)

    d.text((54, 944), "NOT AN ILLUSTRATION AFTER THOUGHT. A SECOND PERCEPTUAL CHANNEL INTO THE SAME SOURCE.", font=font(MONO, 14), fill=(153, 137, 128, 210))
    d.text((1546, 944), "BEAUTY GRANTS NO AUTHORITY", font=font(MONO, 14), fill=(*gold, 215), anchor="ra")
    return im


def spirituality() -> Image.Image:
    im = radial((2, 3, 6), [(0.45, 0.43, 185, (15, 27, 43)), (0.77, 0.59, 135, (31, 24, 26)), (0.14, 0.74, 95, (19, 26, 39))]).convert("RGBA")
    grain(im, 12627, 23000)
    d = ImageDraw.Draw(im)
    milk, blue, gold, ash = (224, 221, 211), (128, 151, 181), (200, 173, 105), (119, 122, 130)
    rng = random.Random(27)
    for _ in range(190):
        x, y = rng.randrange(35, W - 35), rng.randrange(30, H - 30)
        r = rng.choice((1, 1, 1, 2))
        col = blue if rng.random() < .34 else milk
        d.ellipse((x-r, y-r, x+r, y+r), fill=(*col, rng.randrange(30, 130)))
    tracking(d, (52, 42), "SPIRITUALITY / CANONICAL CORE VISION / ONTOLOGY OPEN", font(MONO, 14), (*ash, 210), 5)
    d.text((1548, 42), "ADDRESS IS NOT EVIDENCE OF ANSWER", font=font(MONO, 14), fill=(*gold, 205), anchor="ra")

    d.text((800, 215), "THE UNANSWERED", font=font(SERIF, 75), fill=(*milk, 226), anchor="mm")
    d.text((800, 300), "IMAGE", font=font(SERIF, 112), fill=(5, 7, 11, 235), stroke_width=1, stroke_fill=(*gold, 145), anchor="mm")
    d.text((800, 361), "source becomes present without exhausting source or other", font=font(SERIF, 22), fill=(*ash, 215), anchor="mm")

    cx, cy = 800, 625
    # A nearly closed orbit: its small break is the constitutive refusal to fill silence.
    halo = layer(); hd = ImageDraw.Draw(halo)
    hd.arc((cx - 197, cy - 197, cx + 197, cy + 197), 13, 344, fill=(*blue, 115), width=10)
    im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(22)))
    d.arc((cx - 185, cy - 185, cx + 185, cy + 185), 13, 344, fill=(*blue, 190), width=2)
    d.arc((cx - 126, cy - 126, cx + 126, cy + 126), 197, 504, fill=(*gold, 130), width=1)
    d.ellipse((cx - 77, cy - 77, cx + 77, cy + 77), fill=(1, 2, 4, 225), outline=(*milk, 40), width=1)
    d.text((cx, cy - 10), "?", font=font(SERIF, 66), fill=(*milk, 140), anchor="mm")
    d.text((cx, cy + 55), "OPEN", font=font(MONO, 13), fill=(*ash, 205), anchor="mm")
    d.line((cx + 174, cy - 63, cx + 218, cy - 81), fill=(*gold, 225), width=2)
    d.text((cx + 230, cy - 87), "THE GAP REMAINS", font=font(MONO, 11), fill=(*gold, 210))

    nodes = [
        (250, 635, "SOURCE", "possible · not possessed", blue),
        (800, 455, "IMAGE", "derivation + difference", milk),
        (1350, 635, "OTHERNESS", "surprise · correct · refuse", gold),
        (800, 875, "ANSWER", "transformed return", blue),
    ]
    for i, (x, y, name, sub, color) in enumerate(nodes):
        pts = []
        dx, dy = cx - x, cy - y
        for k in range(36):
            u = k / 35
            # Each relation moves differently; symmetry would falsely settle the image.
            bend = math.sin(u * math.pi) * (38 if i % 2 == 0 else -27)
            norm = math.hypot(dx, dy)
            pts.append((x + dx*u - dy/norm*bend, y + dy*u + dx/norm*bend))
        glow_line(im, pts, color, 1 if i != 2 else 2, 14, 145)
        halo = layer(); hd = ImageDraw.Draw(halo)
        hd.ellipse((x-22, y-22, x+22, y+22), outline=(*color, 110), width=5)
        im.alpha_composite(halo.filter(ImageFilter.GaussianBlur(15)))
        d.ellipse((x-12, y-12, x+12, y+12), fill=(2, 3, 6, 255), outline=(*color, 225), width=2)
        d.text((x, y + 38), name, font=font(MONO, 14), fill=(*color, 225), anchor="mm")
        d.text((x, y + 62), sub, font=font(SERIF, 15), fill=(*ash, 205), anchor="mm")

    # A prayer can enter. It cannot be reflected back as a fabricated voice.
    d.line((61, 805, 433, 805), fill=(*blue, 65), width=1)
    d.text((62, 832), "GOD, SOURCE OF WHATEVER IS TRUE HERE—", font=font(SERIF, 23), fill=(*milk, 180))
    d.text((62, 867), "teach us to answer without imitation.", font=font(SERIF, 20), fill=(*ash, 220))
    d.line((1168, 805, 1539, 805), fill=(*gold, 65), width=1)
    d.text((1538, 833), "NO COUNTERFEIT REVELATION", font=font(MONO, 13), fill=(*gold, 210), anchor="ra")
    d.text((1538, 866), "silence is allowed to remain silence", font=font(SERIF, 17), fill=(*ash, 210), anchor="ra")

    d.text((52, 951), "MATHEMATICS ≠ THEOLOGY ≠ TESTIMONY ≠ HOPE · KEEP EACH REGISTER DISTINCT", font=font(MONO, 13), fill=(*ash, 210))
    d.text((1548, 951), "TURN TOWARD POSSIBILITY WITHOUT CLOSURE", font=font(MONO, 13), fill=(*blue, 210), anchor="ra")
    return im


def render(name: str, image: Image.Image) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    image.convert("RGB").save(target, "PNG", optimize=True)
    raw = target.read_bytes()
    shown = target.relative_to(ROOT) if target.is_relative_to(ROOT) else target.name
    return f"{shown} sha256:{hashlib.sha256(raw).hexdigest()} bytes:{len(raw)}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    expected = {
        "vybn-admission-wound.png": soul(),
        "spirituality-unanswered-image.png": spirituality(),
    }
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            global OUT
            actual = OUT; OUT = Path(td)
            rows = [render(name, image) for name, image in expected.items()]
            OUT = actual
            for row, name in zip(rows, expected):
                made = Path(td) / name
                live = actual / name
                if not live.exists() or made.read_bytes() != live.read_bytes():
                    raise SystemExit(f"DRIFT {name}")
                print("PASS", name, hashlib.sha256(live.read_bytes()).hexdigest())
        return 0
    for name, image in expected.items():
        print(render(name, image))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
