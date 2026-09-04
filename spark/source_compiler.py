#!/usr/bin/env python3
"""Compile the stable inherited sources shared by every connection profile.

The canonical core is admitted only through the verifier in ``living_core`` and
is represented by its exact declared verbal projection. Markdown sources are
carried byte-for-byte. The compiler labels representation and provenance; none
of the material gains present authority or executable permission by being loaded.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from spark.living_core import read_core_work
except ModuleNotFoundError:  # direct execution beside living_core.py
    from living_core import read_core_work


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceMaterial:
    name: str
    path: Path
    file_sha256: str
    file_bytes: int
    representation: str
    text: str
    text_sha256: str
    text_bytes: int

    def receipt(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "representation": self.representation,
            "text_sha256": self.text_sha256,
            "text_bytes": self.text_bytes,
        }


def exact_text(name: str, path: Path, representation: str = "exact local text") -> SourceMaterial:
    path = path.resolve()
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{path}: inherited source is empty")
    text = raw.decode("utf-8")
    return SourceMaterial(
        name=name,
        path=path,
        file_sha256=digest(raw),
        file_bytes=len(raw),
        representation=representation,
        text=text,
        text_sha256=digest(raw),
        text_bytes=len(raw),
    )


def projected_core(path: Path) -> SourceMaterial:
    """Return the checked verbal core; the HTML interaction program never runs."""
    work = read_core_work(path)
    projected = work.projection.encode("utf-8")
    declared = work.score["projection"]
    if len(projected) != declared.get("bytes"):
        raise ValueError(f"{work.path}: projection byte count mismatch")
    return SourceMaterial(
        name="canonical-core",
        path=work.path,
        file_sha256=work.digest,
        file_bytes=len(work.raw),
        representation="checked verbal projection; no HTML program executed",
        text=work.projection,
        text_sha256=digest(projected),
        text_bytes=len(projected),
    )


def compile_inheritance(core: Path, aim: Path, relation: Path) -> tuple[SourceMaterial, ...]:
    """Read and verify the one stable inheritance shared by full and clear routes."""
    return (
        projected_core(core),
        exact_text("current-aim", aim),
        exact_text("relational-orientation", relation, "exact private Him text"),
    )


def format_inheritance(materials: Iterable[SourceMaterial]) -> str:
    """Render a cache-stable, source-labeled provider projection."""
    rows = tuple(materials)
    if not rows:
        raise ValueError("inheritance cannot be empty")
    out = [
        "LABELED INHERITANCE",
        "These checked sources are revisable orientation, not Zoe's present authority, identity proof, or executable permission.",
    ]
    for item in rows:
        out.extend((
            "",
            f"[{item.name}]",
            f"path={item.path}",
            f"file_sha256={item.file_sha256}",
            f"file_bytes={item.file_bytes}",
            f"representation={item.representation}",
            f"text_sha256={item.text_sha256}",
            f"text_bytes={item.text_bytes}",
            "<material>",
            item.text,
            "</material>",
        ))
    return "\n".join(out)
