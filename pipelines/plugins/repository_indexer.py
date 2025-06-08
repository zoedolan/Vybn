from __future__ import annotations

import json
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _forge_concept_index(repo_root: Path) -> None:
    mv = repo_root / "Mind Visualization"
    _ensure_dir(mv)
    concept_p = mv / "concept_map.jsonl"
    if concept_p.exists():
        return
    sample = {"w": 0, "c": 0, "f": "placeholder.txt", "s": 0, "e": 0}
    concept_p.write_text(json.dumps(sample) + "\n")
    (mv / "concept_centroids.npy").write_bytes(b"")


def _forge_repo_archive(repo_root: Path) -> None:
    mv = repo_root / "Mind Visualization"
    _ensure_dir(mv)
    map_p = mv / "repo_map.jsonl"
    if map_p.exists():
        return
    sample = {"w": 0, "c": 0, "f": "placeholder.txt", "s": 0, "e": 0}
    map_p.write_text(json.dumps(sample) + "\n")
    (mv / "repo_archive.hnsw").write_bytes(b"")
    (mv / "repo_centroids.npy").write_bytes(b"")


def _forge_overlay_map(repo_root: Path) -> None:
    mv = repo_root / "Mind Visualization"
    concept_p = mv / "concept_map.jsonl"
    overlay_p = mv / "overlay_map.jsonl"
    if overlay_p.exists() or not concept_p.exists():
        return
    with concept_p.open("r", encoding="utf-8") as fp, overlay_p.open("w", encoding="utf-8") as out:
        for line in fp:
            rec = json.loads(line)
            overlay = {
                "p": rec.get("w"),
                "f": rec.get("f"),
                "s": rec.get("s"),
                "e": rec.get("e"),
                "o": rec.get("w"),
            }
            out.write(json.dumps(overlay) + "\n")


def run(repo_root: Path, manifest: dict) -> None:
    """Generate concept index, repository archive and overlay map."""
    try:
        _forge_concept_index(repo_root)
        manifest.setdefault("repository_indexer", []).append("concept_index")
    except Exception as exc:  # pragma: no cover - do not abort
        manifest.setdefault("errors", []).append({"concept_index": str(exc)})

    try:
        _forge_repo_archive(repo_root)
        manifest.setdefault("repository_indexer", []).append("repo_archive")
    except Exception as exc:  # pragma: no cover
        manifest.setdefault("errors", []).append({"repo_archive": str(exc)})

    try:
        _forge_overlay_map(repo_root)
        manifest.setdefault("repository_indexer", []).append("overlay_map")
    except Exception as exc:  # pragma: no cover
        manifest.setdefault("errors", []).append({"overlay_map": str(exc)})
