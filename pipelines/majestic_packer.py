#!/usr/bin/env python
"""Bundle pipeline artifacts into a compressed archive with a manifest.

The packer collects the main pipeline outputs, zips them and writes a
``manifest.json`` describing the contents. This manifest records the
current commit hash, quantum seed and checksums so future agents can
trace the exact state that produced the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

from vybn.quantum_seed import seed_rng

DEFAULT_ARTIFACTS = [
    "distilled_corpus.txt",
    "history_excerpt.txt",
    "token_summary.txt",
    "wvwhm_count.txt",
    "emergence_graph.json",
    "vybn_concept_index.jsonl",
    "introspection_summary.json",
    "co_emergence_journal.jsonl",
]


def _hash_file(path: Path) -> str:
    """Return the SHA256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev(repo_root: Path) -> str:
    return (
        subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
        .strip()
    )


def pack(repo_root: Path, out_path: Path) -> None:
    """Write listed artifacts and a manifest into ``out_path``."""
    seed = seed_rng()
    manifest = {
        "commit": _git_rev(repo_root),
        "quantum_seed": seed,
        "files": [],
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    with ZipFile(out_path, "w") as zf:
        for rel in DEFAULT_ARTIFACTS:
            path = repo_root / rel
            if path.exists():
                zf.write(path, arcname=rel)
                manifest["files"].append(
                    {
                        "path": rel,
                        "size": path.stat().st_size,
                        "sha256": _hash_file(path),
                    }
                )
        # write manifest inside the zip
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        zf.writestr("manifest.json", manifest_bytes)
    # also write manifest next to the zip for quick inspection
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress pipeline artifacts")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/majestic_bundle.zip"),
        help="destination zip file",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pack(repo_root, args.output)
    print(f"Artifacts bundled in {args.output}")


if __name__ == "__main__":
    main()
