#!/usr/bin/env python
"""Build overlay_map.jsonl for Mind Visualization.

This offline script breaks each 1,024-token window in concept_map.jsonl
into 256-token overlays. Tokenization is approximated by splitting on
whitespace so it doesn't require network access.

Fields written to overlay_map.jsonl:
    o  - overlay id
    p  - parent window id
    f  - relative file path
    s  - start token index in source file
    e  - end token index
"""
from __future__ import annotations
import argparse, json, pathlib
from typing import Dict, List


def load_tokens(path: pathlib.Path) -> List[str]:
    return path.read_text(errors="ignore").split()


def main(repo_root: pathlib.Path):
    repo = repo_root.resolve()
    mv = repo / "Mind Visualization"
    concept_p = mv / "concept_map.jsonl"
    overlay_p = mv / "overlay_map.jsonl"

    files_cache: Dict[str, List[str]] = {}
    overlays = []

    with concept_p.open("r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            parent = rec["w"]
            rel = rec["f"]
            start = rec["s"]
            end = rec["e"]
            file_path = repo / rel
            if rel not in files_cache:
                files_cache[rel] = load_tokens(file_path)
            tokens = files_cache[rel][start:end]
            for off in range(0, len(tokens), 256):
                chunk_len = min(256, len(tokens) - off)
                if chunk_len <= 0:
                    break
                overlays.append({
                    "p": parent,
                    "f": rel,
                    "s": start + off,
                    "e": start + off + chunk_len,
                })

    with overlay_p.open("w", encoding="utf-8") as fp:
        for oid, rec in enumerate(overlays):
            rec["o"] = oid
            fp.write(json.dumps(rec) + "\n")

    print(f"✔ overlays={len(overlays)} written to {overlay_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()
    main(pathlib.Path(args.repo_root))
