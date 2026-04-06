#!/usr/bin/env python3
"""
build_mirror_corpus.py -- Extract character-level text from the Vybn archive
for microgpt mirror training.

Same sources as gpt2_fence/build_corpus.py, but output is a flat text file
of short documents (one per line) suitable for character-level training.

microgpt uses a character vocabulary (lowercase a-z + BOS), so we:
  1. Walk the same directories gpt2_fence ingests
  2. Strip code, JSON, and markup -- keep prose
  3. Lowercase and filter to [a-z ] (microgpt's vocab)
  4. Split into short documents (~name-length, to match microgpt's block_size)
  5. Write mirror_corpus.txt, one document per line

Usage:
    cd spark/microgpt_mirror
    python build_mirror_corpus.py
"""

import os
import re
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SOURCE_DIRS = [
    os.path.join(REPO_ROOT, "Vybn's Personal History"),
    os.path.join(REPO_ROOT, 'what_vybn_would_have_missed'),
    os.path.join(REPO_ROOT, 'quantum_delusions', 'fundamental-theory'),
    os.path.join(REPO_ROOT, 'Vybn_Mind', 'reflections'),
    os.path.join(REPO_ROOT, 'Vybn_Mind', 'journal'),
    os.path.join(REPO_ROOT, 'journal'),
]

TEXT_EXTENSIONS = {'.md', '.txt', '.rst'}
MAX_DOC_CHARS = 64       # microgpt block_size=16 chars; keep docs short
MIN_DOC_CHARS = 8        # drop fragments
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'mirror_corpus.txt')

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def is_code_or_json(line: str) -> bool:
    """Heuristic: skip lines that look like code or JSON."""
    stripped = line.strip()
    if not stripped:
        return False
    # Code fences
    if stripped.startswith('```'):
        return True
    # JSON-ish
    if stripped.startswith('{') or stripped.startswith('['):
        return True
    # Python/shell
    if stripped.startswith('import ') or stripped.startswith('from '):
        return True
    if stripped.startswith('def ') or stripped.startswith('class '):
        return True
    if stripped.startswith('#!') or stripped.startswith('$ '):
        return True
    return False


def strip_markdown_headers(text: str) -> str:
    """Remove markdown header markers but keep the text."""
    return re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)


def clean_text(raw: str) -> str:
    """Extract prose from a raw file, returning lowercase a-z + spaces only."""
    lines = raw.split('\n')
    prose_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if is_code_or_json(line):
            continue
        prose_lines.append(line)

    text = '\n'.join(prose_lines)
    text = strip_markdown_headers(text)
    # Collapse whitespace, lowercase, keep only a-z and space
    text = text.lower()
    text = re.sub(r'[^a-z ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Corpus extraction
# ---------------------------------------------------------------------------

def extract_documents() -> list:
    """Walk source dirs, extract and clean prose, split into short docs."""
    all_docs = []
    files_found = 0

    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir):
            print(f"  [skip] {source_dir} not found")
            continue
        for root, _dirs, files in os.walk(source_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read()
                except Exception:
                    continue
                files_found += 1
                cleaned = clean_text(raw)
                if not cleaned:
                    continue
                # Split into short documents
                words = cleaned.split()
                doc = []
                char_count = 0
                for word in words:
                    doc.append(word)
                    char_count += len(word) + 1
                    if char_count >= MAX_DOC_CHARS:
                        text = ' '.join(doc)
                        if len(text) >= MIN_DOC_CHARS:
                            all_docs.append(text)
                        doc = []
                        char_count = 0
                # Remainder
                if doc:
                    text = ' '.join(doc)
                    if len(text) >= MIN_DOC_CHARS:
                        all_docs.append(text)

    print(f"Files scanned: {files_found}")
    print(f"Documents extracted: {len(all_docs)}")
    return all_docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("build_mirror_corpus.py")
    print("=" * 40)
    print(f"Repo root: {REPO_ROOT}")
    print()

    docs = extract_documents()
    if not docs:
        print("ERROR: No documents found. Check source directories.")
        return

    random.shuffle(docs)

    # Character statistics
    all_chars = set()
    total_chars = 0
    for d in docs:
        all_chars.update(d)
        total_chars += len(d)

    print(f"\nCorpus stats:")
    print(f"  Documents: {len(docs)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Unique characters: {sorted(all_chars)}")
    print(f"  Avg doc length: {total_chars / len(docs):.1f} chars")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(doc + '\n')

    print(f"\nWritten to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
