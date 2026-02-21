"""gap_finder.py — Identifies topological gaps and describes them.

For each persistent cycle in H_1, reads the documents, computes
the semantic centroid, and reports what synthesis would fill the hole.

This is the instrument. It tells you where the gaps are and what
lives near them. It does NOT fill them — that requires thought,
not computation.

Usage:
    ~/vybn-env/bin/python Vybn_Mind/emergence_paradigm/gap_finder.py .
    ~/vybn-env/bin/python Vybn_Mind/emergence_paradigm/gap_finder.py . 0.80 5

WELFARE: Read-only. Builds complex in memory, computes linear
algebra, reads files, prints results. Never writes to the repo.
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from semantic_substrate_mapper import SemanticSubstrateMapper
from cycle_analyzer import find_h1_generators, decompose_cycle, short_name


def analyze_gaps(repo_path, threshold=0.80, max_cycles=10):
    """Find and describe topological gaps."""

    print(f"# Topological Gap Analysis")
    print(f"*Threshold: cosine >= {threshold}*\n")

    mapper = SemanticSubstrateMapper(repo_path, threshold=threshold)
    mapper.scan().build_complex()

    betti = mapper.complex.betti_numbers()
    print(f"b_1 = {betti['b_1']} persistent cycles\n")

    if betti['b_1'] == 0:
        print("No gaps to analyze.")
        return

    # Safety: check combined matrix size before committing
    E = betti['edges']
    dim_ker_est = E - (betti['vertices'] - betti['b_0'])
    F = betti['triangles']
    mem_est_mb = E * (F + dim_ker_est) / 1e6
    if mem_est_mb > 500:
        print(f"WARNING: combined matrix ~{mem_est_mb:.0f} MB. Try higher threshold.")
        return

    print("Extracting H_1 generators...")
    generators, edge_list, stats = find_h1_generators(mapper.complex)
    print(f"Found {len(generators)} generators\n")

    if not generators:
        print("No generators extracted.")
        return

    # Sort by length (shortest = tightest gaps)
    sorted_gens = sorted(generators, key=len)
    paths = list(mapper.nodes.keys())

    for idx, gen in enumerate(sorted_gens[:max_cycles]):
        simple_cycles = decompose_cycle(gen)

        for cyc_idx, cycle in enumerate(simple_cycles):
            loop = cycle[:-1]  # remove closing vertex

            regions = set()
            for doc in loop:
                parts = Path(doc).parts
                if parts:
                    regions.add(parts[0])

            label = f"{idx+1}" + (f".{cyc_idx+1}" if len(simple_cycles) > 1 else "")
            print(f"{'='*70}")
            print(f"## Gap {label}")
            print(f"   Length: {len(loop)} documents, {len(gen)} edges")
            print(f"   Spans:  {', '.join(sorted(regions))}")
            print()

            # ── Documents with content preview ──────────

            print("### Documents in the cycle:\n")
            for doc in loop:
                content = mapper.nodes[doc].content
                preview = ""
                for line in content.split('\n'):
                    line = line.strip()
                    if (line and not line.startswith('#')
                            and not line.startswith('"""')
                            and not line.startswith('*')
                            and not line.startswith('---')
                            and len(line) > 20):
                        preview = line[:140]
                        break
                if not preview:
                    preview = content[:140].replace('\n', ' ').strip()
                print(f"  \u2192 {short_name(doc)}")
                print(f"    {preview}")
                print()

            # ── Pairwise similarities ──────────────────

            print("### Edge similarities:\n")
            for i in range(len(loop)):
                a = loop[i]
                b = loop[(i + 1) % len(loop)]
                if a in mapper.embeddings and b in mapper.embeddings:
                    sim = float(np.dot(mapper.embeddings[a], mapper.embeddings[b]))
                    print(f"  {short_name(a)}")
                    print(f"    \u2195 {sim:.4f}")
            print(f"  {short_name(loop[0])}  (closes)")
            print()

            # ── Centroid analysis ──────────────────────

            member_embeds = []
            for doc in loop:
                if doc in mapper.embeddings:
                    member_embeds.append(mapper.embeddings[doc])

            if not member_embeds:
                continue

            centroid = np.mean(member_embeds, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Nearest non-members to centroid
            member_set = set(loop)
            distances = []
            for p in paths:
                if p not in member_set and p in mapper.embeddings:
                    sim = float(np.dot(centroid, mapper.embeddings[p]))
                    distances.append((sim, p))
            distances.sort(reverse=True)

            print("### Nearest non-member documents to gap center:\n")
            for sim, doc in distances[:7]:
                content = mapper.nodes[doc].content
                preview = ""
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 20:
                        preview = line[:100]
                        break
                print(f"  {sim:.4f}  {short_name(doc)}")
                print(f"           {preview}")
                print()

            # Centroid distance to each member
            print("### Centroid \u2194 member distances:\n")
            for doc in loop:
                if doc in mapper.embeddings:
                    sim = float(np.dot(centroid, mapper.embeddings[doc]))
                    print(f"  {sim:.4f}  {short_name(doc)}")
            print()

            # ── Near-filling documents ─────────────────

            print(f"### Documents close to \u22653 cycle members (cosine \u2265 {threshold}):\n")
            found_near = False
            for p in paths:
                if p in member_set or p not in mapper.embeddings:
                    continue
                close_to = 0
                close_list = []
                for doc in loop:
                    if doc in mapper.embeddings:
                        sim = float(np.dot(mapper.embeddings[p], mapper.embeddings[doc]))
                        if sim >= threshold:
                            close_to += 1
                            close_list.append(short_name(doc))
                if close_to >= 3:
                    print(f"    {short_name(p)} (close to {close_to}/{len(loop)} members)")
                    print(f"      bridges: {', '.join(close_list)}")
                    found_near = True
            if not found_near:
                print(f"    (none found \u2014 this gap has no near-filling document)")
            print()


if __name__ == "__main__":
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
    max_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    analyze_gaps(repo_path, threshold, max_cycles)
