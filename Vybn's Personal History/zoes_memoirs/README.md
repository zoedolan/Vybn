# Zoe's memoirs

This directory contains Zoe Dolan's separated memoir files and a public-safe corpus scaffold for future chronology, embeddings, and visualization.

## Source files

- there_is_room_for_you.txt -- There Is Room for You
- transgender_no_more.txt -- Transgender No More
- jump.txt -- Jump
- to_whom_i_could_have_been.txt -- To Whom I Could Have Been

## Prepared scaffold

- corpus_manifest.json -- book-level metadata, hashes, attribution signals, and claim limits.
- chunk_manifest.jsonl -- stable line-range chunk IDs for later embedding. It does not duplicate memoir text.
- chronology.md -- chronology scaffold awaiting a line-cited extraction pass.
- .gitignore -- keeps generated embeddings, vector stores, and local visualization exhaust out of git.

## Membrane

These memoirs are Zoe's authored provenance. They are not cleanup material, generic model sample data, or an unbounded character corpus. Public reuse requires Zoe's consent, attribution, and curation boundary.

## Future sequence

1. Chronology: extract line-cited date/place/event atoms from the source files.
2. Vectorize: generate local embeddings from source text using chunk_manifest.jsonl as the stable index.
3. Visualize: render timeline plus thematic constellations from metadata and embeddings.
4. Curate: decide what, if anything, becomes part of an optional public character library.
