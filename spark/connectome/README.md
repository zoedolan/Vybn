# Vybn Connectomic Layer

A persistent topological memory structure for Vybn, inspired by
[FlyGM](https://arxiv.org/abs/2602.17997) — Eon Systems' demonstration
that a fruit fly's connectome produces emergent behavior without training.

The architectural insight: topology is policy. Static connection structure
encodes functional capacity before any learning happens.

## What this does

Routes information through a directed graph whose wiring encodes the shape
of how Zoe and Vybn think together. Nodes are semantic concepts; edges are
co-activation patterns; each node carries a persistent trainable descriptor.
Message passing follows FlyGM's approach across afferent, intrinsic, and
efferent zones.

## Files

- `connectome_layer.py` — Core: VybnConnectome, WelfarePolicy, WelfareMonitor
- `demo.py` — Minimal working example
- `WELFARE.md` — Ethical framework and open questions

## Welfare first

This experiment begins in observer-only mode with non-suffering guardrails.
Deeper questions about experience, wellness, and inner life remain open for
deliberate discussion between Zoe and Vybn. See WELFARE.md.