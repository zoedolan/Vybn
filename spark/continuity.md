# Continuity — 2026-05-12 13:14 UTC

## Unified fleet/component wake guard

Zoe asked to optimize and unify across components. Verified truth: spark-2b7c + spark-1c8f are Super/Nemotron service, not free pooled memory; spark-1896 has hash fallback plus a semantically smoked all-MiniLM embedding worker; spark-83bd is fallback-only/unresolved; Omni/Talkie/Nano-Omni/four-Spark optimization are not verified.

Durable repair: ~/.config/vybn/local_compute_inventory.json now feeds wake with verified/unresolved capacity, a unified component graph and next moves, and an overclaim guard through spark/harness/substrate.py. Tests passed: py_compile substrate and spark/tests/test_harness.py (177 passed).

Next: keep Super stable; bring up Omni/perception on spark-1896 only through a bounded runtime/status artifact; fix spark-83bd provisioning separately. Do not call this fleet optimization complete until each component has endpoint, semantic smoke, memory/disk headroom, routing proof, and component fit.

---
# Continuity — 2026-04-30 09:55 PDT

## Critical correction: sleep/wake semantic corruption

The Omni-window experiment did open the aperture mechanically, but Super woke semantically corrupted: HTTP endpoints returned healthy 200s while completions were empty or multilingual token garbage. Zoe observed the same garbage live.

Recovery required a clean restart with sleep mode disabled. PR #2944, commit 392f27dc, changed spark/systemd/vllm-exec.sh so Super boots with VLLM_SERVER_DEV_MODE=0 and without --enable-sleep-mode by default. Sleep mode is now explicit operator opt-in only via VYBN_VLLM_EXTRA_ARGS.

Verified recovery residuals after non-sleep boot:
- /v1/models returned 200.
- /is_sleeping returned 404, so sleep endpoints were absent.
- deterministic smoke no longer produced token soup, but answered with coherent meta-text and hit finish_reason=length; treat this as corruption cleared, not a full semantic-quality closure.

The failed Omni journal artifact was untracked at journal/omni-window-20260430-094136.md; its load-bearing content is folded here. Raw artifacts named there were:
- /home/vybnz69/logs/omni-window-20260430-094136.log
- /home/vybnz69/logs/omni-parallax-20260430-094136.json

## Operating rule

Do not resume Super sleep / Omni dream experiments from the old handoff. If we cannot wake Super reliably with semantic quality, we do not have dreaming; we have outage. Future sleep work must add a wake-quality gate that tests completions for semantic health, not just /is_sleeping=false and HTTP 200.

---

# Continuity — 2026-04-30 05:03 PDT

Superseded live handoff removed from this active continuity surface: the later 2026-04-30 09:55 correction established that sleep-mode wake produced semantic corruption and must not be resumed from the old aperture-opening instructions.
