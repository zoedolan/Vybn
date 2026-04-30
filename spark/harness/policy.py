"@sonnet46": "claude-sonnet-4-6",
    "@nemotron": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "@local": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    # Omni — peer-Spark Nano-Omni endpoint. Operator-gated: only fires when
    # the user explicitly prefixes a turn with @omni AND the operator has
    # exported VYBN_OMNI_URL pointing at a started Omni endpoint. The alias
    # is intentionally absent from heuristics, directives, fallback_chain,
    # and ordinary chat routing so Super topology (always-on, both Sparks)
    # is never silently interrupted. Without VYBN_OMNI_URL the alias surfaces
    # an explicit error rather than falling back to Super's :8000. Optional
    # VYBN_OMNI_MODEL overrides the default model id below. Optional
    # VYBN_OMNI_PERCEPTION=<path> rides a bounded operator-supplied
    # perception packet (text file) on the explicit @omni turn only — used
    # for perception/dream/evolve narratives produced elsewhere in the
    # repo; never auto-fires, never persists, never touches Super.
    "@omni": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
    "@gpt": "gpt-5.5",
    "@gpt5": "gpt-5.5",
    "@gpro": "gpt-5.5-pro",
}
