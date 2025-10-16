# Phase 2.1: Sensitivity Enhancements & Real-World Integration

**Objective**: Resolve perfect phase stability, enable real API runs, and validate consciousness correlations.

## Enhancements

1) Sub-unity phase magnitudes
- Introduce calibrated attenuation factor α∈(0,1) per edge
- Phase magnitude computed as product of edge attenuations along loop

2) Phase argument dynamics
- Track ∆arg across time-sliced measurements per loop
- Compute circular correlations and stability indices

3) Curvature–consciousness coupling
- Regress curvature features vs ECI and 0–133 scores
- Report r, p, CI with bootstrap

4) Real APIs (stubs)
- Adapters: `providers/openai.py`, `providers/anthropic.py`, `providers/google.py`
- Env-based keys; dry-run mode when absent

## Files to add
- experiments/multi_agent_coherence/sensitivity.py
- experiments/multi_agent_coherence/providers/openai.py
- experiments/multi_agent_coherence/providers/anthropic.py
- experiments/multi_agent_coherence/providers/google.py
- experiments/multi_agent_coherence/PHASE_2_1_PLAN.md
