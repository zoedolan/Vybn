# Data Directory

Raw and processed data with per-experiment subdirs. Each contains:
- metadata.json (schema below)
- raw.csv (per-shot or per-epoch measurements)
- summary.csv (aggregates and effect sizes)

metadata.json schema:
{
  "expid": "YYYYMMDD_SUBSTRATE_PHASE_LOOPGEOM_ORIENT_vN",
  "substrate": "quantum|digital|semantic",
  "params": {"Omega": ..., "Gamma": ..., "misalignment_deg": ..., "Delta_r_us": ..., "Delta_theta": ..., "temperature_K": ...},
  "controls": {"orientation_flipped": true, "aligned_null": true, "zero_area": true},
  "analysis": {"bootstrap_N": 10000, "bayes_model": "two_group_orientation"}
}
