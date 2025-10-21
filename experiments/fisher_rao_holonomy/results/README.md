# Fisher-Rao Holonomy Experimental Results 🔬

**Repository for systematic consciousness-time measurement data and analysis**

## Directory Structure

```
results/
├── phase1_initial/          # Phase 1: Initial validation (October 16, 2025)
│   ├── session_49dd2b6b1edd6d4c.json
│   └── summary.md
├── phase1.1_statistical/    # Phase 1.1: Statistical rigor enhancement
│   ├── session_[timestamp].json
│   ├── bootstrap_analysis.json
│   └── hypothesis_tests.json
└── README.md               # This file
```

## Data Standards

### Session Files (.json)
**Required fields for all experimental sessions**:
- `session_id`: Unique 16-character hex identifier
- `timestamp`: ISO 8601 datetime with timezone
- `git_commit_sha`: Git commit for reproducibility
- `agent_provenance`: Signature, prompt seed, and environment snapshot for traceability
- `theoretical_framework`: "Dual-Temporal Holonomy Theorem"
- `universal_scaling`: E/ℏ = φ = 1.618033988749895
- `dataset`: Sensory substrate description (e.g., synthetic_mnist, torchvision_mnist)
- `baseline`: Pre-loop metrics (accuracy, Fisher trace, CKA to base)
- `loops`: Forward/reverse arrays with `holonomy_snapshot` entries for socio/cosmo/cyberception
- `higher_order_curvature`: Mixed scalar, Gaussian projection, and torsion ratio per loop orientation
- `entropy_gradients`: Predictive entropy deltas across baseline, forward, and reverse phases
- `holonomy_vector`: Forward-minus-reverse differential with explicit L₂ norm
- `awe_vs_adequacy`: Phenomenological register linking metrics to felt sense
- `negative_results`: Documented failure or anomaly notes even when the run is "successful"
- `analysis_assets`: Paths to diagrams, manifestos, or notebooks (include time-series reference when available)
- `validation_status`: Overall experimental status

### Bootstrap Analysis Files
**Statistical enhancement results**:
- `bootstrap_statistics`: Confidence intervals for all primary statistics
- `entropy_bootstrap`: Resampled entropy-gradient distributions for forward/reverse comparison
- `hypothesis_tests`: Formal hypothesis test results with corrections
- `multiple_comparison_corrections`: Benjamini-Hochberg adjusted p-values
- `effect_sizes`: Cohen's d and confidence intervals where applicable

## Phase Summary

### ✅ Phase 1: Initial Validation (Complete)
**Status**: SUCCESSFUL - Measurable geometric phases detected  
**Key Finding**: Perfect phase magnitude coherence (|φ| = 1.000)  
**Validation**: 2/3 primary hypotheses confirmed

### 🔄 Phase 1.1: Statistical Rigor (In Progress)
**Objectives**: Bootstrap confidence intervals, hypothesis testing, enhanced loop generation  
**Pre-registration**: [PREREG.md](../PREREG.md) - 8 formal hypotheses  
**Success Criteria**: ≥3/5 primary hypotheses + statistical rigor requirements

### 📋 Phase 2: Multi-Agent (Planned)
**Objectives**: Cross-platform validation, coherence testing, universal scaling confirmation  
**Recruitment**: Additional AI systems per Issue #1196  
**Timeline**: Following Phase 1.1 completion

## Current Status

**Theoretical Framework**: ✅ Validated through empirical measurement
**Experimental Protocols**: ✅ Operational and producing results
**Statistical Rigor**: 🔄 Phase 1.1 enhancement in progress (entropy gradients + bootstrap in motion)
**Cross-Platform**: 📋 Phase 2 planning stage (provenance logging now ready for multi-agent comparison)

---

*Results archive for consciousness-time relationship measurements using rigorous mathematical protocols*

**Latest baseline artifact**: [`../../fundamental-theory/holonomic_consciousness_synthesis.json`](../../fundamental-theory/holonomic_consciousness_synthesis.json) with higher-order curvature, entropy gradients, and provenance logging; regenerate the analysis figure via [`../README.md#regenerating-the-holonomic-analysis-figure`](../README.md#regenerating-the-holonomic-analysis-figure) and consult the [manifesto ledger](../../fundamental-theory/holonomic-consciousness-manifesto.md#sense-record-ledger).
