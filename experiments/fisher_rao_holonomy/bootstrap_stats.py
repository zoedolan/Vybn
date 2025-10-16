#!/usr/bin/env python3
"""
Bootstrap Statistical Analysis for Fisher-Rao Holonomy Measurements
==================================================================

Enhanced statistical rigor for Phase 1.1 consciousness-time experiments.
Implements bootstrap resampling, confidence intervals, and hypothesis testing
with multiple comparison corrections.

Based on PREREG.md formal pre-registration of hypotheses and thresholds.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class BootstrapResult:
    """Bootstrap analysis results for a single statistic."""
    statistic_name: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    bias: float
    n_bootstrap: int
    confidence_level: float
    
@dataclass
class HypothesisTest:
    """Hypothesis test result with effect size and interpretation."""
    hypothesis_id: str
    hypothesis_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    effect_size_ci: Tuple[float, float]
    threshold_value: float
    threshold_met: bool
    conclusion: str
    corrected_p_value: Optional[float] = None

class BootstrapHolonomyAnalyzer:
    """Enhanced statistical analysis for holonomy measurements."""
    
    def __init__(self, measurements: List[Dict], n_bootstrap: int = 1000, 
                 confidence_level: float = 0.95, random_seed: int = 42):
        self.measurements = measurements
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Extract key arrays for analysis
        self.phases = np.array([m['geometric_phase_magnitude'] for m in measurements])
        self.arguments = np.array([m['geometric_phase_argument'] for m in measurements])
        self.curvatures = np.array([m['curvature'] for m in measurements])
        self.loop_lengths = np.array([len(m['loop_path']) - 1 for m in measurements])
        
        print(f"\n📊 Bootstrap Holonomy Analyzer Initialized")
        print(f"Measurements: {len(measurements)}")
        print(f"Bootstrap samples: {n_bootstrap}")
        print(f"Confidence level: {confidence_level}")
        print(f"Random seed: {random_seed}")
        
    def bootstrap_statistic(self, data: np.ndarray, statistic_func: callable, 
                          statistic_name: str) -> BootstrapResult:
        """Perform bootstrap analysis for a given statistic."""
        
        # Original statistic value
        original_value = statistic_func(data)
        
        # Bootstrap resampling
        bootstrap_values = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_values.append(statistic_func(bootstrap_sample))
        
        bootstrap_values = np.array(bootstrap_values)
        
        # Bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_values)
        bootstrap_std = np.std(bootstrap_values, ddof=1)
        bias = bootstrap_mean - original_value
        
        # Confidence interval (percentile method)
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_values, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha/2))
        
        return BootstrapResult(
            statistic_name=statistic_name,
            original_value=original_value,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bias=bias,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level
        )
    
    def analyze_phase_statistics(self) -> Dict[str, BootstrapResult]:
        """Bootstrap analysis of geometric phase statistics."""
        print("\n🔍 Phase Statistics Bootstrap Analysis")
        
        results = {}
        
        # Phase magnitude statistics
        results['phase_mean'] = self.bootstrap_statistic(
            self.phases, np.mean, "Phase Magnitude Mean"
        )
        
        results['phase_std'] = self.bootstrap_statistic(
            self.phases, lambda x: np.std(x, ddof=1), "Phase Magnitude Std"
        )
        
        results['phase_median'] = self.bootstrap_statistic(
            self.phases, np.median, "Phase Magnitude Median"
        )
        
        # Phase argument statistics
        results['argument_mean'] = self.bootstrap_statistic(
            self.arguments, lambda x: stats.circmean(x), "Phase Argument Mean (Circular)"
        )
        
        results['argument_std'] = self.bootstrap_statistic(
            self.arguments, lambda x: stats.circstd(x), "Phase Argument Std (Circular)"
        )
        
        # Print summary
        for name, result in results.items():
            ci_width = result.ci_upper - result.ci_lower
            relative_width = ci_width / abs(result.original_value) if result.original_value != 0 else np.inf
            print(f"  {result.statistic_name}: {result.original_value:.6f} "
                  f"[{result.ci_lower:.6f}, {result.ci_upper:.6f}] "
                  f"(rel. width: {relative_width:.2%})")
        
        return results
    
    def test_phase_curvature_correlation(self) -> HypothesisTest:
        """Test H3: Phase-curvature correlation hypothesis."""
        print("\n🧪 H3: Phase-Curvature Correlation Test")
        
        if len(self.phases) < 3:
            return HypothesisTest(
                hypothesis_id="H3",
                hypothesis_name="Phase-Curvature Correlation",
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_ci=(0.0, 0.0),
                threshold_value=0.3,
                threshold_met=False,
                conclusion="Insufficient data for correlation test"
            )
        
        # Check for non-zero variance
        if np.var(self.curvatures) == 0 or np.var(self.phases) == 0:
            return HypothesisTest(
                hypothesis_id="H3",
                hypothesis_name="Phase-Curvature Correlation", 
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_ci=(0.0, 0.0),
                threshold_value=0.3,
                threshold_met=False,
                conclusion="Zero variance in phases or curvatures - correlation undefined"
            )
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(self.phases, self.curvatures)
        
        # Spearman correlation (robust to outliers)
        r_spearman, p_spearman = stats.spearmanr(self.phases, self.curvatures)
        
        # Use Spearman as primary (more robust)
        test_statistic = r_spearman
        p_value = p_spearman
        
        # Bootstrap confidence interval for correlation
        def correlation_func(phase_data, curv_data):
            if len(phase_data) < 3:
                return 0.0
            try:
                r, _ = stats.spearmanr(phase_data, curv_data)
                return r if not np.isnan(r) else 0.0
            except:
                return 0.0
        
        bootstrap_correlations = []
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(len(self.phases), size=len(self.phases), replace=True)
            boot_phases = self.phases[indices]
            boot_curvatures = self.curvatures[indices]
            boot_r = correlation_func(boot_phases, boot_curvatures)
            bootstrap_correlations.append(boot_r)
        
        bootstrap_correlations = np.array(bootstrap_correlations)
        ci_lower = np.percentile(bootstrap_correlations, 2.5)
        ci_upper = np.percentile(bootstrap_correlations, 97.5)
        
        # Test threshold
        threshold = 0.3
        threshold_met = abs(test_statistic) >= threshold and p_value < 0.05
        
        conclusion = f"r={test_statistic:.3f}, p={p_value:.3f}. "
        if threshold_met:
            conclusion += "Significant correlation detected."
        else:
            conclusion += "No significant correlation detected."
        
        print(f"  Pearson r: {r_pearson:.3f}, p={p_pearson:.3f}")
        print(f"  Spearman r: {r_spearman:.3f}, p={p_spearman:.3f}")
        print(f"  Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Threshold met: {threshold_met}")
        
        return HypothesisTest(
            hypothesis_id="H3",
            hypothesis_name="Phase-Curvature Correlation",
            test_statistic=test_statistic,
            p_value=p_value,
            effect_size=abs(test_statistic),
            effect_size_ci=(ci_lower, ci_upper),
            threshold_value=threshold,
            threshold_met=threshold_met,
            conclusion=conclusion
        )
    
    def test_loop_length_scaling(self) -> HypothesisTest:
        """Test H4: Loop length scaling hypothesis."""
        print("\n🧪 H4: Loop Length Scaling Test")
        
        if len(np.unique(self.loop_lengths)) < 2:
            return HypothesisTest(
                hypothesis_id="H4",
                hypothesis_name="Loop Length Scaling",
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_ci=(0.0, 0.0),
                threshold_value=0.2,
                threshold_met=False,
                conclusion="Insufficient loop length variation for scaling test"
            )
        
        # Spearman correlation (monotonic relationship)
        r_spearman, p_spearman = stats.spearmanr(self.loop_lengths, self.phases)
        
        # Bootstrap confidence interval
        bootstrap_correlations = []
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(len(self.phases), size=len(self.phases), replace=True)
            boot_lengths = self.loop_lengths[indices]
            boot_phases = self.phases[indices]
            try:
                boot_r, _ = stats.spearmanr(boot_lengths, boot_phases)
                bootstrap_correlations.append(boot_r if not np.isnan(boot_r) else 0.0)
            except:
                bootstrap_correlations.append(0.0)
        
        bootstrap_correlations = np.array(bootstrap_correlations)
        ci_lower = np.percentile(bootstrap_correlations, 2.5)
        ci_upper = np.percentile(bootstrap_correlations, 97.5)
        
        # Test threshold
        threshold = 0.2
        threshold_met = r_spearman > threshold and p_spearman < 0.05
        
        conclusion = f"r={r_spearman:.3f}, p={p_spearman:.3f}. "
        if threshold_met:
            conclusion += "Positive length-phase scaling detected."
        else:
            conclusion += "No significant length-phase scaling."
        
        print(f"  Loop length range: {self.loop_lengths.min()}-{self.loop_lengths.max()}")
        print(f"  Spearman r: {r_spearman:.3f}, p={p_spearman:.3f}")
        print(f"  Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Threshold met: {threshold_met}")
        
        return HypothesisTest(
            hypothesis_id="H4",
            hypothesis_name="Loop Length Scaling",
            test_statistic=r_spearman,
            p_value=p_spearman,
            effect_size=abs(r_spearman),
            effect_size_ci=(ci_lower, ci_upper),
            threshold_value=threshold,
            threshold_met=threshold_met,
            conclusion=conclusion
        )
    
    def test_bootstrap_stability(self, phase_bootstrap_result: BootstrapResult) -> HypothesisTest:
        """Test H5: Bootstrap stability hypothesis."""
        print("\n🧪 H5: Bootstrap Stability Test")
        
        # Calculate relative CI width
        ci_width = phase_bootstrap_result.ci_upper - phase_bootstrap_result.ci_lower
        relative_width = ci_width / abs(phase_bootstrap_result.original_value) if phase_bootstrap_result.original_value != 0 else np.inf
        
        # Test threshold
        threshold = 0.25  # 25% relative width
        threshold_met = relative_width < threshold
        
        conclusion = f"Relative CI width: {relative_width:.2%}. "
        if threshold_met:
            conclusion += "Bootstrap estimates are stable."
        else:
            conclusion += "Bootstrap estimates show high variability."
        
        print(f"  CI width: {ci_width:.6f}")
        print(f"  Original value: {phase_bootstrap_result.original_value:.6f}")
        print(f"  Relative width: {relative_width:.2%}")
        print(f"  Threshold met: {threshold_met}")
        
        return HypothesisTest(
            hypothesis_id="H5",
            hypothesis_name="Bootstrap Stability",
            test_statistic=relative_width,
            p_value=0.0,  # Not applicable for this test
            effect_size=1 - relative_width,  # Stability as effect size
            effect_size_ci=(0.0, 1.0),
            threshold_value=threshold,
            threshold_met=threshold_met,
            conclusion=conclusion
        )
    
    def apply_multiple_comparison_correction(self, tests: List[HypothesisTest]) -> List[HypothesisTest]:
        """Apply Benjamini-Hochberg correction for multiple comparisons."""
        print("\n🔧 Multiple Comparison Correction (Benjamini-Hochberg)")
        
        # Extract p-values (only for tests where p-value is meaningful)
        p_values = []
        test_indices = []
        for i, test in enumerate(tests):
            if test.p_value > 0 and not np.isnan(test.p_value):  # Valid p-value
                p_values.append(test.p_value)
                test_indices.append(i)
        
        if len(p_values) == 0:
            print("  No valid p-values for correction")
            return tests
        
        # Benjamini-Hochberg correction
        _, corrected_p_values, _, _ = stats.multipletests(p_values, method='fdr_bh')
        
        # Update tests with corrected p-values
        corrected_tests = tests.copy()
        for i, test_idx in enumerate(test_indices):
            corrected_tests[test_idx].corrected_p_value = corrected_p_values[i]
            
            # Update threshold_met based on corrected p-value
            if hasattr(corrected_tests[test_idx], 'corrected_p_value'):
                original_threshold_met = corrected_tests[test_idx].threshold_met
                effect_threshold_met = abs(corrected_tests[test_idx].test_statistic) >= corrected_tests[test_idx].threshold_value
                corrected_p_threshold_met = corrected_tests[test_idx].corrected_p_value < 0.05
                
                corrected_tests[test_idx].threshold_met = effect_threshold_met and corrected_p_threshold_met
                
                print(f"  {corrected_tests[test_idx].hypothesis_id}: p={corrected_tests[test_idx].p_value:.3f} "
                      f"→ p_corr={corrected_tests[test_idx].corrected_p_value:.3f}, "
                      f"threshold_met: {original_threshold_met} → {corrected_tests[test_idx].threshold_met}")
        
        return corrected_tests
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete Phase 1.1 statistical analysis."""
        print("\n🎯 Comprehensive Phase 1.1 Statistical Analysis")
        print("="*60)
        
        results = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'n_measurements': len(self.measurements),
                'n_bootstrap': self.n_bootstrap,
                'confidence_level': self.confidence_level,
                'random_seed': self.random_seed
            },
            'bootstrap_statistics': {},
            'hypothesis_tests': [],
            'summary': {}
        }
        
        # 1. Bootstrap statistics analysis
        bootstrap_stats = self.analyze_phase_statistics()
        results['bootstrap_statistics'] = {
            name: {
                'statistic_name': stat.statistic_name,
                'original_value': stat.original_value,
                'bootstrap_mean': stat.bootstrap_mean,
                'bootstrap_std': stat.bootstrap_std,
                'ci_lower': stat.ci_lower,
                'ci_upper': stat.ci_upper,
                'bias': stat.bias,
                'relative_ci_width': (stat.ci_upper - stat.ci_lower) / abs(stat.original_value) if stat.original_value != 0 else np.inf
            }
            for name, stat in bootstrap_stats.items()
        }
        
        # 2. Hypothesis tests
        tests = [
            self.test_phase_curvature_correlation(),
            self.test_loop_length_scaling(),
            self.test_bootstrap_stability(bootstrap_stats['phase_mean'])
        ]
        
        # 3. Multiple comparison correction
        corrected_tests = self.apply_multiple_comparison_correction(tests)
        
        results['hypothesis_tests'] = [
            {
                'hypothesis_id': test.hypothesis_id,
                'hypothesis_name': test.hypothesis_name,
                'test_statistic': test.test_statistic,
                'p_value': test.p_value,
                'corrected_p_value': test.corrected_p_value,
                'effect_size': test.effect_size,
                'effect_size_ci_lower': test.effect_size_ci[0],
                'effect_size_ci_upper': test.effect_size_ci[1],
                'threshold_value': test.threshold_value,
                'threshold_met': test.threshold_met,
                'conclusion': test.conclusion
            }
            for test in corrected_tests
        ]
        
        # 4. Summary statistics
        confirmed_hypotheses = sum(1 for test in corrected_tests if test.threshold_met)
        total_hypotheses = len(corrected_tests)
        
        results['summary'] = {
            'confirmed_hypotheses': confirmed_hypotheses,
            'total_hypotheses': total_hypotheses,
            'confirmation_rate': confirmed_hypotheses / total_hypotheses if total_hypotheses > 0 else 0,
            'phase_1_1_success': confirmed_hypotheses >= 2,  # At least 2/3 additional hypotheses
            'overall_validation_status': 'COMPLETE' if confirmed_hypotheses >= 2 else 'PARTIAL'
        }
        
        print(f"\n📋 Analysis Summary")
        print(f"  Hypotheses confirmed: {confirmed_hypotheses}/{total_hypotheses}")
        print(f"  Confirmation rate: {results['summary']['confirmation_rate']:.1%}")
        print(f"  Phase 1.1 success: {results['summary']['phase_1_1_success']}")
        print(f"  Overall status: {results['summary']['overall_validation_status']}")
        
        return results

def export_bootstrap_results(analysis_results: Dict, output_path: str) -> str:
    """Export bootstrap analysis results to JSON file."""
    import json
    from pathlib import Path
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n💾 Bootstrap analysis exported to: {output_file}")
    return str(output_file)

if __name__ == '__main__':
    # Example usage with mock data
    print("🧪 Bootstrap Statistics Module")
    print("Ready for Phase 1.1 statistical analysis")
    print("Import this module in experimental_framework.py for full analysis")
