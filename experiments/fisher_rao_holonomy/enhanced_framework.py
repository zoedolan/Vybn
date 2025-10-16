#!/usr/bin/env python3
"""
Enhanced Fisher-Rao Holonomy Experimental Framework - Phase 1.1
==============================================================

Statistical rigor enhancement with bootstrap analysis, formal hypothesis testing,
and enhanced loop generation for systematic consciousness-time measurements.

Integrates bootstrap_stats.py for comprehensive statistical validation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from experimental_framework import FisherRaoHolonomyExperiment
from bootstrap_stats import BootstrapHolonomyAnalyzer, export_bootstrap_results
import json
from datetime import datetime
import subprocess
import hashlib

class EnhancedFisherRaoExperiment(FisherRaoHolonomyExperiment):
    """Enhanced experimental framework with Phase 1.1 statistical rigor."""
    
    def __init__(self, repo_path: Path, session_id: str = None):
        super().__init__(repo_path, session_id)
        
        # Phase 1.1 enhancements
        self.enhanced_loop_lengths = [3, 4, 5, 6, 7]  # Extended range
        self.min_loops_per_length = 4  # Systematic sampling
        self.bootstrap_samples = 1000  # Statistical rigor
        
        print(f"\n🔬 Phase 1.1 Enhanced Framework Initialized")
        print(f"Extended loop lengths: {self.enhanced_loop_lengths}")
        print(f"Min loops per length: {self.min_loops_per_length}")
        print(f"Bootstrap samples: {self.bootstrap_samples}")
    
    def _generate_enhanced_test_loops(self) -> List[List[str]]:
        """Generate enhanced systematic test loops for Phase 1.1."""
        print("\n📐 Enhanced Loop Generation (Phase 1.1)")
        
        if not self.bundle:
            return []
        
        files = list(self.bundle.fiber_structure.keys())
        loops = []
        failed_attempts = 0
        max_attempts_per_length = 50  # Prevent infinite loops
        
        for target_length in self.enhanced_loop_lengths:
            print(f"  Generating loops of length {target_length}...")
            loops_for_length = 0
            attempts_for_length = 0
            
            while loops_for_length < self.min_loops_per_length and attempts_for_length < max_attempts_per_length:
                attempts_for_length += 1
                
                # Random start point
                start_file = np.random.choice(files)
                loop = [start_file]
                current_file = start_file
                
                # Build loop following references
                success = True
                for step in range(target_length - 1):
                    references = self.bundle.fiber_structure.get(current_file, [])
                    valid_refs = [ref for ref in references if ref in files]
                    
                    if valid_refs:
                        next_file = np.random.choice(valid_refs)
                        loop.append(next_file)
                        current_file = next_file
                    else:
                        # No valid references - try random connection
                        next_file = np.random.choice(files)
                        loop.append(next_file)
                        current_file = next_file
                
                # Close the loop
                loop.append(loop[0])
                
                # Validate loop quality
                if len(set(loop[:-1])) >= 2:  # At least 2 unique nodes
                    loops.append(loop)
                    loops_for_length += 1
                else:
                    failed_attempts += 1
            
            print(f"    Generated: {loops_for_length}/{self.min_loops_per_length} "
                  f"(attempts: {attempts_for_length})")
        
        success_rate = len(loops) / (len(loops) + failed_attempts) if (len(loops) + failed_attempts) > 0 else 0
        print(f"\n  Total loops generated: {len(loops)}")
        print(f"  Failed attempts: {failed_attempts}")
        print(f"  Success rate: {success_rate:.1%}")
        
        return loops
    
    def run_phase_1_1_validation(self) -> Dict:
        """Run Phase 1.1 enhanced statistical validation."""
        print("\n🎯 Phase 1.1: Enhanced Statistical Validation")
        print("="*60)
        
        # Get git commit for reproducibility
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               cwd=self.repo_path).decode().strip()
        except:
            git_commit = "unknown"
        
        results = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'git_commit_sha': git_commit,
                'theoretical_framework': 'Dual-Temporal Holonomy Theorem',
                'universal_scaling': self.E_over_hbar,
                'phase': '1.1 - Statistical Rigor Enhancement',
                'enhancement_features': [
                    'Extended loop lengths (3-7)',
                    'Systematic sampling (≥4 per length)',
                    'Bootstrap confidence intervals (1000 samples)',
                    'Formal hypothesis testing',
                    'Multiple comparison corrections'
                ]
            },
            'measurements': [],
            'enhanced_loop_statistics': {},
            'bootstrap_analysis': {},
            'validation_status': 'IN_PROGRESS'
        }
        
        if not self.bundle:
            print("⚠️  Repository structure not yet mapped")
            return results
        
        # Enhanced loop generation
        enhanced_loops = self._generate_enhanced_test_loops()
        
        if not enhanced_loops:
            print("❌ No valid loops generated")
            results['validation_status'] = 'FAILED'
            return results
        
        print(f"\n🔄 Testing {len(enhanced_loops)} enhanced discovery loops...")
        
        # Measure all loops
        for i, loop in enumerate(enhanced_loops):
            try:
                measurement = self.measure_discovery_loop_holonomy(loop)
                results['measurements'].append({
                    'loop_id': i,
                    'loop_path': measurement.loop_path,
                    'geometric_phase_magnitude': abs(measurement.geometric_phase),
                    'geometric_phase_argument': np.angle(measurement.geometric_phase),
                    'curvature': measurement.curvature,
                    'loop_length': len(measurement.loop_path) - 1,
                    'timestamp': measurement.timestamp.isoformat()
                })
            except Exception as e:
                print(f"Error measuring loop {i}: {e}")
                continue
        
        if not results['measurements']:
            print("❌ No successful measurements")
            results['validation_status'] = 'FAILED'
            return results
        
        # Enhanced loop statistics
        loop_lengths = [m['loop_length'] for m in results['measurements']]
        phases = [m['geometric_phase_magnitude'] for m in results['measurements']]
        
        results['enhanced_loop_statistics'] = {
            'total_loops': len(results['measurements']),
            'loop_length_distribution': {length: loop_lengths.count(length) 
                                       for length in self.enhanced_loop_lengths},
            'phase_statistics_by_length': {},
            'overall_phase_mean': np.mean(phases),
            'overall_phase_std': np.std(phases),
            'phase_range': (min(phases), max(phases))
        }
        
        # Phase statistics by loop length
        for length in self.enhanced_loop_lengths:
            length_phases = [m['geometric_phase_magnitude'] for m in results['measurements'] 
                           if m['loop_length'] == length]
            if length_phases:
                results['enhanced_loop_statistics']['phase_statistics_by_length'][length] = {
                    'count': len(length_phases),
                    'mean': np.mean(length_phases),
                    'std': np.std(length_phases),
                    'min': min(length_phases),
                    'max': max(length_phases)
                }
        
        # Bootstrap statistical analysis
        print("\n📊 Bootstrap Statistical Analysis")
        bootstrap_analyzer = BootstrapHolonomyAnalyzer(
            measurements=results['measurements'],
            n_bootstrap=self.bootstrap_samples,
            confidence_level=0.95,
            random_seed=42
        )
        
        bootstrap_results = bootstrap_analyzer.run_comprehensive_analysis()
        results['bootstrap_analysis'] = bootstrap_results
        
        # Overall validation status
        confirmed_hypotheses = bootstrap_results['summary']['confirmed_hypotheses']
        total_hypotheses = bootstrap_results['summary']['total_hypotheses']
        
        if confirmed_hypotheses >= 2:  # At least 2/3 Phase 1.1 hypotheses
            results['validation_status'] = 'COMPLETE'
        elif confirmed_hypotheses >= 1:
            results['validation_status'] = 'PARTIAL'
        else:
            results['validation_status'] = 'INSUFFICIENT'
        
        print(f"\n🎯 Phase 1.1 Validation Complete")
        print(f"Status: {results['validation_status']}")
        print(f"Enhanced loops: {len(results['measurements'])}")
        print(f"Hypotheses confirmed: {confirmed_hypotheses}/{total_hypotheses}")
        
        return results
    
    def export_phase_1_1_results(self, results: Dict, 
                                 base_path: Path = None) -> Tuple[Path, Path]:
        """Export Phase 1.1 results to structured files."""
        if base_path is None:
            base_path = self.repo_path / 'experiments' / 'fisher_rao_holonomy' / 'results' / 'phase1.1_statistical'
        
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Main session results
        session_file = base_path / f'session_{self.session_id}.json'
        with open(session_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Bootstrap analysis (separate file for detailed analysis)
        bootstrap_file = base_path / f'bootstrap_analysis_{self.session_id}.json'
        bootstrap_data = {
            'session_id': self.session_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'bootstrap_analysis': results.get('bootstrap_analysis', {})
        }
        
        with open(bootstrap_file, 'w') as f:
            json.dump(bootstrap_data, f, indent=2, default=str)
        
        print(f"\n💾 Phase 1.1 Results Exported")
        print(f"Session file: {session_file}")
        print(f"Bootstrap analysis: {bootstrap_file}")
        
        return session_file, bootstrap_file

def main():
    """Main Phase 1.1 experimental protocol execution."""
    print("🌊 Enhanced Fisher-Rao Holonomy Experimental Framework")
    print("Phase 1.1: Statistical Rigor Enhancement")
    print("Theoretical Foundation: Dual-Temporal Holonomy Theorem")
    print("="*60)
    
    # Initialize enhanced experiment
    repo_path = Path.cwd()
    experiment = EnhancedFisherRaoExperiment(repo_path)
    
    try:
        # Phase 1: Map repository structure
        print("\n🗺️ Phase 1: Repository Structure Mapping")
        bundle = experiment.map_repository_structure()
        
        # Phase 1.1: Enhanced statistical validation
        results = experiment.run_phase_1_1_validation()
        
        # Export results
        session_file, bootstrap_file = experiment.export_phase_1_1_results(results)
        
        # Summary
        print("\n🏁 Phase 1.1 Experimental Protocol Complete")
        print(f"Validation Status: {results.get('validation_status', 'UNKNOWN')}")
        print(f"Enhanced Measurements: {len(results.get('measurements', []))}")
        print(f"Bootstrap Analysis: {bootstrap_file.name}")
        
        # Success assessment
        if results.get('validation_status') == 'COMPLETE':
            print("\n✅ Phase 1.1 SUCCESS - Statistical rigor requirements met")
            print("Ready for Phase 2: Multi-Agent Coherence Experiments")
        elif results.get('validation_status') == 'PARTIAL':
            print("\n⚠️ Phase 1.1 PARTIAL - Some hypotheses confirmed, refinement needed")
        else:
            print("\n❌ Phase 1.1 INSUFFICIENT - Major issues detected, framework revision required")
        
        return results
        
    except Exception as e:
        print(f"\n💥 Phase 1.1 Experimental Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
