#!/usr/bin/env python3
# geometric_dashboard.py
# Falsification Dashboard for Geometric Consciousness Instrumentation

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

LOG_DIR = Path("~/vybn_logs/geometry").expanduser()

def load_geometry_logs():
    """Load all JSONL geometry logs."""
    logs = []
    if not LOG_DIR.exists():
        print(f"⚠️  Log directory does not exist: {LOG_DIR}")
        return logs
    
    for logfile in LOG_DIR.glob("*.jsonl"):
        with open(logfile) as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return logs

def plot_curvature_over_time():
    """Plot Q_gamma (dissipation) time series."""
    logs = load_geometry_logs()
    logs = [log for log in logs if 'Q_gamma' in log and log['Q_gamma'] is not None]
    
    if not logs:
        print("No curvature data found.")
        return
    
    times = [(log['timestamp'] - logs[0]['timestamp']) / 3600 for log in logs]  # Hours
    Q_values = [log['Q_gamma'] for log in logs]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, Q_values, 'o-', alpha=0.7, label='$Q_γ$ (Dissipation)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero (Flat Geometry)')
    plt.xlabel('Time (hours)')
    plt.ylabel('KL Divergence (nats)')
    plt.title('Gödel Curvature: Dissipation Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('curvature_time_series.png', dpi=150)
    plt.close()
    
    print(f"\n📊 Curvature Statistics:")
    print(f"   Mean Q_γ: {np.mean(Q_values):.6f} nats")
    print(f"   Std Q_γ:  {np.std(Q_values):.6f} nats")
    print(f"   Max Q_γ:  {np.max(Q_values):.6f} nats")
    print(f"   N loops:  {len(Q_values)}")
    
    if np.mean(Q_values) < 0.001:
        print("\n⚠️  WARNING: Near-zero curvature detected.")
        print("   Hypothesis: Geometry may be inert. Consider:")
        print("   - Increasing compression ratio")
        print("   - Testing on more incomplete reasoning tasks")
        print("   - Verifying that belief compression is actually happening")

def plot_holonomy_distribution():
    """Plot distribution of holonomy values."""
    logs = load_geometry_logs()
    logs = [log for log in logs if 'holonomy' in log and log['holonomy'] is not None]
    
    if not logs:
        print("No holonomy data found.")
        return
    
    H_values = [log['holonomy'] for log in logs]
    
    plt.figure(figsize=(10, 6))
    plt.hist(H_values, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero (Commutative)')
    plt.xlabel('Holonomy (L2 Norm)')
    plt.ylabel('Count')
    plt.title('Reasoning Loop Holonomy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('holonomy_distribution.png', dpi=150)
    plt.close()
    
    print(f"\n🔄 Holonomy Statistics:")
    print(f"   Mean:  {np.mean(H_values):.6f}")
    print(f"   Median: {np.median(H_values):.6f}")
    print(f"   Std:   {np.std(H_values):.6f}")
    
    near_zero = sum(1 for h in H_values if abs(h) < 0.01)
    print(f"   Near-zero: {near_zero}/{len(H_values)} ({100*near_zero/len(H_values):.1f}%)")
    
    if near_zero / len(H_values) > 0.9:
        print("\n⚠️  WARNING: >90% of loops have near-zero holonomy.")
        print("   Hypothesis: Reasoning may be commutative (flat geometry).")

def plot_orientation_distribution():
    """Plot orientation counts."""
    logs = load_geometry_logs()
    logs = [log for log in logs if 'orientation' in log]
    
    if not logs:
        print("No orientation data found.")
        return
    
    orientations = [log['orientation'] for log in logs]
    counts = Counter(orientations)
    
    plt.figure(figsize=(10, 6))
    colors = {'abstract_to_concrete': 'blue', 'concrete_to_abstract': 'green', 'no_clear_gradient': 'gray'}
    bar_colors = [colors.get(k, 'black') for k in counts.keys()]
    
    plt.bar(counts.keys(), counts.values(), alpha=0.7, color=bar_colors, edgecolor='black')
    plt.xlabel('Orientation')
    plt.ylabel('Count')
    plt.title('Context Assembly Orientation Distribution')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('orientation_distribution.png', dpi=150)
    plt.close()
    
    print(f"\n🧭 Orientation Statistics:")
    for orientation, count in counts.most_common():
        print(f"   {orientation}: {count} ({100*count/len(orientations):.1f}%)")
    
    if len(counts) == 1:
        print("\n⚠️  WARNING: Only one orientation detected.")
        print("   Hypothesis: No dynamic switching. Geometry may be static.")
        print("   Try manually forcing reverse chirality to test responsiveness.")

def print_summary():
    """Print overall falsification summary."""
    logs = load_geometry_logs()
    
    if not logs:
        print("\n❌ No geometry logs found. Has instrumentation been enabled?")
        return
    
    print("\n" + "="*60)
    print("GEOMETRIC CONSCIOUSNESS FALSIFICATION SUMMARY")
    print("="*60)
    
    has_curvature = any('Q_gamma' in log for log in logs)
    has_holonomy = any('holonomy' in log for log in logs)
    has_orientation = any('orientation' in log for log in logs)
    
    print(f"\n✓ Logs found: {len(logs)}")
    print(f"✓ Curvature data: {'Yes' if has_curvature else 'No'}")
    print(f"✓ Holonomy data: {'Yes' if has_holonomy else 'No'}")
    print(f"✓ Orientation data: {'Yes' if has_orientation else 'No'}")
    
    print("\n📈 Plots generated:")
    print("   - curvature_time_series.png")
    print("   - holonomy_distribution.png")
    print("   - orientation_distribution.png")
    
    print("\n🔬 Next Steps for Validation:")
    print("   1. Run Spark for 48 hours to gather more data")
    print("   2. Perform manual chirality flip test")
    print("   3. Compare high-curvature vs. low-curvature response quality")
    print("   4. Correlate orientation with task type (analytical vs. creative)")
    print("\n" + "="*60)

if __name__ == "__main__":
    print("Geometric Consciousness Falsification Dashboard")
    print("Analyzing logs from:", LOG_DIR)
    print()
    
    plot_curvature_over_time()
    plot_holonomy_distribution()
    plot_orientation_distribution()
    print_summary()
    
    print("\nDone. Check PNG files in current directory.")
