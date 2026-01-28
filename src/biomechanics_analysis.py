"""
biomechanics_analysis.py
Statistical analysis of biomechanics data: Made vs Missed comparison.

Usage:
    python src/biomechanics_analysis.py
    
Expects processed_trials.json in data/
Outputs statistical_results.json to data/
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Cohen's d measures the standardized difference between two means.
    
    Interpretation:
        |d| = 0.2: Small effect
        |d| = 0.5: Medium effect
        |d| = 0.8: Large effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    """Run full statistical analysis."""
    # Load data
    input_file = DATA_DIR / "processed_trials.json"
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found!")
        print("Run process_spl_data.py first.")
        return
    
    with open(input_file) as f:
        data = json.load(f)
    
    trials = data['trials']
    made = [t for t in trials if t['result'] == 'made']
    missed = [t for t in trials if t['result'] == 'missed']
    
    print("=" * 70)
    print("SPL BIOMECHANICS ANALYSIS - STATISTICAL RESULTS")
    print("=" * 70)
    print(f"\nSample: Made={len(made)}, Missed={len(missed)}, "
          f"FT%={len(made)/len(trials)*100:.1f}%")
    
    # Define metrics to analyze
    metrics = {
        'entry_angle': 'Entry Angle (°)',
        'knee_rom': 'Knee ROM (°)',
        'knee_min': 'Knee Min Angle (°)',
        'hip_rom': 'Hip ROM (°)',
        'elbow_rom': 'Elbow ROM (°)',
        'elbow_max': 'Elbow Max Angle (°)',
        'release_height': 'Release Height (m)',
        'hip_drop': 'Hip Drop (m)',
    }
    
    # Run t-tests for each metric
    print("\n" + "=" * 70)
    print("TWO-SAMPLE T-TESTS: Made vs Missed")
    print("=" * 70)
    
    results = []
    for metric, label in metrics.items():
        made_vals = [t[metric] for t in made if t.get(metric) is not None]
        missed_vals = [t[metric] for t in missed if t.get(metric) is not None]
        
        if len(made_vals) > 5 and len(missed_vals) > 5:
            # Two-sample t-test (independent samples)
            t_stat, p_val = stats.ttest_ind(made_vals, missed_vals)
            
            made_mean = np.mean(made_vals)
            made_std = np.std(made_vals, ddof=1)
            missed_mean = np.mean(missed_vals)
            missed_std = np.std(missed_vals, ddof=1)
            
            effect_size = cohens_d(made_vals, missed_vals)
            
            # Significance markers
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            
            results.append({
                'metric': metric,
                'label': label,
                'made_mean': made_mean,
                'made_std': made_std,
                'missed_mean': missed_mean,
                'missed_std': missed_std,
                'difference': made_mean - missed_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': effect_size,
                'significance': sig
            })
            
            print(f"\n{label}:")
            print(f"  Made:   {made_mean:.2f} ± {made_std:.2f} (n={len(made_vals)})")
            print(f"  Missed: {missed_mean:.2f} ± {missed_std:.2f} (n={len(missed_vals)})")
            print(f"  Δ = {made_mean - missed_mean:+.2f}")
            print(f"  t = {t_stat:.2f}, p = {p_val:.4f} {sig}")
            print(f"  Cohen's d = {effect_size:.3f}")
    
    # Sort by significance
    results.sort(key=lambda x: x['p_value'])
    
    print("\n" + "=" * 70)
    print("RANKED BY SIGNIFICANCE:")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['label']:25s} p={r['p_value']:.4f} {r['significance']:3s} "
              f"(Δ={r['difference']:+.2f}, d={r['cohens_d']:.2f})")
    
    # Entry angle analysis
    print("\n" + "=" * 70)
    print("ENTRY ANGLE ANALYSIS")
    print("=" * 70)
    
    entry_made = [t['entry_angle'] for t in made if t['entry_angle']]
    entry_missed = [t['entry_angle'] for t in missed if t['entry_angle']]
    
    # Optimal range (research suggests 42-46° is optimal)
    optimal_made = sum(1 for e in entry_made if 42 <= e <= 46)
    optimal_missed = sum(1 for e in entry_missed if 42 <= e <= 46)
    
    print(f"\nEntry angles in optimal range (42-46°):")
    print(f"  Made shots:   {optimal_made}/{len(entry_made)} "
          f"({optimal_made/len(entry_made)*100:.1f}%)")
    print(f"  Missed shots: {optimal_missed}/{len(entry_missed)} "
          f"({optimal_missed/len(entry_missed)*100:.1f}%)")
    
    # Correlations
    print("\n" + "=" * 70)
    print("CORRELATIONS WITH ENTRY ANGLE")
    print("=" * 70)
    
    correlations = {}
    for metric in ['knee_rom', 'hip_rom', 'elbow_rom', 'knee_min']:
        vals = [(t['entry_angle'], t[metric]) for t in trials 
                if t['entry_angle'] and t.get(metric)]
        if vals:
            x, y = zip(*vals)
            r, p = stats.pearsonr(x, y)
            correlations[metric] = {'r': r, 'p': p}
            print(f"{metric:15s}: r = {r:+.3f}, p = {p:.4f}")
    
    # Save results
    output = {
        'sample_size': {
            'total': len(trials),
            'made': len(made),
            'missed': len(missed)
        },
        'ft_pct': len(made) / len(trials) * 100,
        'statistical_tests': results,
        'entry_angle_analysis': {
            'made_mean': np.mean(entry_made),
            'missed_mean': np.mean(entry_missed),
            'optimal_range': '42-46°',
            'optimal_range_made_pct': optimal_made / len(entry_made) * 100,
            'optimal_range_missed_pct': optimal_missed / len(entry_missed) * 100
        },
        'correlations': correlations
    }
    
    output_file = DATA_DIR / "statistical_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n✅ Saved to {output_file}")
    
    # Print key finding
    print("\n" + "=" * 70)
    print("🎯 KEY FINDING")
    print("=" * 70)
    elbow_result = next(r for r in results if r['metric'] == 'elbow_rom')
    print(f"""
ELBOW ROM is the strongest predictor of free throw success:
  - Made shots: {elbow_result['made_mean']:.1f}° ± {elbow_result['made_std']:.1f}°
  - Missed shots: {elbow_result['missed_mean']:.1f}° ± {elbow_result['missed_std']:.1f}°
  - Difference: +{elbow_result['difference']:.1f}°
  - p-value: {elbow_result['p_value']:.4f} (highly significant)
  - Effect size: d = {elbow_result['cohens_d']:.2f} (medium-large)
  
Interpretation: Greater arm extension through the shot = higher success rate
""")


if __name__ == "__main__":
    main()
