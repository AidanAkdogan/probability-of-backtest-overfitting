"""
Backtest Overfitting Detector - PBO Analysis
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.performance_metrics import calculate_all_metrics
from core.pbo import PBO


def load_data(filepath: str):
    df = pd.read_csv(filepath)
    
    # remove date column if present
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
    
    # validate multi-trial format
    if df.shape[1] < 2:
        raise ValueError(
            f"PBO requires multiple strategies (found {df.shape[1]} column).\n"
            f"Expected CSV format: date, strategy_1, strategy_2, ..., strategy_N"
        )
    
    print(f"Loaded {df.shape[0]} observations × {df.shape[1]} strategies")
    
    return df.values


def analyze_pbo(returns_matrix: np.ndarray, filepath: str):
    print("\n" + "="*70)
    print(f"PBO ANALYSIS: {Path(filepath).name}")
    print("="*70)
    
    n_obs, n_trials = returns_matrix.shape
    print(f"\nDataset: {n_obs} days, {n_trials} strategies")
    
    # summary stats
    print("\n" + "="*70)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*70)
    
    sharpes = []
    for i in range(n_trials):
        trial_returns = returns_matrix[:, i]
        metrics = calculate_all_metrics(trial_returns)
        sharpes.append(metrics['sharpe_ratio'])
    
    sharpes = np.array(sharpes)
    
    print(f"\nSharpe Ratios across {n_trials} strategies:")
    print(f"  Mean:      {np.mean(sharpes):>8.3f}")
    print(f"  Median:    {np.median(sharpes):>8.3f}")
    print(f"  Std Dev:   {np.std(sharpes):>8.3f}")
    print(f"  Min:       {np.min(sharpes):>8.3f}")
    print(f"  Max:       {np.max(sharpes):>8.3f}")
    print(f"  Range:     {np.ptp(sharpes):>8.3f}")
    
    pbo = PBO(n_groups=10)
    pbo_results = pbo.fit(returns_matrix, verbose=True)
    
    # Actionable recommendations
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    pbo_val = pbo_results['pbo']
    
    if pbo_val < 0.30:
        print("\n -- LOW OVERFITTING RISK")
        print("   • Selection bias is minimal")
        print("   • Best backtest likely translates to live performance")
        print("   • Safe to proceed with top-ranked strategy")
        print("   • Still monitor live performance closely")
        
    elif pbo_val < 0.50:
        print("\n -- MODERATE OVERFITTING RISK")
        print("   • Some selection bias detected")
        print("   • Best backtest may degrade OOS")
        print("\n   Recommendations:")
        print(f"   • Reduce parameters tested (currently {n_trials})")
        print("   • Use walk-forward validation")
        
    else:
        print("\n -- HIGH OVERFITTING RISK - DO NOT TRADE")
        print("   • Severe selection bias detected")
        print("   • Best backtest is likely dumb luck")
        print("   • High probability of failure in live trading")
        print("\n   Required actions:")
        print("   • Drastically reduce parameter count")
        print("   • Simplify strategy logic")
        print("   • Add economic/fundamental rationale")
        print("   • Consider abandoning this approach entirely (I would!)")
    
    # best strategies
    print("\n" + "="*70)
    print("TOP 5 STRATEGIES (Ranked by Sharpe Ratio)")
    print("="*70)
    
    sorted_indices = np.argsort(sharpes)[::-1][:min(5, n_trials)]
    
    print(f"\n{'Rank':<6} {'Strategy':<12} {'Sharpe':<10} {'Assessment':<40}")
    print("-"*70)
    
    for rank, idx in enumerate(sorted_indices, 1):
        strategy_name = f'Strategy_{idx+1}'
        sharpe_val = sharpes[idx]
        
        # Assessment based on rank and PBO
        if rank == 1:
            if pbo_val > 0.50:
                assessment = "XXX  High overfitting risk"
            elif pbo_val > 0.30:
                assessment = "⚠️  Moderate risk, use caution"
            else:
                assessment = "-- Appears robust"
        else:
            assessment = ""
        
        print(f"{rank:<6} {strategy_name:<12} {sharpe_val:>6.3f}{'':<4} {assessment:<40}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nProbability of Backtest Overfitting: {pbo_val:.1%}")
    print(f"Strategies tested: {n_trials}")
    print(f"Best Sharpe: {np.max(sharpes):.3f}")
    print(f"Expected live Sharpe: {pbo_results['mean_test_sharpe']:.3f} "
          f"({(pbo_results['performance_degradation']*100):.0f}% degradation)")
    
    if pbo_val < 0.30:
        print("\n-- Decision: SAFE TO TRADE (with monitoring)")
    elif pbo_val < 0.50:
        print("\n⚠️  Decision: PROCEED WITH CAUTION")
    else:
        print("\nXXX Decision: DO NOT TRADE (PLEASE)")
    
    print("\n")


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description='Detect backtest overfitting using PBO (Bailey et al. 2014)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to CSV file with multi-strategy returns'
    )
    
    parser.add_argument(
        '--groups',
        type=int,
        default=10,
        help='Number of CSCV groups (default: 10, range: 6-12)'
    )
    
    args = parser.parse_args()
    
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    
    try:
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        print(f"File: {args.filepath}")
        
        returns_matrix = load_data(args.filepath)
        analyze_pbo(returns_matrix, args.filepath)
        
    except ValueError as e:
        print(f"\n Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()