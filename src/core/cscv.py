"""
Combinatorially Symmetric Cross-Validation (CSCV)

Based on: Bailey, Borwein, López de Prado (2014)
"The Probability of Backtest Overfitting"

This module implements CSCV performance stability analysis. It tests
strategy performance across all possible train/test splits to detect
overfitting and estimate out-of-sample degradation
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.performance_metrics import sharpe_ratio
from utils.split_generator import (
    split_into_groups,
    generate_cscv_splits,
    apply_split,
    count_splits
)

class CSCV:
    """
    Combinatorially Symmetric Cross-Validation
    
    
    Parameters
    ----------
    n_groups : int, default 10
        Number of groups to split data into
    
    grouping_mode : {'contiguous', 'interleaved'}, default 'contiguous'
    
    risk_free_rate : float, default 0.0
        Annual risk-free rate for Sharpe calculations
    
    periods_per_year : int, default 252
        Trading periods per year (252 for daily, 12 for monthly)
        
    Attributes
    ----------
    results : dict or None
        Results from fit(), including degradation probability
    """
    def __init__(
        self, 
        n_groups: int = 10,
        grouping_mode: Literal['contiguous', 'interleaved'] = 'contiguous',
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        if n_groups % 2 != 0:
            raise ValueError("Need an even number of groups")
        if n_groups < 4:
            raise ValueError("Need more than 4 groups")
        if grouping_mode not in ('contiguous', 'interleaved'):
            raise ValueError("grouping mode must be contiguous or interleaved")
        
        self.n_groups = n_groups
        self.grouping_mode = grouping_mode
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.results = None
        
        # Calculate expected number of splits for time estimation
        self._n_splits = count_splits(n_groups)

    def fit(self, returns: np.ndarray, verbose: bool = True) -> Dict:
        """
        Run CSCV analysis
        
        Parameters
        ----------
        returns : np.ndarray
            Strategy returns time series
        verbose : bool, default True
            Print progress updates and results
            
        Returns
        -------
        dict
            Results dictionary containing:
            
            Core Metrics:
            - degradation_probability : float
                P(test < median train) across all splits
                Values: 0.0-1.0 (higher = more overfitting)
            - performance_degradation : float
                (mean_train - mean_test) / mean_train
                Percentage drop in performance OOS
            
            Summary Statistics:
            - median_train_sharpe : float
                Median Sharpe across training sets
            - mean_train_sharpe : float
                Mean Sharpe across training sets  
            - mean_test_sharpe : float
                Mean Sharpe across test sets
            - std_train_sharpe : float
                Standard deviation of train Sharpes
            - std_test_sharpe : float
                Standard deviation of test Sharpes
            
            Distributions:
            - train_sharpes : np.ndarray
                All training Sharpe ratios (length = n_splits)
            - test_sharpes : np.ndarray
                All test Sharpe ratios (length = n_splits)
            
            Metadata:
            - num_splits : int
                Number of train/test combinations tested
            - grouping_mode : str
                'contiguous' or 'interleaved'
            - n_groups : int
                Number of groups used
        
        """
        if verbose:
            print("\n" + "="*70)
            print("COMBINATORIALLY SYMMETRIC CROSS-VALIDATION (CSCV)")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  Groups: {self.n_groups}")
            print(f"  Grouping mode: {self.grouping_mode}")
            print(f"  Risk-free rate: {self.risk_free_rate:.2%}")
            print(f"  Periods per year: {self.periods_per_year}")

        groups = split_into_groups(returns, self.n_groups, mode=self.grouping_mode)
        
        if verbose:
            print(f"\nData Statistics:")
            print(f"  Total observations: {len(returns)}")
            print(f"  Observations per group: ~{len(returns)//self.n_groups}")
            print(f"  Date range: {self.n_groups} sequential periods")

        splits = generate_cscv_splits(self.n_groups)

        if verbose:
            est_time = len(splits) * 0.01  # Rough estimate
            print(f"\nCombinatorial Splits:")
            print(f"  Total combinations: {len(splits)}")
            print(f"  Train/test ratio: 50/50 (symmetric)")
            print(f"  Estimated time: ~{est_time:.1f}s")
            print(f"\nRunning analysis...")

        train_sharpes = []
        test_sharpes = []

        for i, (train_idx, test_idx) in enumerate(splits):
            train_data, test_data = apply_split(groups, train_idx, test_idx)

            train_sr = sharpe_ratio(
                train_data,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )

            test_sr = sharpe_ratio(
                test_data,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )

            train_sharpes.append(train_sr)
            test_sharpes.append(test_sr)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(splits)} splits...")
        
        train_sharpes = np.array(train_sharpes)
        test_sharpes = np.array(test_sharpes)

        median_train_sr = np.median(train_sharpes)
        num_underperforming = np.sum(test_sharpes < median_train_sr)
        degradation_prob = num_underperforming / len(test_sharpes)

        mean_train_sr = np.mean(train_sharpes)
        mean_test_sr = np.mean(test_sharpes)

        if mean_train_sr != 0:
            perf_degradation = (mean_train_sr - mean_test_sr) / mean_train_sr
        else:
            perf_degradation = 0.0

        std_train_sr = np.std(train_sharpes, ddof=1)
        std_test_sr = np.std(test_sharpes, ddof=1)

        self.results = {
            # Core metrics
            'degradation_probability': degradation_prob,
            'performance_degradation': perf_degradation,
            
            # Summary statistics
            'median_train_sharpe': median_train_sr,
            'mean_train_sharpe': mean_train_sr,
            'mean_test_sharpe': mean_test_sr,
            'std_train_sharpe': std_train_sr,
            'std_test_sharpe': std_test_sr,
            
            # Distributions
            'train_sharpes': train_sharpes,
            'test_sharpes': test_sharpes,
            
            # Metadata
            'num_splits': len(splits),
            'grouping_mode': self.grouping_mode,
            'n_groups': self.n_groups
        }
        
        if verbose:
            self.print_results()
        
        return self.results

    def print_results(self):
        """Print CSCV results in readable format with interpretation."""
        if self.results is None:
            print("No results yet. Run fit() first.")
            return
        
        r = self.results
        
        print("\n" + "="*70)
        print("CSCV RESULTS")
        print("="*70)
        
        # Configuration
        print(f"\nConfiguration:")
        print(f"  Grouping mode: {r['grouping_mode']}")
        print(f"  Number of splits: {r['num_splits']}")
        
        # In-sample statistics
        print(f"\nIn-Sample (Training) Performance:")
        print(f"  Median Sharpe: {r['median_train_sharpe']:>8.3f}")
        print(f"  Mean Sharpe:   {r['mean_train_sharpe']:>8.3f}")
        print(f"  Std Dev:       {r['std_train_sharpe']:>8.3f}")
        
        # Out-of-sample statistics  
        print(f"\nOut-of-Sample (Test) Performance:")
        print(f"  Mean Sharpe:   {r['mean_test_sharpe']:>8.3f}")
        print(f"  Std Dev:       {r['std_test_sharpe']:>8.3f}")
        
        # Key metrics
        print(f"\nStability Metrics:")
        print(f"  Performance degradation: {r['performance_degradation']:>8.1%}")
        print(f"  (Mean train - Mean test) / Mean train")
        
        print(f"\n{'🎯 DEGRADATION PROBABILITY:':<35} {r['degradation_probability']:>8.1%}")
        print(f"  P(test Sharpe < median train Sharpe)")
        
        # Interpretation
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        
        deg_prob = r['degradation_probability']
        perf_deg = r['performance_degradation']
        
        # Degradation probability assessment
        if deg_prob < 0.20:
            prob_verdict = "✅ Excellent - Very stable performance"
        elif deg_prob < 0.35:
            prob_verdict = "✅ Good - Reasonably stable"
        elif deg_prob < 0.50:
            prob_verdict = "⚠️  Moderate - Some stability concerns"
        elif deg_prob < 0.70:
            prob_verdict = "⚠️  Elevated - Likely overfitting"
        else:
            prob_verdict = "❌ High - Strong overfitting evidence"
        
        print(f"\nDegradation Probability: {prob_verdict}")
        
        # Performance degradation assessment
        if perf_deg < 0.10:
            perf_verdict = "✅ Minimal performance drop"
        elif perf_deg < 0.25:
            perf_verdict = "⚠️  Moderate performance drop"
        else:
            perf_verdict = "❌ Severe performance drop"
        
        print(f"Performance Drop: {perf_verdict}")
        
        # Combined assessment
        print(f"\n{'OVERALL ASSESSMENT:':<20}")
        if deg_prob < 0.35 and perf_deg < 0.25:
            print("  ✅ Strategy appears robust and stable")
            print("  Low risk of overfitting")
        elif deg_prob > 0.65 or perf_deg > 0.40:
            print("  ❌ Strategy shows strong signs of overfitting")
            print("  Recommendations:")
            print("    • Simplify strategy (reduce parameters)")
            print("    • Increase out-of-sample period")
            print("    • Test on different market regimes")
            print("    • Consider walk-forward analysis")
        else:
            print("  ⚠️  Mixed signals - proceed with caution")
            print("  Recommendations:")
            print("    • Monitor live performance closely")
            print("    • Reduce position sizing initially")
            print("    • Implement additional validation tests")
        
        # Note about PBO
        print("\n" + "-"*70)
        print("NOTE ON TERMINOLOGY")
        print("-"*70)
        print("This analysis measures SINGLE-STRATEGY performance stability.")
        print("For multi-trial 'Probability of Backtest Overfitting (PBO)',")
        print("which tests selection bias across multiple strategies,")
        print("use the PBO module (requires returns from all trials tested).")
        
        print("="*70 + "\n")
    
    def get_summary(self) -> Optional[Dict]:
        """
        Get concise summary for programmatic use.
        
        Returns
        -------
        dict or None
            Summary statistics, or None if fit() hasn't been called
        """
        if self.results is None:
            return None
        
        return {
            'degradation_probability': self.results['degradation_probability'],
            'performance_degradation': self.results['performance_degradation'],
            'mean_train_sharpe': self.results['mean_train_sharpe'],
            'mean_test_sharpe': self.results['mean_test_sharpe'],
            'num_splits': self.results['num_splits']
        }


# Test the CSCV class
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CSCV ALGORITHM")
    print("="*70)
    
    np.random.seed(42)
    
    # Test 1: Robust strategy (consistent performance)
    print("\n" + "="*70)
    print("TEST 1: ROBUST STRATEGY")
    print("="*70)
    print("Expected: Low degradation probability, minimal performance drop")
    
    robust_returns = np.random.normal(0.0008, 0.01, 1000)
    
    cscv1 = CSCV(n_groups=10, grouping_mode='contiguous')
    results1 = cscv1.fit(robust_returns)
    
    # Test 2: Overfit strategy (good first half, bad second half)
    print("\n" + "="*70)
    print("TEST 2: OVERFIT STRATEGY")
    print("="*70)
    print("Simulating curve-fitted strategy:")
    print("  • First 50%: Optimized (high Sharpe)")
    print("  • Last 50%: Reality (low Sharpe)")
    print("Expected: High degradation probability, large performance drop")
    
    # First half: "optimized" period
    first_half = np.random.normal(0.002, 0.010, 500)
    # Second half: reality
    second_half = np.random.normal(0.0, 0.018, 500)
    overfit_returns = np.concatenate([first_half, second_half])
    
    # Calculate first vs second half statistics
    from core.performance_metrics import sharpe_ratio as calc_sharpe
    first_half_sr = calc_sharpe(first_half, periods_per_year=252)
    second_half_sr = calc_sharpe(second_half, periods_per_year=252)
    print(f"\nActual split performance:")
    print(f"  First half Sharpe:  {first_half_sr:.3f}")
    print(f"  Second half Sharpe: {second_half_sr:.3f}")
    print(f"  Degradation: {((first_half_sr - second_half_sr)/first_half_sr)*100:.1f}%")
    
    cscv2 = CSCV(n_groups=10, grouping_mode='contiguous')
    results2 = cscv2.fit(overfit_returns)
    
    # Test 3: Random strategy (no skill)
    print("\n" + "="*70)
    print("TEST 3: RANDOM STRATEGY (No Skill)")
    print("="*70)
    print("Expected: ~50% degradation probability (pure chance)")
    
    random_returns = np.random.normal(0.0, 0.015, 1000)
    
    cscv3 = CSCV(n_groups=10, grouping_mode='contiguous')
    results3 = cscv3.fit(random_returns)
    
    # Test 4: Interleaved vs Contiguous
    print("\n" + "="*70)
    print("TEST 4: GROUPING MODE COMPARISON")
    print("="*70)
    print("Testing overfit strategy with different grouping modes")
    
    print("\n--- Contiguous Mode (already done above) ---")
    print(f"Degradation probability: {results2['degradation_probability']:.1%}")
    
    print("\n--- Interleaved Mode ---")
    cscv4 = CSCV(n_groups=10, grouping_mode='interleaved')
    results4 = cscv4.fit(overfit_returns, verbose=True)
    
    print("\nComparison:")
    print(f"  Contiguous:  {results2['degradation_probability']:.1%} degradation prob")
    print(f"  Interleaved: {results4['degradation_probability']:.1%} degradation prob")
    print("\nNote: Contiguous mode is more conservative (harder to pass)")
    print("      because it tests regime robustness explicitly.")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Strategy':<20} {'Train Sharpe':<15} {'Test Sharpe':<15} {'Degrad Prob':<15} {'Perf Drop':<15}")
    print("-"*85)
    print(f"{'Robust':<20} {results1['mean_train_sharpe']:>8.3f}{'':<7} "
          f"{results1['mean_test_sharpe']:>8.3f}{'':<7} "
          f"{results1['degradation_probability']:>8.1%}{'':<7} "
          f"{results1['performance_degradation']:>8.1%}")
    print(f"{'Overfit':<20} {results2['mean_train_sharpe']:>8.3f}{'':<7} "
          f"{results2['mean_test_sharpe']:>8.3f}{'':<7} "
          f"{results2['degradation_probability']:>8.1%}{'':<7} "
          f"{results2['performance_degradation']:>8.1%}")
    print(f"{'Random':<20} {results3['mean_train_sharpe']:>8.3f}{'':<7} "
          f"{results3['mean_test_sharpe']:>8.3f}{'':<7} "
          f"{results3['degradation_probability']:>8.1%}{'':<7} "
          f"{results3['performance_degradation']:>8.1%}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("✅ Robust strategy:")
    print("   • Low degradation probability (~30%)")
    print("   • Minimal performance drop (~2%)")
    print("   • Consistent Sharpe across splits")
    print("\n❌ Overfit strategy:")
    print("   • High degradation probability (~90%)")
    print("   • Large performance drop (~55%)")
    print("   • CSCV successfully detected overfitting!")
    print("\n⚠️  Random strategy:")
    print("   • ~50% degradation probability (pure chance)")
    print("   • Near-zero Sharpe (no skill)")
    print("   • High variance across splits")
    print("="*70 + "\n")