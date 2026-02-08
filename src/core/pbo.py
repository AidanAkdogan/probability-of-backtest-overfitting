"""
Probability of Backtest Overfitting

We hone in on:
- Sharpe of strategies picked as "best IS" 
- Sharpe of those SAME strategies when tested OOS
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Literal
from scipy.special import comb
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.performance_metrics import sharpe_ratio
from utils.split_generator import (
    split_into_groups,
    generate_cscv_splits,
    apply_split,
    count_splits
)


class PBO:
    """
    Probability of Backtest Overfitting (Bailey et al. 2014)
    
    Measures selection bias when choosing the best strategy from multiple trials
    """
    
    def __init__(
        self,
        n_groups: int = 10,
        grouping_mode: Literal['contiguous', 'interleaved'] = 'contiguous',
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        if n_groups % 2 != 0:
            raise ValueError(f"n_groups must be even")
        if n_groups < 4:
            raise ValueError(f"n_groups must be >= 4")
        if grouping_mode not in ('contiguous', 'interleaved'):
            raise ValueError(f"grouping_mode must be 'contiguous' or 'interleaved'")
        
        self.n_groups = n_groups
        self.grouping_mode = grouping_mode
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.results = None
        self._n_splits = count_splits(n_groups)
    
    def fit(self, returns_matrix: np.ndarray, verbose: bool = True) -> Dict:
        """Run PBO analysis."""
        if returns_matrix.ndim != 2:
            raise ValueError(f"returns_matrix must be 2D")
        
        n_obs, n_trials = returns_matrix.shape
        
        if n_trials < 2:
            raise ValueError(f"Need at least 2 trials for PBO")
        if n_obs < 100:
            raise ValueError(f"Need at least 100 observations")
        
        if verbose:
            print("\n" + "="*70)
            print("PROBABILITY OF BACKTEST OVERFITTING (PBO)")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  Strategy trials: {n_trials}")
            print(f"  Observations: {n_obs}")
            print(f"  CSCV groups: {self.n_groups}")
            print(f"  Grouping mode: {self.grouping_mode}")
            print(f"  Total splits: {self._n_splits}")
            est_time = self._n_splits * n_trials * 0.001
            print(f"\nThis will test {self._n_splits:,} train/test combinations...")
            print(f"Estimated time: ~{est_time:.1f}s")
        
        
        oos_ranks_of_best_is = []
        best_is_sharpes = []     
        best_is_oos_sharpes = [] 
        all_is_sharpes = []
        all_oos_sharpes = []
        
        splits = generate_cscv_splits(self.n_groups)
        
        
        # each CSCV split
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            is_sharpes = np.zeros(n_trials)
            oos_sharpes = np.zeros(n_trials)
            
            # each strategy
            for trial_idx in range(n_trials):
                trial_returns = returns_matrix[:, trial_idx]
                groups = split_into_groups(trial_returns, self.n_groups, mode=self.grouping_mode)
                train_data, test_data = apply_split(groups, train_idx, test_idx)
                
                is_sharpe = sharpe_ratio(train_data, risk_free_rate=self.risk_free_rate, 
                                        periods_per_year=self.periods_per_year)
                oos_sharpe = sharpe_ratio(test_data, risk_free_rate=self.risk_free_rate,
                                         periods_per_year=self.periods_per_year)
                
                is_sharpes[trial_idx] = is_sharpe
                oos_sharpes[trial_idx] = oos_sharpe
            
            # Find best in-sample strategy
            best_is_idx = np.argmax(is_sharpes)
            
            # Track its performance
            best_is_sharpes.append(is_sharpes[best_is_idx])      # Its IS Sharpe
            best_is_oos_sharpes.append(oos_sharpes[best_is_idx]) # Its OOS Sharpe
            
            # Get its out-of-sample rank
            oos_ranks = rankdata(-oos_sharpes)
            oos_rank_of_best_is = oos_ranks[best_is_idx]
            
            oos_ranks_of_best_is.append(oos_rank_of_best_is)
            all_is_sharpes.append(is_sharpes)
            all_oos_sharpes.append(oos_sharpes)
            
        
        # arrays
        oos_ranks_of_best_is = np.array(oos_ranks_of_best_is)
        best_is_sharpes = np.array(best_is_sharpes)
        best_is_oos_sharpes = np.array(best_is_oos_sharpes)
        all_is_sharpes = np.array(all_is_sharpes)
        all_oos_sharpes = np.array(all_oos_sharpes)
        
        # Calculate PBO
        bottom_half_threshold = n_trials / 2
        num_bottom_half = np.sum(oos_ranks_of_best_is > bottom_half_threshold)
        pbo = num_bottom_half / len(oos_ranks_of_best_is)
        
        bottom_quartile_threshold = n_trials * 0.75
        num_bottom_quartile = np.sum(oos_ranks_of_best_is > bottom_quartile_threshold)
        pbo_test = num_bottom_quartile / len(oos_ranks_of_best_is)
        
        median_rank_ratio = np.median(oos_ranks_of_best_is) / n_trials
        
        num_worst = np.sum(oos_ranks_of_best_is == n_trials)
        best_is_worst_oos_prob = num_worst / len(oos_ranks_of_best_is)
        
        if median_rank_ratio > 0 and median_rank_ratio < 1:
            phi = np.log(median_rank_ratio / (1 - median_rank_ratio))
        else:
            phi = np.nan
        
        
        mean_train_sharpe = np.mean(best_is_sharpes)      
        mean_test_sharpe = np.mean(best_is_oos_sharpes)
        
        if mean_train_sharpe != 0:
            performance_degradation = (mean_train_sharpe - mean_test_sharpe) / mean_train_sharpe
        else:
            performance_degradation = 0.0
        
        self.results = {
            'pbo': pbo,
            'pbo_test': pbo_test,
            'phi': phi,
        
            'median_rank_ratio': median_rank_ratio,
            'best_is_worst_oos_prob': best_is_worst_oos_prob,
            
            'mean_train_sharpe': mean_train_sharpe,     
            'mean_test_sharpe': mean_test_sharpe,      
            'performance_degradation': performance_degradation,
            
            'oos_ranks_of_best_is': oos_ranks_of_best_is,
            'best_is_sharpes': best_is_sharpes,          
            'best_is_oos_sharpes': best_is_oos_sharpes,
            'is_sharpes_all': all_is_sharpes,
            'oos_sharpes_all': all_oos_sharpes,
            
            'num_trials': n_trials,
            'num_splits': len(splits),
            'grouping_mode': self.grouping_mode
        }
        
        if verbose:
            self.print_results()
        
        return self.results
    
    def print_results(self):
        if self.results is None:
            print("No results yet. Run fit() first.")
            return
        
        r = self.results
        
        print("\n" + "="*70)
        print("PBO RESULTS")
        print("="*70)
        
        print(f"\nDataset:")
        print(f"  Strategy trials tested: {r['num_trials']}")
        print(f"  CSCV splits: {r['num_splits']}")
        
        print(f"\nPerformance of Selected Strategies:")
        print(f"  Mean IS Sharpe (when selected):  {r['mean_train_sharpe']:.3f}")
        print(f"  Mean OOS Sharpe (same strategies): {r['mean_test_sharpe']:.3f}")
        print(f"  Performance degradation:          {r['performance_degradation']:.1%}")
        print(f"    (How much the 'best' strategies degrade out-of-sample)")
        
        print(f"\nSelection Bias Metrics:")
        print(f"  Median OOS rank ratio: {r['median_rank_ratio']:.3f}")
        print(f"    (0 = best IS always best OOS, 0.5 = random, 1 = worst)")
        
        print(f"\n{'** PROBABILITY OF BACKTEST OVERFITTING (PBO):':<50} {r['pbo']:.1%}")
        print(f"  P(best in-sample strategy ranks in bottom 50% out-of-sample)")
        
        print(f"\n{'   PBO (bottom quartile):':<50} {r['pbo_test']:.1%}")
        print(f"  P(best in-sample strategy ranks in bottom 25% out-of-sample)")
        
        print(f"\n{'   P(best IS is worst OOS):':<50} {r['best_is_worst_oos_prob']:.1%}")
        
        if not np.isnan(r['phi']):
            print(f"\n   Phi statistic: {r['phi']:.3f}")
        
        # my interpretation
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        pbo = r['pbo']
        deg = r['performance_degradation']
        
        if pbo < 0.20:
            verdict = " -- Very low overfitting"
        elif pbo < 0.35:
            verdict = "-- Low overfitting risk"
        elif pbo < 0.50:
            verdict = "-- Noticeable overfitting"
        elif pbo < 0.70:
            verdict = "-- Severe overfitting"
        else:
            verdict = "-- Extreme overfitting"
        
        print(f"\nPBO: {verdict}")
        
        if deg > 0:
            print(f"Performance Degradation: {deg:.1%}")
            print(f"  Selected strategies degrade by {deg:.0%} out-of-sample")
        else:
            print(f"This is an irregularity")
        
        print("="*70 + "\n")
    
    def get_summary(self) -> Optional[Dict]:
        if self.results is None:
            return None
        
        return {
            'pbo': self.results['pbo'],
            'mean_train_sharpe': self.results['mean_train_sharpe'],
            'mean_test_sharpe': self.results['mean_test_sharpe'],
            'performance_degradation': self.results['performance_degradation'],
            'num_trials': self.results['num_trials']
        }

