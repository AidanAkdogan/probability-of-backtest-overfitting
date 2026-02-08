"""
Generate PBO test datasets

We directly generate strategy returns with realistic characteristics, instead of generating prices. This circumvents getting unreal sharpe ratios.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from core.performance_metrics import sharpe_ratio

np.random.seed(42)
N_DAYS = 1000

# =============================================================================
# Example 1: The moving average parameter sweep
# =============================================================================

print("\n" + "="*70)
print("1. MA PARAMETER SWEEP (20 strategies)")
print("="*70)

dates = pd.date_range('2020-01-01', periods=N_DAYS, freq='B')
ma_returns = pd.DataFrame({'date': dates})

base_mean = 0.0002  # ~5% annual return (fairly modest)
base_vol = 0.012    # ~19% annual volatility

print(f"\nGenerating 20 MA strategies with:")
print(f"  Target mean: {base_mean * 252:.1%} annual")
print(f"  Target vol:  {base_vol * np.sqrt(252):.1%} annual")
print(f"  Expected Sharpe: {(base_mean * 252) / (base_vol * np.sqrt(252)):.2f}")

for i in range(20):
    # Each strategy has slightly different characteristics
    # (simulating different MA periods having different performance)
    
    # Wiggle the mean and volatility slightly
    strategy_mean = base_mean + np.random.normal(0, 0.0001)
    strategy_vol = base_vol + np.random.normal(0, 0.001)
    
    # get returns
    strategy_returns = np.random.normal(strategy_mean, strategy_vol, N_DAYS)
    
    # Add some autocorrelation (strategies aren't pure random)
    # This simulates the "momentum" effect trends
    for j in range(1, N_DAYS):
        # Small correlation with previous day
        strategy_returns[j] += 0.15 * strategy_returns[j-1]
    
    ma_returns[f'ma_{(i+1)*10}'] = strategy_returns


print(f"\nGenerated strategy Sharpe ratios:")
sharpes_ma = []
for col in ma_returns.columns[1:]:
    sr = sharpe_ratio(ma_returns[col].values)
    sharpes_ma.append(sr)

sharpes_ma = np.array(sharpes_ma)

print(f"  Mean:   {np.mean(sharpes_ma):.3f}")
print(f"  Median: {np.median(sharpes_ma):.3f}")
print(f"  Std:    {np.std(sharpes_ma):.3f}")
print(f"  Min:    {np.min(sharpes_ma):.3f}")
print(f"  Max:    {np.max(sharpes_ma):.3f}")
print(f"  Range:  {np.ptp(sharpes_ma):.3f}")


ma_returns.to_csv('examples/pbo_ma_sweep.csv', index=False)
print(f"\n Saved: examples/pbo_ma_sweep.csv")
print(f"   Expected PBO: 30-40% (realistic parameter optimization)")

# =============================================================================
# Example 2: Random Noise - SUPER HIGH PBO
# =============================================================================

print("\n" + "="*70)
print("2. RANDOM NOISE (50 strategies)")
print("="*70)

n_trials = 50
noise_returns = pd.DataFrame({'date': dates})

print(f"Generating {n_trials} random strategies (zero edge)...")

for i in range(n_trials):
    # Pure rubbish - zero E[return]
    strategy_returns = np.random.normal(0.0, 0.008, N_DAYS)
    noise_returns[f'random_{i+1}'] = strategy_returns

sharpes_noise = []
for col in noise_returns.columns[1:]:
    sr = sharpe_ratio(noise_returns[col].values)
    sharpes_noise.append(sr)

sharpes_noise = np.array(sharpes_noise)

print(f"\nRandom strategy Sharpes:")
print(f"  Mean:   {np.mean(sharpes_noise):.3f}")
print(f"  Std:    {np.std(sharpes_noise):.3f}")
print(f"  Min:    {np.min(sharpes_noise):.3f}")
print(f"  Max:    {np.max(sharpes_noise):.3f}")

noise_returns.to_csv('examples/pbo_random_noise.csv', index=False)
print(f"\nSaved: examples/pbo_random_noise.csv")
print(f"   Expected PBO: 50-60% (pure luck -> higher PBO)")

# =============================================================================
# Example 3: Mixed Strategies - MODERATE PBO
# =============================================================================

print("\n" + "="*70)
print("3. MIXED STRATEGIES (20 strategies: 5 real + 15 noise)")
print("="*70)

n_real = 5
n_noise = 15
mixed_returns = pd.DataFrame({'date': dates})

print(f"Generating {n_real} strategies with real edge...")
# Real strategies (modest positive edge)
for i in range(n_real):
    strategy_mean = 0.0003  # ~7.5% annual
    strategy_vol = 0.011
    strategy_returns = np.random.normal(strategy_mean, strategy_vol, N_DAYS)
    
    # Add autocorrelation
    for j in range(1, N_DAYS):
        strategy_returns[j] += 0.1 * strategy_returns[j-1]
    
    mixed_returns[f'real_{i+1}'] = strategy_returns

print(f"Generating {n_noise} noise strategies...")
# Noise strategies (zero edge)
for i in range(n_noise):
    strategy_returns = np.random.normal(0.0, 0.011, N_DAYS)
    mixed_returns[f'noise_{i+1}'] = strategy_returns

sharpes_mixed = []
for col in mixed_returns.columns[1:]:
    sr = sharpe_ratio(mixed_returns[col].values)
    sharpes_mixed.append(sr)

sharpes_mixed = np.array(sharpes_mixed)

print(f"\nMixed strategy Sharpes:")
print(f"  Mean:   {np.mean(sharpes_mixed):.3f}")
print(f"  Min:    {np.min(sharpes_mixed):.3f}")
print(f"  Max:    {np.max(sharpes_mixed):.3f}")

mixed_returns.to_csv('examples/pbo_mixed_strategies.csv', index=False)
print(f"\n Saved: examples/pbo_mixed_strategies.csv")
print(f"   Expected PBO: 35-45% (moderate risk)")
