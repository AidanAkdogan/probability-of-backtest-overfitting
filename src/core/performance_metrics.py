import numpy as np
from typing import Union, Dict

# Constants
TRADING_DAYS_PER_YEAR = 252
TRADING_MONTHS_PER_YEAR = 12

def sharpe_ratio(
    returns: Union[np.ndarray, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    A function to calculate the annualised Sharpe Ratio

    Sharpe = (return - rfr)/volatility [Higher is better, measures risk adjusted return]

    Parameters
    ----------
    returns : array-like
        Time series of returns (e.g., daily returns)
        Example: [0.01, -0.005, 0.02, ...] means 1%, -0.5%, 2% returns
    
    risk_free_rate : float, default 0.0
        Annual risk-free rate (e.g., 0.02 for 2% T-bill rate)
    
    periods_per_year : int, default 252
        Number of trading periods per year
        - 252 for daily returns
        - 12 for monthly returns
        - 52 for weekly returns
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """

    # Cast to a numpy array with no NaN values
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return np.nan

    # the risk free rate per period
    rf_per_period = risk_free_rate/periods_per_year

    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1) # 1 df for sample std

    if std_excess == 0:
        return np.nan
        #handles divode by zero edge case
    
    sharpe = mean_excess / std_excess

    annualised_sharpe = sharpe * np.sqrt(periods_per_year)

    return annualised_sharpe

def sortino_ratio(
    returns: Union[np.ndarray, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:

    """
    Calculate annualized Sortino ratio.
    
    Like Sharpe ratio, but only penalizes DOWNSIDE volatility.
    Better for strategies with positive skew (small losses, big wins).
    
    Parameters
    ----------
    returns : array-like
        Time series of returns
    risk_free_rate : float, default 0.0
        Annual risk-free rate
    periods_per_year : int, default 252
        Trading periods per year
    
    Returns
    -------
    float
        Annualized Sortino ratio
        
    Notes
    -----
    Sortino only uses negative returns to calculate risk.
    This is more intuitive: we don't care about upside "risk"!
    
    Sortino >= Sharpe (always)
    Why? Sortino has smaller denominator (only downside vol).
    """

    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return np.nan

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf

    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return np.nan
    
    # sortino = mean / downside_std
    mean_excess = np.mean(excess_returns)
    sortino = mean_excess / downside_std
    
    annualised_sortino = sortino * np.sqrt(periods_per_year)
    
    return annualised_sortino


def max_drawdown(returns : Union[np.ndarray, list]) -> float:
    """
    Calculate maximum drawdown.
    
    Max drawdown = worst peak-to-trough decline an investor experienced.
    
    Parameters
    ----------
    returns : array-like
        Time series of returns
    
    Returns
    -------
    float
        Maximum drawdown as positive number (e.g., 0.25 for -25% drawdown)
        
    Examples
    --------
    Portfolio value: $100k → $120k → $90k → $110k
    Peak: $120k
    Trough: $90k
    Max Drawdown: 25% = ($120k - $90k) / $120k
    
    Notes
    -----
    This is what clients feel in their gut.
    A strategy with Sharpe 2.0 but 50% drawdown will lose clients!
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return np.nan
    
    cumulative_returns = np.cumprod(1 + returns)

    # gets highest value seen so far at that point in time
    running_max = np.maximum.accumulate(cumulative_returns)

    drawdown = (cumulative_returns - running_max) / running_max

    return abs(np.min(drawdown))

def calmar_ratio(
    returns: Union[np.ndarray, list],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Calmar ratio.
    
    Calmar = Annual Return / Max Drawdown
    
    Measures return per unit of worst-case loss.
    Better than Sharpe for strategies with fat tails (rare but big losses).
    
    Parameters
    ----------
    returns : array-like
        Time series of returns
    periods_per_year : int, default 252
        Trading periods per year
    
    Returns
    -------
    float
        Calmar ratio
        
    Notes
    -----
    Hedge funds often report Calmar because:
    - Investors care about drawdowns
    - Max DD is easy to understand
    - Better for non-normal returns than Sharpe
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return np.nan

    annual_return = np.mean(returns) * periods_per_year

    mdd = max_drawdown(returns)

    if mdd == 0:
        return np.inf
    
    calmar = annual_return / mdd
    return calmar

def calculate_all_metrics(
    returns: Union[np.ndarray, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, float]:
    """
    Calculate all performance metrics at once.
    
    Convenience function that returns a dictionary of all metrics.
    Useful for generating reports or comparing strategies.
    
    Parameters
    ----------
    returns : array-like
        Time series of returns
    risk_free_rate : float, default 0.0
        Annual risk-free rate
    periods_per_year : int, default 252
        Trading periods per year
    
    Returns
    -------
    dict
        Dictionary with all metrics:
        - sharpe_ratio
        - sortino_ratio  
        - max_drawdown
        - calmar_ratio
        - annual_return
        - annual_volatility
        - win_rate (% of positive returns)
        - num_observations
        
    Examples
    --------
    >>> returns = [0.01, -0.005, 0.02, ...]
    >>> metrics = calculate_all_metrics(returns)
    >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max DD: {metrics['max_drawdown']:.1%}")
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    
    # Calculate all metrics
    metrics = {
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'annual_return': np.mean(returns) * periods_per_year,
        'annual_volatility': np.std(returns, ddof=1) * np.sqrt(periods_per_year),
        'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0,
        'num_observations': len(returns)
    }
    
    return metrics
