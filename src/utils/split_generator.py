"""
The Train/Test Split Generator for CSCV

Generates all possible combinatorially symmetric splits

Based on: Bailey, Borwein, López de Prado (2013)
"The Probability of Backtest Overfitting"
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Literal

def split_into_groups(
    data: np.ndarray,
    n_groups: int,
    mode: Literal['contiguous', 'interleaved'] = 'contiguous'
) -> List[np.ndarray]:

    """
    Split data into n_groups equal-sized groups.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data to split (the returns here)
    n_groups : int
        Number of groups to create
    mode : {'contiguous', 'interleaved'}, default 'contiguous'
        
    Returns
    -------
    list of np.ndarray
        Each element is one group
        
    """
    n = len(data)

    if mode == 'contiguous':
        group_size = n // n_groups
        groups = []

        for i in range(n_groups):
            start_idx = i * group_size

            if i == n_groups -1:
                end_idx = n
            else:
                end_idx = (i+1) * group_size
            
            groups.append(data[start_idx:end_idx])

    elif mode == 'interleaved':
        groups = [[] for _ in range(n_groups)]

        for i, value in enumerate(data):
            group_idx = i % n_groups
            groups[group_idx].append(value)

        groups = [np.array(g) for g in groups]

    return groups

def generate_cscv_splits(n_groups: int) -> List[Tuple[tuple, tuple]]:
    """
    Generate all CSCV train-test split combinations
    
    Implements the combinatorially symmetric split structure from
    Bailey et al. (2014). Each split uses exactly half the groups
    for training and half for testing.
    
    Parameters
    ----------
    n_groups : int
        Total number of groups (must be even)
        
    Returns
    -------
    list of tuples
        Each element is (train_indices, test_indices)
        Length = C(n_groups, n_groups/2) combinations
    """
    if n_groups % 2 != 0:
        raise ValueError("Need an even number of groups")

    if n_groups < 4:
        raise ValueError("Need more than 4 groups")

    train_size = n_groups // 2

    all_group_indicies = range(n_groups)
    all_train_combinations = list(combinations(all_group_indicies, train_size))

    splits = []
    for train_idx in all_train_combinations:
        test_idx = tuple(i for i in all_group_indicies if i not in train_idx)
        splits.append((train_idx, test_idx))

    return splits

def apply_split(
    groups: List[np.ndarray],
    train_indicies: tuple,
    test_indicies: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a train-test split to data groups
    
    Parameters
    ----------
    groups : list of np.ndarray
        Data split into groups (from split_into_groups)
    train_indices : tuple
        Indices of groups to use for training
    test_indices : tuple
        Indices of groups to use for testing
        
    Returns
    -------
    train_data : np.ndarray
        Concatenated training data
    test_data : np.ndarray
        Concatenated testing data
    """
    if not train_indicies or not test_indicies:
        raise ValueError("train or test indicies can't be empty")

    if max(train_indicies) >= len(groups) or max(test_indicies) >= len(groups):
        raise ValueError("Invalid group indicies for this group length")
    
    train_data = np.concatenate([groups[i] for i in train_indicies])
    test_data = np.concatenate([groups[i] for i in test_indicies])

    return train_data, test_data

def count_splits(n_groups: int) -> int:
    """
    Count number of CSCV splits without generating them
    
    Parameters
    ----------
    n_groups : int
        Number of groups
        
    Returns
    -------
    int
        Number of train/test combinations = C(n_groups, n_groups/2)
    """
    from math import comb
    return comb(n_groups, n_groups//2)
