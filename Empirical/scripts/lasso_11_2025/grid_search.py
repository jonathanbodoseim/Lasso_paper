# =======================
#        IMPORTS
# =======================

import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm

# Update this line to match your project structure:
from stage1 import (
    lasso_rolling_window,
    calculate_r_squared
)

from stage2 import compute_stage2_r_squared

# =======================
#      FUNCTIONS
# =======================

def estimate_single_config(X, y, window_size, n_lags, lambda_val):
    """
    Estimate 1st and 2nd stage for a single configuration and return all metrics.
    
    Returns
    -------
    dict
        Dictionary containing all performance metrics, or None if estimation fails
    """
    try:
        # Stage 1: Rolling LASSO
        lasso_results = lasso_rolling_window(
            X=X, y=y, 
            window_size=window_size, 
            n_lags=n_lags,
            lambda_mode="fixed", 
            fixed_lambda=lambda_val, 
            verbose=True
        )
        
        # Extract predictions and align data
        preds = np.array(lasso_results["predictions"])
        y_valid = y[-len(preds):] 
        y_vals = y_valid.values 
        
        # Stage 1
        r2_oos_stage1 = calculate_r_squared(y_vals, preds)
        r2_insample_stage1 = np.mean(lasso_results['insample_r_squareds'])
        
        # Stage 2
        stage2_input = pd.DataFrame({"vwretd": y_vals, "predictions": preds})
        stage2_results = compute_stage2_r_squared(stage2_input, min_train_size=100)
        
        return {
            'window_size': window_size,
            'n_lags': n_lags,
            'lambda': lambda_val,
            'r2_insample_stage1': r2_insample_stage1,
            'r2_oos_stage1': r2_oos_stage1,
            'r2_insample_stage2': stage2_results['r2_insample'],
            'r2_oos_stage2': stage2_results['r2_oos'],
            'kappa': stage2_results['kappa'],
            'kappa_tstat': stage2_results['kappa_tstat'],
            'intercept': stage2_results['intercept'],
            'intercept_tstat': stage2_results['intercept_tstat'],
            'n_observations': len(y_vals),
            'n_windows': len(lasso_results['predictions']),
            'n_oos_predictions_stage2': stage2_results.get('n_oos_predictions', np.nan)
        }
        
    except Exception as e:
        return {
            'window_size': window_size,
            'n_lags': n_lags,
            'lambda': lambda_val,
            'r2_insample_stage1': np.nan,
            'r2_oos_stage1': np.nan,
            'r2_insample_stage2': np.nan,
            'r2_oos_stage2': np.nan,
            'kappa': np.nan,
            'kappa_tstat': np.nan,
            'intercept': np.nan,
            'intercept_tstat': np.nan,
            'n_observations': np.nan,
            'n_windows': np.nan,
            'n_oos_predictions_stage2': np.nan,
            'error': str(e)
        }


def grid_search(X, y, param_grid, verbose=True):
    """
    Perform grid search over all parameter combinations.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray or pd.Series
        Target variable (returns)
    param_grid : dict
        Dictionary with keys 'window_sizes', 'n_lags', 'lambdas'
        Each value should be a list of values to try
    verbose : bool
        Whether to show progress bar
        
    Returns
    -------
    pd.DataFrame
        Results for all configurations, sorted by OOS R² Stage 2

    """

    
    # Create all combinations
    combinations = list(product(
        param_grid['window_sizes'],
        param_grid['n_lags'],
        param_grid['lambdas']
    ))
    
    if verbose:
        print(f"Testing {len(combinations)} configurations...")
        print(f"Window sizes: {param_grid['window_sizes']}")
        print(f"N lags: {param_grid['n_lags']}")
        print(f"Lambdas: {param_grid['lambdas']}")
    
    # Run grid search
    results = []
    iterator = tqdm(combinations, desc="Grid search") if verbose else combinations
    
    for window_size, n_lags, lambda_val in iterator:
        result = estimate_single_config(X, y, window_size, n_lags, lambda_val)
        results.append(result)
    
    # Convert to DataFrame and sort by OOS R² Stage 2
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2_oos_stage2', ascending=False)
    
    if verbose:
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        n_failed = results_df['kappa'].isna().sum()
        if n_failed > 0:
            print(f"⚠️  {n_failed}/{len(results_df)} configurations failed")
    
    return results_df