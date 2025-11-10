"""
Lasso Rolling Window Implementation
====================================

This module implements rolling window LASSO regression for time series analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Optional, Dict, Any, List


def create_lagged_features(X: np.ndarray, 
                          n_lags: int = 3,
                          include_contemporaneous: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Create lagged features from predictor matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Matrix of predictors with shape (n_samples, n_features)
    n_lags : int
        Number of lags to create for each predictor
    include_contemporaneous : bool
        Whether to include the contemporaneous (lag 0) values
        
    Returns
    -------
    tuple
        (X_lagged, feature_names) where X_lagged is the matrix with lagged features
        
    Notes
    -----
    For a model y_t = x_{t-1} + x_{t-2} + x_{t-3}, we need lags 1, 2, and 3
    of the predictors. The response y at time t is predicted using past values
    of X only (no look-ahead bias).
    """
    n_samples, n_features = X.shape
    
    # Determine which lags to use
    if include_contemporaneous:
        lag_range = range(0, n_lags + 1)
    else:
        lag_range = range(1, n_lags + 1)
    
    # Create lagged features
    lagged_features = []
    feature_names = []
    
    for lag in lag_range:
        if lag == 0:
            # Contemporaneous values (use with caution - risk of look-ahead bias)
            lagged_features.append(X[n_lags:])
            feature_names.extend([f'feature_{i}_lag_0' for i in range(n_features)])
        else:
            # Lagged values: shift data forward by 'lag' periods
            # X[n_lags-lag:-lag] gives us the lagged values aligned with y[n_lags:]
            lagged_features.append(X[n_lags-lag:-lag if lag > 0 else None])
            feature_names.extend([f'feature_{i}_lag_{lag}' for i in range(n_features)])
    
    # Stack all lagged features horizontally
    X_lagged = np.hstack(lagged_features)
    
    return X_lagged, feature_names


def lasso_rolling_window(X: np.ndarray,
                         y: np.ndarray,
                         window_size: int = 30,
                         n_lags: int = 3,
                         burn_in: Optional[int] = None,
                         standardize: bool = True,
                         cv_folds: int = 5,
                         alphas: Optional[np.ndarray] = None,
                         include_contemporaneous: bool = False,
                         verbose: bool = True) -> Dict[str, Any]:

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    import warnings
    from tqdm import tqdm  


    # --- detect if pandas object and extract index
    if isinstance(y, (pd.Series, pd.DataFrame)):
        date_index = y.index
    else:
        date_index = None

    # Ensure numpy arrays for computation
    y = np.asarray(y).flatten()
    X = np.asarray(X)

    n_samples, n_features = X.shape

    if verbose:
        print(f"Creating lagged features with {n_lags} lags...")

    X_lagged, feature_names = create_lagged_features(X, n_lags, include_contemporaneous)
    y_aligned = y[n_lags:]

    # Align date index (if available)
    if date_index is not None:
        date_index = date_index[n_lags:]

    if burn_in is None:
        burn_in = window_size

    if len(y_aligned) < burn_in:
        raise ValueError("Not enough data for specified window size and lags.")

    lambdas, coefficients, window_starts, window_ends = [], [], [], []
    predictions, intercepts = [], []

    # --- NEW: date containers
    window_start_dates, window_end_dates, prediction_dates = [], [], []

    n_windows = len(y_aligned) - window_size + 1
    if verbose:
        print(f"Running {n_windows} rolling windows of size {window_size}...")

    for i in tqdm(range(burn_in - window_size, n_windows), desc="Rolling windows"):
        start_idx = max(0, i)
        end_idx = start_idx + window_size

        X_window = X_lagged[start_idx:end_idx]
        y_window = y_aligned[start_idx:end_idx]

        if standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window)
        else:
            scaler = None
            X_window_scaled = X_window

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                lasso_cv = LassoCV(cv=min(cv_folds, window_size // 2),
                                   max_iter=2000,
                                   n_alphas=100,
                                   fit_intercept=True)
                lasso_cv.fit(X_window_scaled, y_window)

                lambdas.append(lasso_cv.alpha_)
                intercepts.append(lasso_cv.intercept_)

                if standardize:
                    coef_original = lasso_cv.coef_ / scaler.scale_
                else:
                    coef_original = lasso_cv.coef_
                coefficients.append(coef_original)

                window_starts.append(start_idx + n_lags)
                window_ends.append(end_idx + n_lags)

                # --- NEW: store corresponding dates if available
                if date_index is not None:
                    window_start_dates.append(date_index[start_idx])
                    window_end_dates.append(date_index[end_idx - 1])

                # OOS prediction
                if end_idx < len(y_aligned):
                    X_next = X_lagged[end_idx:end_idx + 1]
                    X_next_scaled = scaler.transform(X_next) if standardize else X_next
                    pred = lasso_cv.predict(X_next_scaled)
                    predictions.append(pred[0])

                    if date_index is not None:
                        prediction_dates.append(date_index[end_idx])

        except Exception as e:
            if verbose:
                print(f"Warning: LASSO failed for window {i}: {e}")
            lambdas.append(np.nan)
            coefficients.append(np.full(X_lagged.shape[1], np.nan))
            intercepts.append(np.nan)
            window_starts.append(start_idx + n_lags)
            window_ends.append(end_idx + n_lags)
            if date_index is not None:
                window_start_dates.append(date_index[start_idx])
                window_end_dates.append(date_index[min(end_idx - 1, len(date_index) - 1)])
            if end_idx < len(y_aligned):
                predictions.append(np.nan)
                if date_index is not None:
                    prediction_dates.append(date_index[min(end_idx, len(date_index) - 1)])

    if verbose:
        print("Rolling LASSO complete!")
        print(f"Average lambda: {np.nanmean(lambdas):.6f}")

    # --- include date tracking in results
    results = {
        'lambdas': np.array(lambdas),
        'coefficients': np.array(coefficients),
        'intercepts': np.array(intercepts),
        'window_starts': np.array(window_starts),
        'window_ends': np.array(window_ends),
        'feature_names': feature_names,
        'predictions': np.array(predictions),
        'n_lags': n_lags,
        'window_size': window_size,
    }

    if date_index is not None:
        results.update({
            'window_start_dates': np.array(window_start_dates),
            'window_end_dates': np.array(window_end_dates),
            'prediction_dates': np.array(prediction_dates)
        })

    return results


















def analyze_results(results: Dict[str, Any], 
                   feature_names: Optional[list] = None) -> pd.DataFrame:
    """
    Analyze and summarize the results from rolling LASSO.
    
    Parameters
    ----------
    results : dict
        Output from lasso_rolling_window function
        
    feature_names : list, optional
        Original feature names (before lagging)
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for each feature's coefficients
    """
    coefficients = results['coefficients']
    lagged_feature_names = results['feature_names']
    
    # Create DataFrame with coefficient evolution
    coef_df = pd.DataFrame(coefficients, columns=lagged_feature_names)
    
    # Calculate summary statistics
    summary = pd.DataFrame({
        'mean_coef': coef_df.mean(),
        'std_coef': coef_df.std(),
        'min_coef': coef_df.min(),
        'max_coef': coef_df.max(),
        'pct_nonzero': (coef_df != 0).mean() * 100,
        'pct_positive': (coef_df > 0).mean() * 100,
        'pct_negative': (coef_df < 0).mean() * 100
    })
    
    return summary


def get_coefficient_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert coefficients array to a DataFrame with proper column names.
    
    Parameters
    ----------
    results : dict
        Output from lasso_rolling_window function
        
    Returns
    -------
    pd.DataFrame
        DataFrame with coefficients for each window
    """
    coefficients = results['coefficients']
    feature_names = results['feature_names']
    window_starts = results['window_starts']
    
    # Create DataFrame
    coef_df = pd.DataFrame(coefficients, columns=feature_names)
    coef_df['window_start'] = window_starts
    coef_df['lambda'] = results['lambdas']
    coef_df['intercept'] = results.get('intercepts', np.nan)
    
    return coef_df


# Alias for backward compatibility (both names work)
rolling_window_lasso = lasso_rolling_window