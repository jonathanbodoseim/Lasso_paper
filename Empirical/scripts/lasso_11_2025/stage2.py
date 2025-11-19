import numpy as np
from scipy.optimize import minimize, curve_fit
from stage1 import calculate_r_squared


def estimate_kappa_curve_fit(stage2):
    """
    Estimate kappa and intercept using scipy's curve_fit.
    
    Returns
    -------
    popt : array
        Optimal parameters [kappa, intercept]
    pcov : 2D array
        Covariance matrix of parameters
    """
    r = stage2['vwretd'].values[1:]
    pred_t = stage2['predictions'].values[:-1]
    pred_t1 = stage2['predictions'].values[1:]
    
    def alm_model(x, kappa, intercept):
        pred_t, pred_t1 = x
        return np.log(1 - kappa * np.exp(pred_t)) - np.log(1 - kappa * np.exp(pred_t1)) + intercept
    
    try:
        return curve_fit(
            alm_model, 
            (pred_t, pred_t1), 
            r,
            p0=[0.8, 0.0],
            bounds=([0, -1], [1, 1])
        )
    except Exception as e:
        raise RuntimeError(f"Curve fitting failed: {e}")


def compute_alm_returns(predictions, kappa, intercept):
    pred_t, pred_t1 = predictions[:-1], predictions[1:]

    # validity mask
    valid = (1 - kappa * np.exp(pred_t) > 0) & (1 - kappa * np.exp(pred_t1) > 0)

    # initialize output array
    alm = np.full(len(pred_t), np.nan)

    # compute only valid entries
    alm[valid] = (
        np.log(1 - kappa * np.exp(pred_t[valid]))
        - np.log(1 - kappa * np.exp(pred_t1[valid]))
        + intercept
    )

    return alm



def compute_stage2_r_squared(stage2_input, min_train_size=100):
    """
    Compute in-sample and out-of-sample R² for Stage 2.
    
    In-sample R²:
    - Estimate kappa and intercept on full sample
    - Calculate fitted values on full sample
    - Compute R²
    
    Out-of-sample R²:
    - Use expanding window: for each time t, estimate kappa and intercept on data up to t-1
    - Predict return at time t using this kappa
    - Compute R² on all OOS predictions
    
    Parameters
    ----------
    stage2_input : pd.DataFrame
        Must have columns 'vwretd' (actual returns) and 'predictions' (Stage 1 predictions)
    min_train_size : int
        Minimum number of observations needed to estimate kappa
        
    Returns
    -------
    dict
        Contains r2_insample, r2_oos, kappa_full, intercept_full, and their t-stats
    """
    
    # ===== IN-SAMPLE R² =====
    # Estimate kappa on FULL sample
    try:
        popt_full, pcov_full = estimate_kappa_curve_fit(stage2_input)
        kappa_full, intercept_full = popt_full
        se_full = np.sqrt(np.diag(pcov_full))
        
        # Handle zero or near-zero standard errors
        if se_full[0] < 1e-10:
            kappa_tstat = np.nan  # Can't compute t-stat
        else:
            kappa_tstat = kappa_full / se_full[0]
        
        if se_full[1] < 1e-10:
            intercept_tstat = np.nan
        else:
            intercept_tstat = intercept_full / se_full[1]
        
        # Generate fitted values on FULL sample
        preds_full = stage2_input['predictions'].values
        alm_fitted = compute_alm_returns(preds_full, kappa_full, intercept_full)
        
        # Calculate in-sample R²
        y_full = stage2_input['vwretd'].values[1:len(alm_fitted)+1]
        
        # Check if we have valid fitted values
        if len(alm_fitted) == 0:
            return {
                'r2_insample': np.nan,
                'r2_oos': np.nan,
                'kappa': kappa_full,
                'kappa_tstat': kappa_tstat,
                'intercept': intercept_full,
                'intercept_tstat': intercept_tstat,
                'error': "No valid ALM fitted values (constraint violations)"
            }
        
        r2_insample = calculate_r_squared(y_full, alm_fitted)
        
    except Exception as e:
        return {
            'r2_insample': np.nan,
            'r2_oos': np.nan,
            'kappa': np.nan,
            'kappa_tstat': np.nan,
            'intercept': np.nan,
            'intercept_tstat': np.nan,
            'error': f"Full sample estimation failed: {e}"
        }
    
    # ===== OUT-OF-SAMPLE R² =====
    # Use expanding window to generate true OOS predictions
    n_obs = len(stage2_input)
    oos_predictions = []
    oos_actuals = []
    
    # Start predicting after we have enough data to estimate kappa
    for t in range(min_train_size, n_obs - 1):  # -1 because we need t+1 for prediction
        # Training data: everything up to time t-1
        train_data = stage2_input.iloc[:t].copy()
        
        try:
            # Estimate kappa on training data only
            popt_train, _ = estimate_kappa_curve_fit(train_data)
            kappa_train, intercept_train = popt_train
            
            # Make prediction for time t (this is out-of-sample!)
            # We need predictions at t-1 and t to compute the ALM return at t
            pred_t_minus_1 = stage2_input['predictions'].iloc[t-1]
            pred_t = stage2_input['predictions'].iloc[t]
            
            # Check validity
            if (1 - kappa_train * np.exp(pred_t_minus_1) > 0) and \
               (1 - kappa_train * np.exp(pred_t) > 0):
                
                # Compute OOS prediction for return at time t
                r_hat_t = (np.log(1 - kappa_train * np.exp(pred_t_minus_1)) - 
                          np.log(1 - kappa_train * np.exp(pred_t)) + 
                          intercept_train)
                
                oos_predictions.append(r_hat_t)
                oos_actuals.append(stage2_input['vwretd'].iloc[t])
                
        except Exception:
            # Skip this observation if estimation fails
            continue
    
    # Calculate OOS R²
    if len(oos_predictions) > 0:
        oos_predictions = np.array(oos_predictions)
        oos_actuals = np.array(oos_actuals)
        r2_oos = calculate_r_squared(oos_actuals, oos_predictions)
    else:
        r2_oos = np.nan
    
    return {
        'r2_insample': r2_insample,
        'r2_oos': r2_oos,
        'kappa': kappa_full,
        'kappa_tstat': kappa_tstat,
        'intercept': intercept_full,
        'intercept_tstat': intercept_tstat,
        'n_oos_predictions': len(oos_predictions)
    }