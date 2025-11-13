import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def estimate_kappa(stage2):
    """
    Estimate kappa from the ALM:
    r_{t+1} = log(eps_{t+1}) + log(1 - kappa * exp(pred_t)) - log(1 - kappa * exp(pred_{t+1}))
    """

    r = stage2['vwretd'].values[1:]
    pred_t = stage2['predictions'].values[:-1]
    pred_t1 = stage2['predictions'].values[1:]

    def objective(kappa):
        kappa = float(kappa)
        if np.any(1 - kappa * np.exp(pred_t) <= 0) or np.any(1 - kappa * np.exp(pred_t1) <= 0):
            return 1e10
        r_hat = np.log(1 - kappa * np.exp(pred_t)) - np.log(1 - kappa * np.exp(pred_t1))
        val = np.sum((r - r_hat) ** 2)
        return np.inf if not np.isfinite(val) else val

    # Compute safe upper bound
    exp_max = np.exp(np.max([pred_t.max(), pred_t1.max()]))
    kappa_max = min(0.99 / exp_max, 1.0)  # ensure < 1 to keep logs positive
    bounds = [(1e-6, kappa_max)]

    # Try multiple initial guesses
    guesses = np.linspace(bounds[0][0], bounds[0][1], 5)
    best_val, best_kappa = np.inf, np.nan

    for guess in guesses:
        res = minimize(objective, x0=guess, bounds=bounds, method="L-BFGS-B")
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best_val, best_kappa = res.fun, res.x[0]

    return best_kappa


def estimate_kappa_curve_fit(stage2):
    """
    Alternative method to estimate kappa using curve fitting.
    """


    r = stage2['vwretd'].values[1:]
    pred_t = stage2['predictions'].values[:-1]
    pred_t1 = stage2['predictions'].values[1:]

    def alm_model(x, kappa, intercept):
        pred_t, pred_t1 = x
        return np.log(1 - kappa * np.exp(pred_t)) - np.log(1 - kappa * np.exp(pred_t1)) + intercept



    # Initial guess and bounds
    initial_kappa = 0.8
    initial_intercept = 0.0
    bounds = ([0, -1], [1, 1])  # first bracket is for the lower bounds, second for the upper bounds

    try:
        popt, pcov = curve_fit(alm_model, (pred_t, pred_t1), r,
                            p0=[initial_kappa, initial_intercept], bounds=bounds)
        return popt, pcov
    except Exception as e:
        raise RuntimeError("Curve fitting failed: " + str(e))
