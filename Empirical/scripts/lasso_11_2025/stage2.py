import numpy as np
from scipy.optimize import minimize

def estimate_kappa(stage2):
    """
    Estimate kappa from the ALM:
    r_{t+1} = log(eps_{t+1}) + log(1 - kappa * exp(pred_t)) - log(1 - kappa * exp(pred_{t+1}))
    """

    # align data for t and t+1
    r = stage2['vwretd'].values[1:]                 # r_{t+1}
    eps = stage2['epsilon'].values[1:]              # ε_{t+1}
    pred_t = stage2['predictions'].values[:-1]      # x'_t β
    pred_t1 = stage2['predictions'].values[1:]      # x'_{t+1} β

    def objective(kappa):
        if np.any(1 - kappa * np.exp(pred_t) <= 0) or np.any(1 - kappa * np.exp(pred_t1) <= 0):
            return np.inf  # ensure valid domain for log
        r_hat = np.log(eps**2) + np.log(1 - kappa * np.exp(pred_t)) - np.log(1 - kappa * np.exp(pred_t1))
        return np.sum((r - r_hat)**2)

    # initial guess and bounds
    res = minimize(objective, x0=0.5, bounds=[(1e-6, 1-1e-6)], method='L-BFGS-B')

    return res.x[0]


