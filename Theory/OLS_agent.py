import numpy as np
from scipy import stats

class OLSAgent:
    def __init__(self, n: int, alpha: float = 0.05):
        self.n = n  # number of past observations to use
        self.alpha = alpha  # significance level

    def predict_price(self, p_history: np.ndarray, theta_history: np.ndarray) -> float:
        """
        Predict next price based on OLS of past returns on theta.
        
        Parameters:
            p_history (np.ndarray): historical prices (length T)
            theta_history (np.ndarray): historical theta values (length T)
            
        Returns:
            float: predicted price for next period
        """
        T = len(p_history)
        if T < self.n + 1:
            raise ValueError(f"Not enough data. Require at least {self.n + 1} price points.")

        # Compute returns: r_t = (p_t - p_{t-1}) / p_{t-1}
        returns = (p_history[1:] - p_history[:-1]) / p_history[:-1]
        
        # Select last n observations
        X = theta_history[-self.n - 1:-1]  # lagged theta
        y = returns[-self.n:]

        # Add intercept for OLS
        X_mat = np.vstack([np.ones(self.n), X]).T

        # OLS estimate: beta = (X'X)^{-1} X'y
        beta_hat = np.linalg.lstsq(X_mat, y, rcond=None)[0]

        # Compute standard errors
        residuals = y - X_mat @ beta_hat
        s2 = np.sum(residuals**2) / (self.n - 2)
        cov_beta = s2 * np.linalg.inv(X_mat.T @ X_mat)
        se_beta_1 = np.sqrt(cov_beta[1, 1])

        # t-statistic for beta_1
        t_stat = beta_hat[1] / se_beta_1
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=self.n - 2))

        # Get latest price and theta
        p_last = p_history[-1]
        theta_last = theta_history[-1]

        # Predict next price
        if p_value < self.alpha:
            return (1 + beta_hat[1] * theta_last) * p_last
        else:
            return p_last
